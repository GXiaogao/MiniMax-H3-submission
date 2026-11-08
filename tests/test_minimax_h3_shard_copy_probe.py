import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/infinifabric-build-v1'


class ShardCopyProbeTests(unittest.TestCase):
    def setUp(self):
        path=FOLDER/'probe_minimax_h3_shard_copy.py'
        self.assertTrue(path.exists(),'copy probe missing')
        spec=importlib.util.spec_from_file_location('shard_copy_test',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_copy_controls_retain_materialized_source_and_sync_only_when_requested(self):
        events=[]
        class Tensor:
            def __init__(self,tag):self.tag=tag
            def narrow(self,dim,start,width):events.append(('narrow',dim,start,width));return Tensor('view')
            def contiguous(self):events.append(('materialize',));return Tensor('packed')
            def copy_(self,source):events.append(('copy',source.tag))
        class Core:
            bfloat16='BF16'
            @staticmethod
            def device(kind,rank):
                if kind!='cuda':raise KeyError(kind)
                return rank
            @staticmethod
            def empty(shape,**kwargs):events.append(('empty',shape,kwargs));return Tensor('destination')
            @staticmethod
            def set_device(rank):events.append(('set',rank))
            @staticmethod
            def sync_stream():events.append(('sync',))
        for mode in self.m.MODES:
            events.clear()
            rank,dst,keep=self.m.enqueue_copy(Core,Tensor('source'),[4,16],'BF16',3,mode)
            self.assertEqual(rank,3)
            self.assertEqual(events[0],('narrow',1,6,2))
            if mode=='implicit':
                self.assertIsNone(keep);self.assertIn(('copy','view'),events)
                self.assertNotIn(('materialize',),events)
            else:
                self.assertEqual(keep.tag,'packed');self.assertIn(('copy','packed'),events)
            if mode=='retained_source_sync':
                self.assertLess(events.index(('sync',)),events.index(('copy','packed')))
                self.assertIn(('set',0),events)
            else:self.assertNotIn(('sync',),events)

    def test_full_comparison_detects_unsampled_difference_and_wrong_shard(self):
        expected=np.arange(90000,dtype=np.float32).reshape(300,300)
        actual=expected.copy();actual[0,1]+=1
        metric=self.m.compare_full(expected,actual)
        self.assertFalse(metric['exact_equal']);self.assertEqual(metric['mismatched_elements'],1)
        self.assertEqual(metric['elements'],90000);self.assertEqual(metric['scope'],'all-elements')
        with self.assertRaises(ValueError):self.m.compare_full(expected,actual[:1])
        actual[0,0]=np.nan
        with self.assertRaises(ValueError):self.m.compare_full(expected,actual)

    def test_invalid_shapes_modes_and_ranks_rejected(self):
        for shape,rank,mode in [([2,15],0,'implicit'),([2,16],8,'implicit'),([2,16],0,'unknown')]:
            with self.assertRaises(ValueError):self.m.enqueue_copy(None,None,shape,None,rank,mode)

    def test_native_counts_do_not_instantiate_missing_cpu_or_hygon_python_device(self):
        def count(kind):return {'HYGON':8,'CPU':0}[kind]
        cuda=SimpleNamespace(device_count=lambda:8,get_device_name=lambda rank:'Hygon')
        torch=SimpleNamespace(__version__='fixture',version=SimpleNamespace(hip='6.3'),cuda=cuda)
        native=SimpleNamespace(__file__=__file__,get_device_count=count,
            Device=SimpleNamespace(Type=SimpleNamespace(HYGON='HYGON',CPU='CPU')))
        core=SimpleNamespace(__file__=__file__)
        environment=self.m.probe_environment(torch,core,native)
        self.assertEqual(environment['native_cpu_count'],0)
        self.assertEqual(environment['native_device_count'],8)

    def test_failed_probe_keeps_report(self):
        with tempfile.TemporaryDirectory() as d:
            rc=self.m.main(['--run-dir',d,'--work-home',d])
            import json
            data=json.loads((Path(d)/'shard-copy.json').read_text())
            self.assertEqual(rc,1);self.assertEqual(data['result'],'FAILED')
            self.assertFalse(data['performance_measurement_valid'])

    def test_runtime_readback_is_direct_d2h_and_decodes_bf16(self):
        import ctypes
        source=np.array([0x3f80,0xc000,0x0000,0x4040],dtype='<u2')
        calls=[]
        class Runtime:
            def infinirtSetDevice(self,kind,rank):calls.append(('device',kind,rank));return 0
            def infinirtDeviceSynchronize(self):calls.append(('sync',));return 0
            def infinirtMemcpy(self,dst,src,size,kind):
                calls.append(('copy',size,kind));ctypes.memmove(dst,src,size);return 0
        value=SimpleNamespace(shape=[2,2],is_contiguous=lambda:True,data_ptr=lambda:source.ctypes.data)
        result=self.m.readback(Runtime(),value,3,'bfloat16')
        np.testing.assert_array_equal(result,[[1.,-2.],[0.,3.]])
        self.assertEqual(calls,[('device',8,3),('sync',),('copy',8,2)])
        bad=Runtime();bad.infinirtMemcpy=lambda *a:7
        with self.assertRaisesRegex(RuntimeError,'7'):self.m.readback(bad,value,3,'bfloat16')

    def test_workers_stay_alive_through_sync_and_observation(self):
        events=[]
        class Pool:
            def __init__(self,**kw):pass
            def __enter__(self):events.append('enter');return self
            def __exit__(self,*a):events.append('exit')
            def map(self,fn,ranks):return [(r,'dst','keep') for r in ranks]
        def sync():events.append('sync')
        def observe(value,rank):events.append('observe');return rank
        with patch.object(self.m,'ThreadPoolExecutor',Pool):
            result=self.m.complete_batch(lambda r:None,sync,observe,lambda r:events.append('checkpoint'))
        self.assertEqual(result,list(range(8)))
        self.assertEqual(events[-1],'exit')
        self.assertLess(events.index('sync'),events.index('observe'))

    def test_child_signal_does_not_discard_other_group_results(self):
        report=dict(groups={},result='FAILED');checkpoints=[]
        def child(args,group):
            if group=='retained':return dict(returncode=-11,checkpoint=dict(last_stage='observe',active_rank=3))
            return dict(returncode=0,checkpoint=dict(result='PROBE_COMPLETE',cases=self.valid_cases(group)))
        with patch.object(self.m,'run_child',side_effect=child):
            self.m.orchestrate(SimpleNamespace(),report,lambda:checkpoints.append(len(report['groups'])))
        self.assertEqual(len(report['groups']),4)
        self.assertEqual(report['result'],'INCOMPLETE_CHILD_FAILURE')
        self.assertEqual(report['groups']['retained']['returncode'],-11)
        self.assertEqual(checkpoints[-1],4)

    def valid_cases(self,group):
        return [dict(shape=shape,dtype=dtype,epoch=epoch,mode=mode,
            ranks={str(r):dict(shape=[shape[0],shape[1]//8],elements=shape[0]*shape[1]//8,
                mismatched_elements=0,max_abs=0.,exact_equal=True,scope='all-elements') for r in range(8)},all_ranks_exact=True)
            for shape in ([128,1024],[5376,7168]) for dtype in ('float32','bfloat16') for epoch in range(3)
            for mode in (self.m.MODES if group=='mixed' else (group,))]

    def test_success_cannot_hide_duplicate_case_or_nonfinite_metric(self):
        cases=self.valid_cases('implicit')
        result=dict(returncode=0,checkpoint=dict(result='PROBE_COMPLETE',cases=cases))
        self.assertTrue(self.m.group_complete('implicit',result))
        original=cases[-1];cases[-1]=cases[0]
        self.assertFalse(self.m.group_complete('implicit',result))
        cases[-1]=original
        cases[0]['ranks']['2']['max_abs']=float('inf')
        self.assertFalse(self.m.group_complete('implicit',result))

    def test_real_child_nonzero_preserves_checkpoint_and_log(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);script=path/'stub.py'
            script.write_text("import sys,json\nfrom pathlib import Path\np=Path(sys.argv[sys.argv.index('--run-dir')+1])\n(p/'shard-copy.json').write_text(json.dumps(dict(last_stage='readback',active_rank=3)))\nprint('native-child-failed',flush=True)\nsys.exit(139)\n")
            with patch.object(self.m,'__file__',str(script)):
                result=self.m.run_child(SimpleNamespace(run_dir=path,work_home=path),'mixed')
            self.assertEqual(result['returncode'],139)
            self.assertEqual(result['checkpoint']['active_rank'],3)
            self.assertIn('native-child-failed',Path(result['log']).read_text())

    def test_child_timeout_is_bounded_and_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);script=path/'stub.py'
            script.write_text('import time\ntime.sleep(10)\n')
            with patch.object(self.m,'__file__',str(script)),patch.object(self.m,'CHILD_TIMEOUT_SECONDS',.15):
                result=self.m.run_child(SimpleNamespace(run_dir=path,work_home=path),'implicit')
            self.assertTrue(result['timed_out'])
            self.assertNotEqual(result['returncode'],0)
