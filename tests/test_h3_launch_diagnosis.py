"""CPU timing/protocol checks; fake clocks do not claim DCU verification."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest
import subprocess
import tempfile
import os
import json
import zipfile
import contextlib
import io
import ctypes as ct
import tarfile
import hashlib

PACKAGE=Path(__file__).resolve().parents[1]/'scripts/h3-gated-launch-diagnosis-v1'

class TimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('split_timing',PACKAGE/'timing.py')
        cls.api=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.api)

    def test_host_split_excludes_initial_sync_and_wait_is_not_compute(self):
        ticks=[0]
        def sync(): ticks[0]+=7000
        def call(): ticks[0]+=3000;return 'output'
        row,out=self.api.host_split(call,sync,lambda:ticks[0])
        self.assertEqual(out,'output')
        self.assertEqual(row,{'submit_ns':3000,'remaining_wait_ns':7000,'total_ns':10000})

    def test_raw_timing_identity_rejects_corruption_and_nonfinite(self):
        self.api.validate_split({'submit_ns':3,'remaining_wait_ns':7,'total_ns':10})
        for row in ({'submit_ns':3,'remaining_wait_ns':7,'total_ns':9},
                    {'submit_ns':-1,'remaining_wait_ns':7,'total_ns':6},
                    {'submit_ns':math.nan,'remaining_wait_ns':7,'total_ns':10}):
            with self.assertRaises(ValueError): self.api.validate_split(row)

    def test_single_operation_interval_retains_raw_ms_without_batch_division(self):
        self.assertEqual(self.api.graph_interval(3.2,1),{'event_ms':3.2,'operations':1,'device_graph_us_per_op':3200.0})
        for ms,k in ((0,1),(1,0),(math.nan,1),(3.2,2),(3.2,16),(3.2,64)):
            with self.assertRaises(ValueError):self.api.graph_interval(ms,k)

    def test_runtime_enum_calibration_accepts_old_and_new_not_arbitrary(self):
        self.assertEqual(self.api.event_layout([8,8]),{'event':8})
        self.assertEqual(self.api.event_layout([7,7]),{'event':7})
        for types in ([7],[7,8],[1,1],[99,99]):
            with self.assertRaises(ValueError):self.api.event_layout(types)

    def test_compute_type_is_inferred_from_baseline_and_fused_topologies(self):
        baseline={'node_types':[200,200],'edges':[(0,1)]}
        fused={'node_types':[200],'edges':[]}
        self.assertEqual(self.api.compute_layout(baseline,fused,{'event':7}),
                         {'event':7,'compute':200})

    def test_compute_type_inference_rejects_uncontrolled_topologies(self):
        good_baseline={'node_types':[200,200],'edges':[(0,1)]}
        good_fused={'node_types':[200],'edges':[]}
        bad_pairs=(
            ({'node_types':[200,0],'edges':[(0,1)]},good_fused),
            (good_baseline,{'node_types':[0],'edges':[]}),
            ({'node_types':[200,200],'edges':[]},good_fused),
            ({'node_types':[200,200,200],'edges':[(0,1),(1,2)]},good_fused),
            (good_baseline,{'node_types':[200,200],'edges':[(0,1)]}),
            ({'node_types':[7,7],'edges':[(0,1)]},{'node_types':[7],'edges':[]}),
        )
        for baseline,fused in bad_pairs:
            with self.subTest(baseline=baseline,fused=fused):
                with self.assertRaises(ValueError):
                    self.api.compute_layout(baseline,fused,{'event':7})

    def test_graph_gate_rejects_missing_nodes_callbacks_and_broken_dependencies(self):
        layout={'compute':200,'event':7}
        good=[7,200,200,7];edges=[(0,1),(1,2),(2,3)]
        self.api.validate_graph(good,edges,2,layout)
        for types,links in (([7,200,7],[(0,1),(1,2)]),([7,3,200,7],edges),
                            (good,[(0,1),(0,2),(2,3)]),(good,[(0,1),(1,2),(2,1)]),
                            ([200,7,200,7],edges)):
            with self.assertRaises(ValueError):self.api.validate_graph(types,links,2,layout)

    def test_completeness_does_not_accept_partial_or_duplicate_samples(self):
        good=[{'index':i,'path':p,'submit_ns':3,'remaining_wait_ns':7,'total_ns':10}
              for i in range(2) for p in ('baseline','fused','direct')]
        self.assertTrue(self.api.complete_samples(good,2,('baseline','fused','direct')))
        self.assertFalse(self.api.complete_samples(good[:-1],2,('baseline','fused','direct')))
        self.assertFalse(self.api.complete_samples(good[:-1]+[good[0]],2,('baseline','fused','direct')))

class CliTests(unittest.TestCase):
    def test_unverified_batch_never_reaches_graph_measurement(self):
        sys.path.insert(0,str(PACKAGE))
        try:
            spec=importlib.util.spec_from_file_location('launch_diag_batch_test',PACKAGE/'diagnose.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as directory:
                folder=Path(directory)
                class NoGpuAccess:
                    def __getattr__(self,name):
                        raise AssertionError('Unsupported batch reached device state: '+name)
                with self.assertRaisesRegex(ValueError,'single-operation'):
                    module.graph_measure(NoGpuAccess(),NoGpuAccess(),NoGpuAccess(),{},None,folder,16)
                report=json.loads((folder/'graph_16.json').read_text())
                self.assertFalse(report['passed'])
                self.assertEqual(report['samples'],[])
        finally:sys.path.pop(0)

    def test_baseline_replay_must_recompute_the_intermediate_product(self):
        from types import SimpleNamespace
        sys.path.insert(0,str(PACKAGE))
        try:
            spec=importlib.util.spec_from_file_location('launch_diag_poison_test',PACKAGE/'diagnose.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            class Buffer:
                value=123.0
                def fill_(self,value):self.value=value
            fixture=SimpleNamespace(buffers=[Buffer() for _ in range(4)],sync=lambda:None)
            def replay_boundary():
                self.assertTrue(math.isnan(fixture.buffers[0].value),'stale product may mask an omitted mul')
                self.assertTrue(math.isnan(fixture.buffers[1].value),'stale output may mask an omitted add')
                raise RuntimeError('replay boundary verified')
            with self.assertRaisesRegex(RuntimeError,'replay boundary verified'):
                module.correct(fixture,replay_boundary,'baseline',None,'test')
        finally:sys.path.pop(0)

    def test_worker_output_is_preserved_and_nonzero_return_is_not_hidden(self):
        sys.path.insert(0,str(PACKAGE))
        try:
            spec=importlib.util.spec_from_file_location('launch_diag_test',PACKAGE/'diagnose.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()) as out:
                path=Path(directory)/'worker.log'
                code=module.run_worker([sys.executable,'-u','-c',
                    'import sys; print("stage one"); print("stage two",file=sys.stderr); sys.exit(3)'],os.environ.copy(),path,10)
                self.assertEqual(code,3)
                self.assertEqual(path.read_text(),out.getvalue())
                self.assertIn('stage one',out.getvalue());self.assertIn('stage two',out.getvalue())
        finally:sys.path.pop(0)

    def test_missing_bundle_still_archives_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            proc=subprocess.run([sys.executable,str(PACKAGE/'diagnose.py')],
                env=dict(os.environ,WORK_HOME=directory),capture_output=True,text=True)
            self.assertEqual(proc.returncode,1,proc.stdout+proc.stderr)
            archives=list((Path(directory)/'logs').glob('h3_gated_launch_*.zip'))
            self.assertEqual(len(archives),1)
            with zipfile.ZipFile(archives[0]) as archive:
                run=json.loads(archive.read(next(n for n in archive.namelist() if n.endswith('/run.json'))))
                self.assertFalse(run['passed']);self.assertIn('error',run)
            self.assertNotIn('DIAGNOSIS COMPLETE',proc.stdout)

    def test_invalid_cli_stops_before_creating_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            proc=subprocess.run([sys.executable,str(PACKAGE/'diagnose.py'),'--samples','3'],
                env=dict(os.environ,WORK_HOME=directory),capture_output=True,text=True)
            self.assertEqual(proc.returncode,2)
            self.assertIn('even',proc.stderr)
            self.assertFalse((Path(directory)/'logs').exists())

class HipBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(PACKAGE))
        try:
            spec=importlib.util.spec_from_file_location('hip_diag_test',PACKAGE/'hip_graph.py')
            cls.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.module)
        finally:sys.path.pop(0)

    def test_real_capsule_pointer_preserves_64_bit_address_and_name(self):
        new=ct.pythonapi.PyCapsule_New
        new.argtypes=[ct.c_void_p,ct.c_char_p,ct.c_void_p];new.restype=ct.py_object
        capsule=new(0x123456789abc,b'hip.stream.test',None)
        self.assertEqual(self.module.stream_pointer(capsule),0x123456789abc)
        with self.assertRaises(ValueError):self.module.stream_pointer(0)
        with self.assertRaises(TypeError):self.module.stream_pointer('123')

    def test_unverified_batch_is_rejected_before_any_gpu_action(self):
        class NoGpuAccess:
            def __getattr__(self,name):
                raise AssertionError('GPU must not be touched for unsupported batching: '+name)
        for count in (2,16,64):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError,'single-operation'):
                    self.module.Captured(NoGpuAccess(),lambda:None,count,count*2)

    def test_capture_exception_attempts_end_and_releases_events(self):
        events=[]
        class Runtime:
            stream=123
            def __init__(self):self.lib=self
            def sync(self):events.append('sync')
            def call(self,name,*args):
                events.append(name)
                if name=='hipEventCreateWithFlags':args[0]._obj.value=len(events)+100
            def hipStreamEndCapture(self,stream,graph):
                events.append('abort_capture');graph._obj.value=999;return 0
        def fail():raise ValueError('operation failure')
        with self.assertRaisesRegex(ValueError,'operation failure'):
            self.module.Captured(Runtime(),fail,1,1)
        self.assertEqual(events.count('abort_capture'),1)
        self.assertEqual(events.count('hipEventDestroy'),2)
        self.assertEqual(events.count('hipGraphDestroy'),1)
        self.assertNotIn('hipGraphInstantiate',events)

    def test_default_events_that_are_not_captured_still_get_explicit_timing_nodes(self):
        # Realistic runtime boundary: ordinary EventRecord during capture is
        # ignored. Production must explicitly insert start/end graph nodes.
        class Runtime:
            stream=123
            layout={'compute':0,'event':7}
            def __init__(self):
                self.lib=self;self.nodes={11:0};self.edges=[];self.next=20
                self.added=[];self.recorded=[]
            def sync(self):pass
            def call(self,name,*args):
                if name=='hipEventCreateWithFlags':self.next+=1;args[0]._obj.value=self.next
                elif name=='hipStreamEndCapture':args[1]._obj.value=99
                elif name=='hipGraphInstantiate':args[0]._obj.value=88
                elif name=='hipGraphAddEventRecordNode':
                    self.next+=1;args[0]._obj.value=self.next;self.nodes[self.next]=7
                    self.edges.extend((args[2][i],self.next) for i in range(args[3]));self.added.append(self.next)
                elif name=='hipGraphAddDependencies':
                    self.edges.extend((args[1][i],args[2][i]) for i in range(args[3]))
                elif name=='hipEventRecord':self.recorded.append(name)
            def topology(self,graph):
                nodes=list(self.nodes);lookup={node:i for i,node in enumerate(nodes)}
                return {'node_types':list(self.nodes.values()),
                        'edges':[(lookup[a],lookup[b]) for a,b in self.edges],
                        'node_handles':nodes}
        runtime=Runtime()
        with self.module.Captured(runtime,lambda:None,1,1) as graph:
            self.assertEqual(graph.topology['node_types'].count(7),2)
            self.assertEqual(len(graph.topology['edges']),2)
        self.assertEqual(runtime.recorded,[])
        self.assertEqual(len(runtime.added),2)

    def test_unexpected_computation_topology_is_reported_before_rejection(self):
        class Runtime:
            stream=123
            layout={'compute':0,'event':7}
            def __init__(self):self.lib=self;self.events=[]
            def sync(self):pass
            def call(self,name,*args):
                if name=='hipEventCreateWithFlags':args[0]._obj.value=len(self.events)+101;self.events.append(args[0]._obj.value)
                elif name=='hipStreamEndCapture':args[1]._obj.value=99
            def topology(self,graph):
                return {'node_handles':[11,12,13],'node_types':[0,4,0],'edges':[(0,1),(1,2)]}
        with self.assertRaisesRegex(ValueError,
                r'node_types.*\[0,4,0\].*expected_compute_type.*0.*expected_compute_nodes.*2'):
            self.module.Captured(Runtime(),lambda:None,1,2)

class PackageTests(unittest.TestCase):
    def test_package_contains_verified_lf_files_and_identical_reused_helpers(self):
        artifact=PACKAGE.parents[1]/'artifacts/h3-gated-launch-diagnosis-v3-single-op.tar.gz'
        with tarfile.open(artifact) as archive:
            files={Path(member.name).name:archive.extractfile(member).read() for member in archive.getmembers()}
        for line in files['SHA256SUMS'].decode().splitlines():
            digest,name=line.split('  ',1)
            self.assertEqual(hashlib.sha256(files[name]).hexdigest(),digest)
            self.assertNotIn(b'\r\n',files[name]);self.assertEqual(files[name],(PACKAGE/name).read_bytes())
        original=PACKAGE.parent/'h3-gated-benchmark-v1'
        for old,new in [('benchmark.py','benchmark_support.py'),('protocol.py','protocol.py'),
                        ('rounding_reference.py','rounding_reference.py'),('check_sources.py','check_sources.py'),
                        ('patch_manifest.json','patch_manifest.json')]:
            self.assertEqual(files[new],(original/old).read_bytes().replace(b'\r\n',b'\n'))

if __name__=='__main__':unittest.main()
