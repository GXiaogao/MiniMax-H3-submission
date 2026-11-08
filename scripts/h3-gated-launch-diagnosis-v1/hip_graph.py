"""Diagnostic raw HIP capture only. No InfiniCore graph flags or kernels changed."""
import ctypes as ct
import json
import os
from pathlib import Path

from timing import compute_layout,event_layout,validate_graph,serial_chain

P=ct.c_void_p
PP=ct.POINTER(P)


def stream_pointer(value):
    if isinstance(value,int): pointer=value
    elif type(value).__name__=='PyCapsule':
        name=ct.pythonapi.PyCapsule_GetName
        name.argtypes=[ct.py_object];name.restype=ct.c_char_p
        get=ct.pythonapi.PyCapsule_GetPointer
        get.argtypes=[ct.py_object,ct.c_char_p];get.restype=P
        pointer=get(value,name(value))
    else:raise TypeError('Unrecognized InfiniCore stream handle')
    if not pointer:raise ValueError('Explicit non-default InfiniCore stream required')
    return int(pointer)


class Hip:
    def __init__(self,bundle,stream):
        self.stream=stream_pointer(stream)
        path=Path((Path(bundle)/'hip_library.txt').read_text().strip()).resolve(strict=True)
        loaded={Path(line.split(maxsplit=5)[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
                if ('libamdhip64.so' in line or 'libgalaxyhip.so' in line) and line.split(maxsplit=5)[-1].startswith('/')}
        if loaded!={path}:raise RuntimeError(f'HIP runtime identity mismatch: {loaded} != {path}')
        self.path=path;self.lib=ct.CDLL(str(path))
        signatures={
            'hipGetErrorString':([ct.c_int],ct.c_char_p),
            'hipStreamSynchronize':([P],ct.c_int),
            'hipStreamBeginCapture':([P,ct.c_int],ct.c_int),
            'hipStreamEndCapture':([P,PP],ct.c_int),
            'hipGraphCreate':([PP,ct.c_uint],ct.c_int),
            'hipGraphAddEventRecordNode':([PP,P,PP,ct.c_size_t,P],ct.c_int),
            'hipGraphAddDependencies':([P,PP,PP,ct.c_size_t],ct.c_int),
            'hipGraphGetNodes':([P,PP,ct.POINTER(ct.c_size_t)],ct.c_int),
            'hipGraphGetEdges':([P,PP,PP,ct.POINTER(ct.c_size_t)],ct.c_int),
            'hipGraphNodeGetType':([P,ct.POINTER(ct.c_int)],ct.c_int),
            'hipGraphInstantiate':([PP,P,PP,P,ct.c_size_t],ct.c_int),
            'hipGraphLaunch':([P,P],ct.c_int),
            'hipGraphExecDestroy':([P],ct.c_int),'hipGraphDestroy':([P],ct.c_int),
            'hipEventCreateWithFlags':([PP,ct.c_uint],ct.c_int),
            'hipEventDestroy':([P],ct.c_int),
            'hipEventElapsedTime':([ct.POINTER(ct.c_float),P,P],ct.c_int),
        }
        for name,(args,result) in signatures.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=result
        self.layout=None

    def call(self,name,*args):
        rc=getattr(self.lib,name)(*args)
        if rc:
            message=self.lib.hipGetErrorString(rc)
            raise RuntimeError(f'{name} failed ({rc}): '+(message or b'unknown').decode(errors='replace'))

    def sync(self):self.call('hipStreamSynchronize',self.stream)

    def topology(self,graph):
        count=ct.c_size_t()
        self.call('hipGraphGetNodes',graph,None,ct.byref(count))
        if not 1<=count.value<=10000:raise RuntimeError('Unexpected graph size')
        nodes=(P*count.value)();self.call('hipGraphGetNodes',graph,nodes,ct.byref(count))
        indices={value:i for i,value in enumerate(nodes)};types=[]
        for node in nodes:
            kind=ct.c_int();self.call('hipGraphNodeGetType',node,ct.byref(kind));types.append(kind.value)
        num=ct.c_size_t();self.call('hipGraphGetEdges',graph,None,None,ct.byref(num))
        if num.value>10000:raise RuntimeError('Unexpected graph edge count')
        src=(P*num.value)();dst=(P*num.value)()
        if num.value:self.call('hipGraphGetEdges',graph,src,dst,ct.byref(num))
        return {'node_handles':list(nodes),'node_types':types,'edges':[(indices[a],indices[b]) for a,b in zip(src,dst)]}

    def calibrate(self):
        with Captured(self,lambda:None,0,None) as graph:
            self.layout=event_layout(graph.topology['node_types'])
            serial_chain(2,graph.topology['edges'])
            graph.launch();self.sync()
            return {'node_layout':self.layout,'event_only_graph':graph.topology}

    def capture_computation_topology(self,call):
        """Capture one controlled operation without instantiating or replaying it."""
        graph=P();begun=False;failed=False
        try:
            self.sync()
            self.call('hipStreamBeginCapture',self.stream,2);begun=True
            call()
            begun=False;self.call('hipStreamEndCapture',self.stream,ct.byref(graph))
            topology=self.topology(graph)
            return {'node_types':topology['node_types'],'edges':topology['edges']}
        except BaseException:
            failed=True
            if begun:self.lib.hipStreamEndCapture(self.stream,ct.byref(graph))
            raise
        finally:
            if graph:
                try:self.call('hipGraphDestroy',graph)
                except Exception:
                    if not failed:raise

    def calibrate_computation(self,baseline,fused):
        if self.layout is None or set(self.layout)!={'event'}:
            raise RuntimeError('Event layout must be calibrated first')
        baseline_topology=self.capture_computation_topology(baseline)
        fused_topology=self.capture_computation_topology(fused)
        self.layout=compute_layout(baseline_topology,fused_topology,self.layout)
        return {'node_layout':self.layout,'baseline_computation_graph':baseline_topology,
                'fused_computation_graph':fused_topology,
                'inference':'baseline=two serial same-type nodes; fused=one node of the same type'}


class Captured:
    def __init__(self,hip,call,operations,kernels):
        if operations not in (0,1):
            raise ValueError('Only single-operation capture is supported; repeated capture work is unverified')
        self.hip=hip;self.graph=P();self.exec=P();self.events=[];begun=False
        try:
            hip.sync()
            for _ in range(2):
                event=P();hip.call('hipEventCreateWithFlags',ct.byref(event),0);self.events.append(event)
            root=tail=None
            if operations:
                hip.call('hipStreamBeginCapture',hip.stream,2);begun=True # relaxed, single process/thread
                for _ in range(operations):call()
                # Even a failed EndCapture ends the capture; do not invoke it twice.
                begun=False;hip.call('hipStreamEndCapture',hip.stream,ct.byref(self.graph))
                computation=hip.topology(self.graph)
                if kernels is None or computation['node_types']!=[hip.layout['compute']]*kernels:
                    details={'node_types':computation['node_types'],'edges':computation['edges'],
                             'node_count':len(computation['node_types']),
                             'expected_compute_type':hip.layout['compute'],'expected_compute_nodes':kernels}
                    raise ValueError('Unexpected computation graph topology: '+json.dumps(details,separators=(',',':')))
                chain=serial_chain(kernels,computation['edges'])
                root=computation['node_handles'][chain[0]];tail=computation['node_handles'][chain[-1]]
            else:
                hip.call('hipGraphCreate',ct.byref(self.graph),0)
            # Ordinary hipEventRecord during capture can only record dependency
            # information, not real event nodes. Add explicit graph nodes using
            # the stable API available in both HIP 5.2 and 6.x instead.
            start_node=P();end_node=P()
            hip.call('hipGraphAddEventRecordNode',ct.byref(start_node),self.graph,None,0,self.events[0])
            if root is not None:
                hip.call('hipGraphAddDependencies',self.graph,(P*1)(start_node.value),(P*1)(root),1)
            else:tail=start_node.value
            hip.call('hipGraphAddEventRecordNode',ct.byref(end_node),self.graph,(P*1)(tail),1,self.events[1])
            self.topology=hip.topology(self.graph)
            if kernels is not None:
                validate_graph(self.topology['node_types'],self.topology['edges'],kernels,hip.layout)
            error_node=P();log=ct.create_string_buffer(4096)
            try:hip.call('hipGraphInstantiate',ct.byref(self.exec),self.graph,ct.byref(error_node),log,len(log))
            except RuntimeError as exc:raise RuntimeError(str(exc)+'; '+log.value.decode(errors='replace')) from exc
        except BaseException:
            if begun:
                # Cleanup attempt must not hide the original capture error.
                hip.lib.hipStreamEndCapture(hip.stream,ct.byref(self.graph))
            self.close(suppress=True)
            raise

    def launch(self):self.hip.call('hipGraphLaunch',self.exec,self.hip.stream)

    def elapsed_ms(self):
        result=ct.c_float()
        self.hip.call('hipEventElapsedTime',ct.byref(result),*self.events)
        return result.value

    def close(self,suppress=False):
        errors=[]
        try:self.hip.sync()
        except Exception as exc:errors.append(exc)
        for name,value in [('hipGraphExecDestroy',self.exec),('hipGraphDestroy',self.graph),
                           *[('hipEventDestroy',v) for v in self.events]]:
            if value:
                try:self.hip.call(name,value)
                except Exception as exc:errors.append(exc)
        self.exec=P();self.graph=P();self.events=[]
        if errors and not suppress:raise errors[0]

    def __enter__(self):return self
    def __exit__(self,kind,value,tb):self.close(suppress=kind is not None)


class TensorArg(ct.Structure):
    _fields_=[('pointer',P),('rows',ct.c_int64),('width',ct.c_int64),('stride0',ct.c_int64),('stride1',ct.c_int64)]


class Direct:
    """Same verified DSO/HSACO; descriptor preparation excluded from timing."""
    def __init__(self,bundle,hip,fixture):
        self.hip=hip;self.fixture=fixture;self.handle=None
        root=Path(bundle);self.lib=ct.CDLL(str(root/'libh3_gated_host.so'))
        for name,args,result in (
            ('h3_open',[ct.c_char_p,ct.c_char_p,ct.c_char_p],P),
            ('h3_launch',[P,ct.POINTER(TensorArg),P],ct.c_int),
            ('h3_error',[],ct.c_char_p),('h3_calls',[P],ct.c_uint64),('h3_close',[P],ct.c_int)):
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=result
        tensors=[*fixture.views,fixture.buffers[2]]
        self.args=(TensorArg*4)(*(TensorArg(t.data_ptr(),*t.shape[-2:],*t.stride()[-2:]) for t in tensors))
        dtype=fixture.dtype_name
        self.handle=self.lib.h3_open(os.fsencode(hip.path),os.fsencode(root/f'export_{dtype}/kernel.hsaco'),
                                     f'h3_gated_residual_{dtype}'.encode())
        if not self.handle:raise RuntimeError(self.lib.h3_error().decode(errors='replace'))

    def __call__(self):
        if self.lib.h3_launch(self.handle,self.args,self.hip.stream)!=0:
            raise RuntimeError(self.lib.h3_error().decode(errors='replace'))
        return self.fixture.new

    def count(self):return self.lib.h3_calls(self.handle)

    def close(self):
        self.hip.sync()
        if self.handle:
            if self.lib.h3_close(self.handle)!=0:raise RuntimeError(self.lib.h3_error().decode(errors='replace'))
            self.handle=None
