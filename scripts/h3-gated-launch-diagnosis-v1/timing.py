"""CPU-only accounting: wall-clock phases and graph topology, never CPU=total-GPU."""
import math
import time


def validate_split(row):
    values=[row[k] for k in ('submit_ns','remaining_wait_ns','total_ns')]
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in values):
        raise ValueError('Invalid wall-clock phase')
    if values[2]<=0 or values[0]+values[1]!=values[2]:
        raise ValueError('Wall-clock phases must sum exactly to total')


def host_split(call,synchronize,clock=time.perf_counter_ns):
    synchronize()
    start=clock()
    output=call()
    submitted=clock()
    synchronize()
    finished=clock()
    row={'submit_ns':submitted-start,'remaining_wait_ns':finished-submitted,'total_ns':finished-start}
    validate_split(row)
    return row,output


def graph_interval(elapsed_ms,operations):
    # DTK returned only two opaque nodes after 16 baseline submissions.
    # Until batch work is independently proven, a loop count is not a valid divisor.
    if operations!=1:
        raise ValueError('Only single-operation graph timing is validated; batch normalization is disabled')
    if not math.isfinite(elapsed_ms) or elapsed_ms<=0:
        raise ValueError('Invalid captured event interval')
    return {'event_ms':elapsed_ms,'operations':1,'device_graph_us_per_op':elapsed_ms*1000}


def event_layout(types):
    # Event types are learned only from two explicitly-created event nodes.
    # Do not derive a compute-node type arithmetically: Hygon DTK may expose
    # opaque computation nodes whose enum is unrelated to the generic HIP ABI.
    if types not in ([7,7],[8,8]):
        raise ValueError('Unrecognized HIP event-node layout: '+str(types))
    return {'event':types[0]}


def compute_layout(baseline,fused,event):
    """Infer an opaque compute node type from two controlled call paths."""
    baseline_types=baseline.get('node_types')
    fused_types=fused.get('node_types')
    if not isinstance(event,dict) or set(event)!={'event'}:
        raise ValueError('Event layout must be calibrated first')
    if (not isinstance(baseline_types,list) or len(baseline_types)!=2 or
            not isinstance(fused_types,list) or len(fused_types)!=1):
        raise ValueError('Expected two baseline nodes and one fused node')
    serial_chain(2,baseline.get('edges',[]))
    serial_chain(1,fused.get('edges',[]))
    kind=baseline_types[0]
    if (not isinstance(kind,int) or kind<0 or baseline_types!=[kind,kind] or
            fused_types!=[kind] or kind==event['event']):
        raise ValueError('Baseline/fused computation node types do not agree')
    return {'event':event['event'],'compute':kind}


def validate_graph(types,edges,kernels,layout):
    n=len(types)
    if (set(layout)!={'event','compute'} or n!=kernels+2 or
            types.count(layout['event'])!=2 or types.count(layout['compute'])!=kernels):
        raise ValueError('Unexpected graph node types/counts')
    chain=serial_chain(n,edges)
    if types[chain[0]]!=layout['event'] or types[chain[-1]]!=layout['event']:
        raise ValueError('Timed events must bracket all kernels')
    return chain


def serial_chain(n,edges):
    if n<1:raise ValueError('Empty computation graph')
    if len(edges)!=n-1 or len(set(map(tuple,edges)))!=n-1:
        raise ValueError('Graph must be a single dependency chain')
    incoming=[0]*n;outgoing={}
    for a,b in edges:
        if not (0<=a<n and 0<=b<n) or a==b or a in outgoing:
            raise ValueError('Invalid graph dependency')
        outgoing[a]=b;incoming[b]+=1
    roots=[i for i,count in enumerate(incoming) if count==0]
    if len(roots)!=1 or any(count>1 for count in incoming):raise ValueError('Graph is not serial')
    chain=[];node=roots[0]
    while node not in chain:
        chain.append(node)
        if node not in outgoing:break
        node=outgoing[node]
    if len(chain)!=n or chain[-1] in outgoing:raise ValueError('Graph is cyclic/disconnected')
    return chain


def complete_samples(rows,samples,paths):
    if len(rows)!=samples*len(paths) or {(r['index'],r['path']) for r in rows}!={(i,p) for i in range(samples) for p in paths}:
        return False
    try:
        for row in rows:validate_split(row)
    except (KeyError,ValueError):return False
    return True
