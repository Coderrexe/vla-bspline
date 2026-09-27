"""Offline replay of selected PURE methods from the runtime's exact source files.

No runtime module is imported, no hardware class constructed, no API connected.
AST extraction retains the unmodified row selection, scaling, and budget code.
This checks delta bookkeeping only, not IK, physical tracking, or collision safety.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np


def extract_class(path, name, methods, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == name)
    cls.bases,cls.keywords,cls.decorator_list = [],[],[]
    cls.body = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods]
    if {n.name for n in cls.body} != set(methods):
        raise ValueError('Runtime source no longer matches the audited pure methods')
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace[name]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    source=args.runtime_source/'dora_bridge/policy_source.py'
    runner=args.runtime_source/'dagger/policy_runner.py'
    ns={'np':np,'PolicyOutput':SimpleNamespace}
    Slot=extract_class(source,'_ArmSlot',['__init__','current'],ns)
    Source=extract_class(source,'ExternalPolicySource',['latest'],ns)
    Anchor=extract_class(runner,'ActionAnchor',['row_step'],ns)
    report={'purpose':'PURE_RUNTIME_DELTA_BOOKKEEPING_REPLAY','hardware_connected':False,
            'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [source,runner]},
            'cases':[]}
    for count in (1,8):
        # The 400 ms case is an offline candidate only. Adding it here does not
        # enable that clock in the live client, grants, or response watchdogs.
        for rate,row_dt in ((3.125,.04),(25.,.04),(5.,.2),(2.5,.4)):
            slot=Slot()
            slot.rows=np.zeros((count,8),np.float32)
            slot.rows[:,0]=.001
            slot.rows[:,3]=.002
            slot.rows[:,6]=.95
            slot.t0=0.;slot.chunk_dt=row_dt
            src=Source();src._paused=False;src._lock=threading.Lock();src._version=1
            src.arms_meta=[('grip',True)];src._slots={'grip':slot};src._block_dim={'grip':8}
            src._block_mask={'grip':np.array([1]*6+[0,1])};src.period=1/rate
            anchor=Anchor();anchor._rows={}
            total=np.zeros(7)
            for tick in range(400):
                now=tick*.01+1e-9
                src._clock=lambda:now
                out,t=src.latest()
                delta=np.r_[out.actions[:6],out.actions[7]]
                total+=anchor.row_step('grip',(out.version,t,out.chunk_remaining),delta,.01/src.period)
                assert out.actions[6] == slot.rows[0,6]  # gripper must stay absolute
            intended=np.r_[slot.rows[:,:6].sum(0),slot.rows[:,7].sum()]
            multiplier=total[0]/intended[0]
            expected=(1/rate)/row_dt
            np.testing.assert_allclose(total,intended*expected,rtol=1e-6,atol=1e-9)
            report['cases'].append({'chunk_rows':count,'wire_rate_hz':rate,'row_dt_s':row_dt,
                                    'candidate_not_live_enabled': row_dt == .4,
                                    'integrated_translation_m':total[:3].tolist(),
                                    'intended_translation_m':intended[:3].tolist(),
                                    'delta_multiplier':float(multiplier)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
