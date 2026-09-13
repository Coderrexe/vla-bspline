"""One supervised gripper-response probe through the existing gated Dora path.

This is NOT a learned-policy trial. Pose and rail increments are exactly zero;
the sole gripper target reduces the fresh measured opening by 0.08 (6.72 mm of
the G2's 84 mm range). No SDK, runtime restart, homing, or force-setting changes.
The normal session, input-calibration and freshness guards still apply. The
optional repeat-hold check sends three eight-row chunks of the SAME opening
target; it never accumulates closing distance and never commands pose/rail motion.
"""
import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np

from apollo_dora_policy import ApolloDoraPolicy
from run_apollo_dora import GuardedPolicyNode,get_json


class GripperCheckPredictor:
    def __init__(self):
        self.target = None

    def reset(self):
        self.target = None

    def predict_chunk(self,state,view,grip):
        opening=float(state[7])
        if self.target is None:
            if not .9 <= opening <= 1.:
                raise ValueError('Gripper check requires a fresh initially open gripper')
            self.target = opening-.08
        out=np.zeros((8,16),dtype=np.float32)
        # Repeating a target must NOT subtract another .08 from the new reading.
        out[:,6]=self.target
        out[:,14]=1.
        return out


def make_policy(checkpoint,output,authorization_file,clock=time.monotonic,*,repeat_hold=False):
    p=ApolloDoraPolicy(checkpoint,predictor=GripperCheckPredictor(),clock=clock,
                      output=output,confirm_parked_setup=True,
                      align_to_training_hemisphere=True,
                      action_rows=8 if repeat_hold else 1,max_chunks=3 if repeat_hold else 1,
                      bounded_rollout=repeat_hold)
    p.policy_id=('apollo_gripper_hold_check_20260912' if repeat_hold
                 else 'apollo_gripper_response_check_20260912')
    p.detail='DETERMINISTIC GRIPPER CHECK: no learned actions; zero pose/rail increments'
    if repeat_hold:
        p.await_bounded_rollout_authorization(authorization_file)
    else:
        p.await_one_row_authorization(authorization_file)
    p.authorization_purpose=('SUPERVISED_GRIPPER_HOLD_CHECK' if repeat_hold
                             else 'SUPERVISED_GRIPPER_CHECK')
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True,help='State schema/statistics only; no model is loaded')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--authorization-file',type=Path,required=True)
    parser.add_argument('--confirm-supervised-gripper-check',action='store_true')
    parser.add_argument('--repeat-hold',action='store_true',
                        help='Three eight-row chunks repeat ONE fixed target; never cumulative closure')
    args=parser.parse_args()
    if not args.confirm_supervised_gripper_check:parser.error('Explicit supervised gripper check required')
    logging.basicConfig(level=logging.INFO)
    info=get_json('http://127.0.0.1:8765/api/dora')
    if info['state']!='attached' or 'policy' not in info['placeholders']:
        raise RuntimeError('No existing policy placeholder; do not change the dataflow')
    os.environ.update(DORA_ZENOH_CONNECT=info['zenoh_connect'],DORA_ZENOH_LISTEN='tcp/127.0.0.1:0',
                      DORA_ZENOH_MULTICAST='off')
    policy=make_policy(args.checkpoint,args.output,args.authorization_file,repeat_hold=args.repeat_hold)
    node=GuardedPolicyNode(policy,daemon_port=info['daemon_port'],rate_hz=25,
                           max_image_age_s=.2,chunk_dt_s=.04,device='cpu',client='simba-gripper-check')
    def stop(*unused):
        policy.disarm('gripper-check client stopped');node.stop()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    timer=threading.Timer(90,stop);timer.daemon=True;timer.start()
    print('DETERMINISTIC GRIPPER CHECK waiting for exact-session grant; no learned policy',flush=True)
    try:node.run()
    finally:
        timer.cancel()
        (args.output/'node_report.json').write_text(json.dumps({
            'purpose':'DETERMINISTIC_GRIPPER_RESPONSE_CHECK_NOT_LEARNED_POLICY',
            'stats':asdict(node.stats),'chunks_returned':policy.returned_chunks,
            'execute_session':policy.execute_session,'disarmed':policy.disarmed,
            'runtime_api_writes':0,'commanded_pose_and_rail_deltas_zero':True,
            'requested_opening_reduction':.08,'repeated_fixed_target':args.repeat_hold},indent=2)+'\n')


if __name__=='__main__':main()
