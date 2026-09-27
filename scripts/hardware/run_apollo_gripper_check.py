"""One supervised gripper-response probe through the existing gated Dora path.

This is NOT a learned-policy trial. Pose and rail increments are exactly zero;
the sole gripper target reduces the fresh measured opening by 0.08 (6.72 mm of
the G2's 84 mm range). No SDK, runtime restart, homing, or force-setting changes.
The normal session, input-calibration and freshness guards still apply. The
optional repeat-hold check sends three eight-row chunks of the SAME opening
target; it never accumulates closing distance and never commands pose/rail motion.
"""
import argparse
from dataclasses import asdict, replace
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from apollo_dora_policy import ApolloDoraPolicy
from apollo_contact_review import require_contact_review_clear
from run_apollo_dora import GuardedPolicyNode,get_json


class GripperCheckPredictor:
    def __init__(self, direction='close'):
        if direction not in {'close','open'}:
            raise ValueError('Choose close or open')
        self.direction = direction
        self.target = None

    def reset(self):
        self.target = None

    def predict_chunk(self,state,view,grip):
        opening=float(state[7])
        if self.target is None:
            if self.direction == 'close':
                if not .9 <= opening <= 1.:
                    raise ValueError('Gripper check requires a fresh initially open gripper')
                self.target = opening-.08
            else:
                if not .8 <= opening <= .92:
                    raise ValueError('Reopening requires a slightly closed, empty gripper')
                self.target = opening+.08
        out=np.zeros((8,16),dtype=np.float32)
        # Repeating a target must NOT subtract another .08 from the new reading.
        out[:,6]=self.target
        out[:,14]=1.
        return out


class AbsoluteGripperCheckPolicy(ApolloDoraPolicy):
    """Hold the measured TCP/rail while probing the absolute-EE gripper lane.

    The ordinary diagnostic uses ``delta_ee``.  Lamp deployment uses ``abs_ee``;
    this adapter changes only the wire representation, so a one-row check can
    distinguish an absolute transport problem from a model/grasp problem.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        names = [
            'ee.x', 'ee.y', 'ee.z', 'ee.r00', 'ee.r10', 'ee.r20',
            'ee.r01', 'ee.r11', 'ee.r21', 'gripper.pos', 'rail.pos',
        ]
        self.spec = replace(
            self.spec,
            action_space='abs_ee',
            action_names=[f'grip_{name}' for name in names],
        )
        self.policy_id = 'apollo_absolute_gripper_response_check_20260913'

    def _encode_for_transport(self, obs, prefix):
        q = np.asarray(obs.state[12:16], dtype=np.float64)
        if not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1.0) > 1e-3:
            raise ValueError('Invalid measured TCP quaternion')
        matrix = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        # Choose the upper float32 neighbour when needed so the millimetre rail
        # encoder cannot truncate a stationary 635 mm reference to 634 mm.
        rail = np.float32(round(float(obs.state[8]) * 1000.0) / 1000.0)
        if float(rail) < round(float(obs.state[8]) * 1000.0) / 1000.0:
            rail = np.nextafter(rail, np.float32(np.inf))
        row = np.r_[obs.state[9:12], matrix[:, 0], matrix[:, 1], 0.0, rail]
        rows = np.repeat(row[None], len(prefix), axis=0)
        rows[:, 9] = prefix[:, 6]
        return rows.astype(np.float32)


def make_policy(checkpoint,output,authorization_file,clock=time.monotonic,*,repeat_hold=False,
                direction='close', absolute_ee=False):
    policy_class = AbsoluteGripperCheckPolicy if absolute_ee else ApolloDoraPolicy
    p=policy_class(checkpoint,predictor=GripperCheckPredictor(direction),clock=clock,
                      output=output,confirm_parked_setup=True,
                      align_to_training_hemisphere=True,
                      action_rows=8 if repeat_hold else 1,max_chunks=3 if repeat_hold else 1,
                      bounded_rollout=repeat_hold)
    if not absolute_ee:
        p.policy_id=('apollo_gripper_hold_check_20260912' if repeat_hold
                     else 'apollo_gripper_response_check_20260912')
    p.detail='DETERMINISTIC GRIPPER CHECK: no learned actions; zero pose/rail increments'
    if repeat_hold:
        p.await_bounded_rollout_authorization(authorization_file)
    else:
        p.await_one_row_authorization(authorization_file)
    p.authorization_purpose=('SUPERVISED_GRIPPER_HOLD_CHECK' if repeat_hold
                             else 'SUPERVISED_GRIPPER_CHECK')
    if direction == 'open':
        p.policy_id='apollo_gripper_open_check_20260912'
        p.authorization_purpose=('SUPERVISED_GRIPPER_OPEN_HOLD_CHECK' if repeat_hold
                                 else 'SUPERVISED_GRIPPER_OPEN_CHECK')
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True,help='State schema/statistics only; no model is loaded')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--authorization-file',type=Path,required=True)
    parser.add_argument('--confirm-supervised-gripper-check',action='store_true')
    parser.add_argument('--repeat-hold',action='store_true',
                        help='Three eight-row chunks repeat ONE fixed target; never cumulative closure')
    parser.add_argument('--direction',choices=['close','open'],default='close',
                        help='Open is a small reopening of the empty gripper after the closing check')
    parser.add_argument('--absolute-ee', action='store_true',
                        help='Hold measured pose and test the lamp deployment abs_ee gripper lane')
    parser.add_argument('--lamp-task-review', type=Path,
                        help='Current operator review required for the absolute lamp-lane probe')
    args=parser.parse_args()
    require_contact_review_clear(
        lamp_task_review=args.lamp_task_review if args.absolute_ee else None,
        policy_id=('apollo_absolute_gripper_response_check_20260913'
                   if args.absolute_ee else None),
    )
    if not args.confirm_supervised_gripper_check:parser.error('Explicit supervised gripper check required')
    if args.absolute_ee != bool(args.lamp_task_review):
        parser.error('The absolute lamp-lane probe requires its current lamp task review')
    logging.basicConfig(level=logging.INFO)
    info=get_json('http://127.0.0.1:8765/api/dora')
    if info['state']!='attached' or 'policy' not in info['placeholders']:
        raise RuntimeError('No existing policy placeholder; do not change the dataflow')
    os.environ.update(DORA_ZENOH_CONNECT=info['zenoh_connect'],DORA_ZENOH_LISTEN='tcp/127.0.0.1:0',
                      DORA_ZENOH_MULTICAST='off')
    policy=make_policy(args.checkpoint,args.output,args.authorization_file,
                       repeat_hold=args.repeat_hold,direction=args.direction,
                       absolute_ee=args.absolute_ee)
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
            'requested_opening_reduction':.08 if args.direction=='close' else -.08,
            'direction':args.direction,'repeated_fixed_target':args.repeat_hold,
            'wire_action_space':policy.spec.action_space},indent=2)+'\n')


if __name__=='__main__':main()
