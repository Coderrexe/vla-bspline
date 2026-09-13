"""Run the Apollo plugin without changing the lab runtime. Defaults to shadow."""
import argparse
import json
import logging
import os
import signal
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from mavis_policy_node.node import PolicyNode
from mavis_policy_node.messages import parse_event, parse_reset

from apollo_dora_policy import ApolloDoraPolicy


class GuardedPolicyNode(PolicyNode):
    """Additional same-host freshness/session checks around the reference client."""

    max_inference_rate_hz = 3.125

    def __init__(self, policy, **kwargs):
        # ExternalPolicySource rescales by period/chunk_dt before ActionAnchor's
        # per-row budget. For native delta rows these clocks MUST agree. Model
        # invocation remains independently capped below, not at this wire rate.
        wire_rate = 1.0/policy.chunk_dt_s
        if 'rate_hz' in kwargs and abs(kwargs['rate_hz']-wire_rate) > 1e-6:
            raise ValueError('Wire rate must match native action-row cadence, not model-call rate')
        kwargs['rate_hz'] = wire_rate
        super().__init__(policy, **kwargs)

    def _on_session(self, node, event):
        from mavis_policy_node.messages import parse_session
        session = parse_session(event)
        if (session.get('epoch') != self._epoch
                or session.get('session_id') != self._session_id):
            self._frames.clear()
            self._depth.clear()
        super()._on_session(node, event)

    def _release(self, node):
        self.policy.disarm('Dora connection closed; no automatic motion restart')
        super()._release(node)

    def _on_reset(self, event):
        message = parse_reset(event)
        if message.get('reason') != 'session_start':
            self.policy.disarm('runtime reset or operator intervention')
        super()._on_reset(event)

    def _on_event(self, event):
        message = parse_event(event)
        if message.get('kind') in {'gate', 'collision', 'policy_anomaly', 'session_error'}:
            self.policy.disarm(f"runtime event: {message['kind']}")
        super()._on_event(event)

    def _on_camera(self, event, camera_id, is_depth):
        # Ignore synthetic twin/depth streams; neither was a training input.
        if is_depth or camera_id not in self.spec.camera_keys:
            return
        meta = event.get('metadata') or {}
        if (meta.get('epoch') != self._epoch or meta.get('mavis_schema') != 1
                or meta.get('encoding') != 'rgb8'
                or (meta.get('height'), meta.get('width')) != (480, 640)):
            return
        super()._on_camera(event, camera_id, is_depth)

    def _on_obs(self, node, event):
        meta = event.get('metadata') or {}
        if (meta.get('session_id') != self.session_id or meta.get('epoch') != self._epoch
                or meta.get('mavis_schema') != 1):
            return
        if self.validate_against_session(self._session or {}):
            self.policy.disarm('session contract mismatch')
            return
        if meta.get('quat_order') != 'wxyz':
            self.policy.disarm('unexpected observation quaternion convention')
            return
        if meta.get('engaged_arm'):
            self.policy.disarm('operator teleoperation is engaged')
            return
        now = self._clock()
        # Parent accepts at 0.9*period; enforce a full 0.32 s between starts.
        if now-self._last_act_t < 1.0/self.max_inference_rate_hz:
            return
        if not -.02 <= now-float(meta.get('t_mono', 0)) <= .25:
            if self.policy.returned_chunks:
                self.policy.disarm('stale state stream')
            return
        for camera in self.spec.camera_keys:
            frame = self._frames.get(camera)
            if frame is None or not -.02 <= now-frame.t_mono <= .2:
                if self.policy.returned_chunks:
                    self.policy.disarm('stale camera stream')
                return
        super()._on_obs(node, event)

    def _set_health(self, health, detail):
        if health == 'error':
            self.policy.disarm(detail)
        super()._set_health(health, detail)

    def _per_arm_outputs_available(self, node, arms):
        if not super()._per_arm_outputs_available(node, arms):
            self.policy.disarm('required action_grip output is unavailable')
            raise RuntimeError('No fallback to a whole-cell action output in this integration')
        return True


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--url', default='http://127.0.0.1:8765')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--execute-session', help='Explicit CURRENT session ID; omitted = no action publication')
    p.add_argument('--one-row-authorization-file', type=Path,
                   help='Fresh one-row grant written by our separately approved session helper after startup')
    p.add_argument('--bounded-rollout-authorization-file', type=Path,
                   help='At most three chunks, with total 10 mm/0.15 rad/0.1 gripper budget')
    p.add_argument('--confirm-supervised-trial', action='store_true')
    p.add_argument('--supervised-approach', action='store_true',
                   help='Explicit approach profile: at most eight chunks, 80 mm total, no grasp closure')
    p.add_argument('--action-row-dt-s',type=float,default=.04,choices=[.04,.2],
                   help='200 ms rows slow the approach by 5x; delta amplitudes remain unchanged')
    p.add_argument('--extended-approach',action='store_true',
                   help='Supervised slow approach: at most 12 chunks, 250 mm total and 30 seconds')
    p.add_argument('--shadow-constant-diagnostic', action='store_true',
                   help='Prediction-only constant-feature ablation; incompatible with any motion flag')
    p.add_argument('--confirm-parked-setup', action='store_true',
                   help='Operator confirms current camera/parked arm/rail setup; calibrate for 3 s in this session')
    p.add_argument('--align-to-training-hemisphere', action='store_true',
                   help='Choose the equivalent quaternion sign nearest the training mean; requires calibrated path')
    p.add_argument('--max-chunks', type=int, default=1)
    p.add_argument('--action-rows', type=int, default=1,
                   help='Predicted prefix to publish; first commissioning test uses ONE 40 ms row')
    p.add_argument('--max-seconds', type=float, default=10)
    p.add_argument('--run-seconds', type=float, default=30,
                   help='Bounded client lifetime after model warmup (default 30 s)')
    p.add_argument('--seed', type=int, default=20260912)
    args = p.parse_args()
    if sum(bool(x) for x in (args.execute_session,args.one_row_authorization_file,
                            args.bounded_rollout_authorization_file)) > 1:
        p.error('Choose exactly one session authorization mode')
    if bool(args.execute_session or args.one_row_authorization_file or args.bounded_rollout_authorization_file) != args.confirm_supervised_trial:
        p.error('Motion requires a session/grant AND --confirm-supervised-trial')
    if args.shadow_constant_diagnostic and (args.execute_session or args.one_row_authorization_file
                                           or args.bounded_rollout_authorization_file):
        p.error('Constant-feature diagnostic forbids motion authorization')
    if args.confirm_parked_setup and args.shadow_constant_diagnostic:
        p.error('Use calibrated parked inputs OR the shadow diagnostic, not both')
    if args.align_to_training_hemisphere and not args.confirm_parked_setup:
        p.error('Hemisphere correction requires --confirm-parked-setup')
    if args.supervised_approach and not args.bounded_rollout_authorization_file:
        p.error('Approach requires a fresh bounded-rollout session grant')
    if not 0 < args.run_seconds <= 600:
        p.error('--run-seconds must be in (0,600]')
    logging.basicConfig(level=logging.INFO)
    info = get_json(args.url+'/api/dora')
    if info['state'] != 'attached' or 'policy' not in info['placeholders']:
        raise RuntimeError('No existing policy placeholder; do not change the dataflow')
    # The shared runtime is NEVER modified, restarted, or connected directly.
    os.environ['DORA_ZENOH_CONNECT'] = info['zenoh_connect']
    os.environ['DORA_ZENOH_LISTEN'] = 'tcp/127.0.0.1:0'
    os.environ['DORA_ZENOH_MULTICAST'] = 'off'
    import torch
    torch.set_num_threads(4)
    policy = ApolloDoraPolicy(args.checkpoint, device=args.device, output=args.output,
                              execute_session=args.execute_session, max_chunks=args.max_chunks,
                              max_seconds=args.max_seconds, action_rows=args.action_rows,
                              shadow_constant_diagnostic=args.shadow_constant_diagnostic,
                              confirm_parked_setup=args.confirm_parked_setup,
                              align_to_training_hemisphere=args.align_to_training_hemisphere,
                              bounded_rollout=bool(args.bounded_rollout_authorization_file),
                              supervised_approach=args.supervised_approach,
                              action_row_dt_s=args.action_row_dt_s,
                              extended_approach=args.extended_approach)
    if args.one_row_authorization_file:
        policy.await_one_row_authorization(args.one_row_authorization_file)
    if args.bounded_rollout_authorization_file:
        policy.await_bounded_rollout_authorization(args.bounded_rollout_authorization_file)
    # Warm up before attaching, so first-use CUDA initialization never consumes
    # the freshness window of a live observation. No robot/Dora call occurs here.
    with np.load(args.checkpoint/'prediction_fixture.npz') as fixture:
        for _ in range(3):
            torch.manual_seed(int(fixture['seed']))
            actions = policy.predictor.predict_chunk(fixture['state'], fixture['view_wrist'],
                                                      fixture['grip_wrist'])
        (args.output/'warmup.json').write_text(json.dumps({
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(torch.device(args.device)),
            'max_reference_difference_by_channel': np.abs(actions-fixture['expected_action']).max(0).tolist(),
            'matches_original_strict_tolerance': bool(np.allclose(actions,fixture['expected_action'],
                                                                 atol=1e-5,rtol=.005)),
            'seed_live': args.seed, 'actions_published': 0,
        }, indent=2)+'\n')
    policy.predictor.reset()
    torch.manual_seed(args.seed)
    if args.execute_session:
        active = get_json(args.url+'/api/session')
        if active.get('session_id') != args.execute_session:
            raise RuntimeError('Requested session is not the current session')
        if (active.get('kind') != 'hardware' or active.get('mode') != 'inference'
                or active.get('policy_source') != 'external'):
            raise RuntimeError('Requires an existing hardware/external/inference session')
        if not 0 < float(active.get('speed_scale', 1)) <= .1:
            raise RuntimeError('First deployment code requires the operator to select 10% speed')
        policy.deadline = time.monotonic()+args.max_seconds
    node = GuardedPolicyNode(policy, daemon_port=info['daemon_port'], rate_hz=1/policy.chunk_dt_s,
                             max_image_age_s=.2, chunk_dt_s=policy.chunk_dt_s, device=args.device,
                             client='simba-vla-hardware')

    def stop(signum, frame):
        policy.disarm('operator stopped this client')
        node.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    timer = threading.Timer(args.run_seconds, stop, args=(None,None))
    timer.daemon = True
    timer.start()
    print('AWAITING EXPLICIT BOUNDED ROLLOUT GRANT' if args.bounded_rollout_authorization_file else
          'AWAITING EXPLICIT ONE-ROW SESSION GRANT' if args.one_row_authorization_file else
          ('SHADOW ONLY' if not args.execute_session else 'EXPLICIT BOUNDED SUPERVISED TRIAL'), flush=True)
    try:
        node.run()
    finally:
        timer.cancel()
        from dataclasses import asdict
        (args.output/'node_report.json').write_text(json.dumps({
            'stats': asdict(node.stats), 'predictions': policy.predictions,
            'chunks_returned': policy.returned_chunks, 'execute_session': policy.execute_session,
            'authorization_file': str(args.one_row_authorization_file) if args.one_row_authorization_file else None,
            'bounded_rollout_authorization_file': str(args.bounded_rollout_authorization_file)
                if args.bounded_rollout_authorization_file else None,
            'translation_path_used_m': policy.translation_path_used,
            'supervised_approach': policy.supervised_approach,
            'extended_approach': policy.extended_approach,
            'translation_budget_m': policy.translation_budget,
            'rotation_path_used_rad': policy.rotation_path_used,
            'disarmed': policy.disarmed, 'runtime_api_writes': 0,
            'constant_feature_diagnostic': args.shadow_constant_diagnostic,
            'wire_action_rate_hz': node.rate_hz,
            'action_row_dt_s':policy.chunk_dt_s,
            'max_model_prediction_rate_hz':1/(8*policy.chunk_dt_s),
            'max_inference_rate_hz': node.max_inference_rate_hz,
            'parked_calibration': policy.parked_adapter.report() if policy.parked_adapter else None,
        }, indent=2)+'\n')


if __name__ == '__main__':
    main()
