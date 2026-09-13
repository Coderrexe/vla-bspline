"""Supervised finite learned motion, existing runtime only; no SDK or driver edits.

Run only after operator approval. Requires our waiting policy, creates exactly
one keep-current hardware session, issues its short-lived session-specific grant,
records response telemetry, and deletes only its own session without return-home.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

import websockets

from check_apollo_shadow_session import request, current_session


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirm-supervised-one-row', action='store_true')
    p.add_argument('--confirm-supervised-bounded-rollout', action='store_true')
    p.add_argument('--confirm-supervised-gripper-check',action='store_true')
    p.add_argument('--max-chunks',type=int,default=1)
    p.add_argument('--action-rows',type=int,default=1)
    p.add_argument('--supervised-approach',action='store_true')
    p.add_argument('--action-row-dt-s',type=float,default=.04,choices=[.04,.2])
    p.add_argument('--extended-approach',action='store_true')
    p.add_argument('--expected-policy-id', required=True)
    p.add_argument('--policy-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:8765')
    args = p.parse_args()
    if sum((args.confirm_supervised_one_row,args.confirm_supervised_bounded_rollout,
            args.confirm_supervised_gripper_check)) != 1:
        p.error('Choose one explicit supervised motion authorization')
    if args.supervised_approach and not args.confirm_supervised_bounded_rollout:
        p.error('Approach requires explicit bounded-rollout authorization')
    if args.extended_approach and (not args.supervised_approach or args.action_row_dt_s != .2):
        p.error('Extended approach requires the reviewed slow approach clock')
    if args.action_row_dt_s != .04 and (not args.supervised_approach or args.max_chunks > (12 if args.extended_approach else 3)):
        p.error('Slow rows require the approach profile and at most three chunks')
    chunk_cap=12 if args.extended_approach else (8 if args.supervised_approach else 3)
    if not 1 <= args.max_chunks <= chunk_cap or not 1 <= args.action_rows <= 8:
        p.error('Chunk count exceeds the selected short-trial profile')
    if args.confirm_supervised_one_row and (args.max_chunks != 1 or args.action_rows != 1):
        p.error('One-row mode cannot authorize a larger rollout')
    if args.confirm_supervised_gripper_check and ((args.max_chunks,args.action_rows) not in {(1,1),(3,8)}
            or args.action_row_dt_s != .04 or args.supervised_approach or args.extended_approach):
        p.error('Gripper check permits one row or three eight-row fixed-target chunks, native timing only')
    args.output.mkdir(parents=True, exist_ok=False)
    purpose='SUPERVISED_BOUNDED_ROLLOUT' if args.confirm_supervised_bounded_rollout else 'SUPERVISED_ONE_ROW'
    if args.supervised_approach:purpose='SUPERVISED_APPROACH'
    if args.extended_approach:purpose='SUPERVISED_EXTENDED_APPROACH'
    if args.confirm_supervised_gripper_check:purpose='SUPERVISED_GRIPPER_CHECK'
    if args.confirm_supervised_gripper_check and args.max_chunks==3:purpose='SUPERVISED_GRIPPER_HOLD_CHECK'
    grant_seconds=30 if args.extended_approach else 10
    report = {'purpose':purpose, 'session_create_posts':0, 'session_delete_requests':0,
              'grant_written':False, 'max_chunks':args.max_chunks, 'action_rows':args.action_rows,
              'chunk_dt_s':args.action_row_dt_s}
    own_session = None
    try:
        if current_session(args.url) is not None:
            raise RuntimeError('An existing session is present; leaving it untouched')
        wc = request(args.url+'/api/workcell?kind=hardware')
        if not wc.get('hardware_ready') or not wc.get('policy_modes') or any(
                a['error_code'] or a['connected'] or a['reachable'] != 'open' for a in wc['arms']):
            raise RuntimeError('Hardware is not healthy and idle')
        ws_url = args.url.replace('http://','ws://').replace('https://','wss://')+'/ws/telemetry'
        async with websockets.connect(ws_url, open_timeout=5) as ws:
            before = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            report['before'] = before
            ext = before.get('external') or {}
            if (not ext.get('policy_attached') or ext.get('policy_id') != args.expected_policy_id
                    or ext.get('policy_arms') != ['grip'] or ext.get('spec_age_s',99) > 1.5
                    or abs(float(ext.get('policy_rate_hz',0))-1/args.action_row_dt_s) > 1e-6):
                raise RuntimeError('Expected fresh grip-only policy is not attached')
            if (args.policy_output/'consumed_authorization.json').exists():
                raise RuntimeError('Policy output already contains a consumed authorization')
            body = {'mode':'inference', 'kind':'hardware', 'arms':['grip','view'],
                    'frames':{'grip':'arm_base:grip','view':'arm_base:view'},
                    'digital_twin_scene':'mavis_v2', 'start_from':'keep_current', 'speed_scale':.1,
                    'policy_source':'external', 'policy':None, 'task':'Drawer Assembling'}
            if current_session(args.url) is not None:
                raise RuntimeError('A session appeared during preflight')
            report['request'] = body
            report['session_create_posts'] += 1
            print(f'Starting approved hardware session; at most {args.max_chunks} chunks of {args.action_rows} guarded rows', flush=True)
            created = await asyncio.to_thread(request, args.url+'/api/session', 'POST', body)
            own_session = created['session_id']
            report['created'] = created
            grant = {'purpose':purpose, 'session_id':own_session,
                     'epoch':created['epoch'], 'policy_id':args.expected_policy_id,
                     'action_rows':args.action_rows, 'max_chunks':args.max_chunks,
                     'chunk_dt_s':args.action_row_dt_s,
                     'expires_t_mono':time.monotonic()+grant_seconds}
            if args.confirm_supervised_bounded_rollout or purpose=='SUPERVISED_GRIPPER_HOLD_CHECK':
                grant.update(total_translation_path_m=.25 if args.extended_approach else
                             (.08 if args.supervised_approach else .01),
                             total_rotation_path_rad=.15,total_gripper_change=.1)
            if args.supervised_approach:
                grant.update(prefix_translation_path_m=.04,row_translation_m=.006)
            # Atomic, exclusive publication: never expose a partly written JSON
            # and never overwrite an earlier grant. Keep the source as an audit.
            source = args.output/'authorization_source.json'
            with source.open('x') as f:
                f.write(json.dumps(grant,indent=2)+'\n')
            os.link(source, args.output/'authorization.json')
            report['grant_written'] = True
            print(json.dumps({'session_id':own_session, 'grant_expires_t_mono':grant['expires_t_mono']}), flush=True)
            deadline = time.monotonic()+grant_seconds+5
            first_action_received_at = None
            rows = 0
            with (args.output/'telemetry.jsonl').open('x') as output:
                while time.monotonic() < deadline:
                    message = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
                    s = message.get('session') or {}
                    if s.get('session_id') != own_session:
                        continue
                    output.write(json.dumps(message)+'\n')
                    output.flush()
                    rows += 1
                    report['last'] = message
                    if s.get('state') in {'fault','recovering','teardown'}:
                        raise RuntimeError('Session left healthy running state')
                    if s.get('state') != 'running':
                        continue
                    if message.get('collision',{}).get('blocked') or any(
                            a.get('error_code') or a.get('stale') or a.get('recovering')
                            for a in message.get('arms', [])):
                        raise RuntimeError('Hardware/collision check interrupted the trial')
                    if (args.policy_output/'refusals.jsonl').exists():
                        report['status'] = 'POLICY_REFUSED_NO_RETRY'
                        break
                    ext = message.get('external') or {}
                    if ext.get('action_age_s') is not None and first_action_received_at is None:
                        first_action_received_at = time.monotonic()
                        report['first_action_telemetry'] = message
                        print('Runtime reports action receipt; observing response, no further grant', flush=True)
                    # Prediction callbacks are sampled at ~0.33 s and computation
                    # may take 0.4 s. Allow the last accepted chunk to finish,
                    # including callback quantization; never cut it at an ideal
                    # cadence deadline. The independent wall-clock limit remains.
                    response_window = max(3,args.max_chunks*(8*args.action_row_dt_s+.35)+.4)
                    if first_action_received_at is not None and time.monotonic()-first_action_received_at >= response_window:
                        report['status'] = 'ACTION_RECEIPT_RESPONSE_RECORDED'
                        break
            report['telemetry_frames'] = rows
            report.setdefault('status', 'NO_ACTION_RECEIPT')
    except Exception as exc:
        report['error'] = repr(exc)
        report['status'] = 'REFUSED_OR_ERROR'
        raise
    finally:
        try:
            if own_session:
                active = current_session(args.url)
                if active and active['session_id'] == own_session:
                    request(args.url+'/api/session', 'DELETE')
                    report['session_delete_requests'] += 1
                    print('Closed our session without return-home', flush=True)
                report['session_after_cleanup'] = current_session(args.url)
        finally:
            (args.output/'session_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'], flush=True)
    return 0 if report['status'] == 'ACTION_RECEIPT_RESPONSE_RECORDED' else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
