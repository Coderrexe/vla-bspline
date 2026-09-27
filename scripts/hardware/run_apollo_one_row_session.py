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
from apollo_task_guard import DRAWER_TASK_LIMITS, LAMP_TASK_LIMITS, LAMP_TASK_ROW_DT_S
from apollo_contact_review import require_contact_review_clear


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirm-supervised-one-row', action='store_true')
    p.add_argument('--confirm-supervised-bounded-rollout', action='store_true')
    p.add_argument('--confirm-supervised-gripper-check',action='store_true')
    p.add_argument('--gripper-direction',choices=['close','open'],default='close')
    p.add_argument('--reviewed-absolute-gripper-check', action='store_true')
    p.add_argument('--max-chunks',type=int,default=1)
    p.add_argument('--action-rows',type=int,default=1)
    p.add_argument('--supervised-approach',action='store_true')
    p.add_argument('--supervised-lamp-approach',action='store_true')
    p.add_argument('--lamp-approach-review',type=Path)
    p.add_argument('--supervised-lamp-task',action='store_true')
    p.add_argument('--lamp-task-review',type=Path)
    p.add_argument('--initialize-lamp-start', action='store_true')
    p.add_argument('--lamp-absolute-execution', action='store_true')
    p.add_argument('--action-row-dt-s',type=float,default=.04,
                   choices=[.04,LAMP_TASK_ROW_DT_S,.2,.4])
    p.add_argument('--extended-approach',action='store_true')
    p.add_argument('--supervised-grasp',action='store_true')
    p.add_argument('--supervised-drawer-task',action='store_true')
    p.add_argument('--gripper-reference-slew',action='store_true')
    p.add_argument('--model-gripper-key-bridge', action='store_true',
                   help='Mirror the audited lamp grasp phase to the runtime gripper held-key lane')
    p.add_argument('--expected-policy-id', required=True)
    p.add_argument('--policy-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:8765')
    args = p.parse_args()
    # This launcher always enables hardware: block before any API request,
    # session creation, output/grant creation, or controller transition.
    require_contact_review_clear(
        lamp_approach_review=args.lamp_approach_review if args.supervised_lamp_approach else None,
        lamp_task_review=args.lamp_task_review
            if (args.supervised_lamp_task or args.reviewed_absolute_gripper_check) else None,
        policy_id=args.expected_policy_id)
    if bool(args.lamp_approach_review) != args.supervised_lamp_approach:
        p.error('Lamp review requires its distinct approach profile')
    if bool(args.lamp_task_review) != (
            args.supervised_lamp_task or args.reviewed_absolute_gripper_check):
        p.error('Full lamp review requires its distinct task profile')
    if args.reviewed_absolute_gripper_check and (
            not args.confirm_supervised_gripper_check or args.supervised_lamp_task
            or (args.max_chunks, args.action_rows) not in {(1, 1), (3, 8)}
            or args.action_row_dt_s != .04 or args.gripper_direction != 'close'):
        p.error('Reviewed absolute gripper check is one row or three fixed-target chunks at native timing')
    if args.supervised_lamp_approach and args.supervised_lamp_task:
        p.error('Choose the lamp approach or full lamp task')
    if args.lamp_absolute_execution and not (args.supervised_lamp_approach or args.supervised_lamp_task):
        p.error('Absolute execution requires the reviewed lamp profile')
    if args.initialize_lamp_start:
        if not (args.supervised_lamp_approach or args.supervised_lamp_task):
            p.error('Recorded-start initialization is only for the reviewed lamp test')
        from apollo_lamp_initialization import validate_reset_review
        review_path = args.lamp_task_review if args.supervised_lamp_task else args.lamp_approach_review
        validate_reset_review(json.loads(review_path.read_text()))
    if args.supervised_lamp_approach:
        from apollo_lamp_approach import validate_lamp_approach_options
        validate_lamp_approach_options(args)
        if not args.confirm_supervised_bounded_rollout:
            p.error('Lamp approach requires explicit bounded rollout')
    if args.supervised_lamp_task and (not args.confirm_supervised_bounded_rollout
            or not args.initialize_lamp_start
            or args.supervised_approach or args.extended_approach or args.supervised_grasp
            or args.supervised_drawer_task or args.action_row_dt_s !=
                (LAMP_TASK_ROW_DT_S if args.lamp_absolute_execution else .04)
            or args.action_rows != 8 or not 1 <= args.max_chunks <= 100):
        p.error('Full lamp task requires recorded initialization and native-delta or 15 Hz absolute chunks')
    if args.gripper_direction != 'close' and not args.confirm_supervised_gripper_check:
        p.error('Gripper direction applies only to the deterministic gripper check')
    if sum((args.confirm_supervised_one_row,args.confirm_supervised_bounded_rollout,
            args.confirm_supervised_gripper_check)) != 1:
        p.error('Choose one explicit supervised motion authorization')
    if args.supervised_approach and not args.confirm_supervised_bounded_rollout:
        p.error('Approach requires explicit bounded-rollout authorization')
    if args.supervised_grasp and (not args.confirm_supervised_bounded_rollout
            or args.supervised_approach or args.extended_approach or args.action_row_dt_s != .2):
        p.error('Grasp stage requires its own bounded profile and 200 ms rows')
    if args.supervised_drawer_task and (not args.confirm_supervised_bounded_rollout
            or args.supervised_approach or args.extended_approach or args.supervised_grasp
            or args.action_row_dt_s not in (.2,.4) or args.action_rows != 8):
        p.error('Drawer task requires its own bounded profile and eight 200 or 400 ms rows')
    if args.gripper_reference_slew and not (args.supervised_drawer_task or args.supervised_lamp_task):
        p.error('Gripper reference slew requires a supervised full-task profile')
    if args.model_gripper_key_bridge and not (
            args.supervised_lamp_task and args.gripper_reference_slew):
        p.error('Model gripper bridge is only for the reviewed lamp task with phase filtering')
    if args.action_row_dt_s == .4 and not (args.supervised_drawer_task or args.supervised_lamp_task
                                          or (args.supervised_lamp_approach and args.lamp_absolute_execution)):
        p.error('400 ms rows require the supervised drawer task')
    if args.extended_approach and (not args.supervised_approach or args.action_row_dt_s != .2):
        p.error('Extended approach requires the reviewed slow approach clock')
    if args.action_row_dt_s != .04 and (not (args.supervised_approach or args.supervised_grasp
                                             or args.supervised_drawer_task or args.supervised_lamp_task)
            or args.max_chunks > (100 if args.supervised_lamp_task else 150 if args.supervised_drawer_task
                                  else 12 if args.extended_approach or args.supervised_grasp else 3)):
        p.error('Slow rows require the approach profile and at most three chunks')
    chunk_cap=(100 if args.supervised_lamp_task else 150 if args.supervised_drawer_task else
               12 if args.extended_approach or args.supervised_grasp else (8 if args.supervised_approach else 3))
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
    if args.supervised_lamp_approach:purpose='SUPERVISED_LAMP_APPROACH'
    if args.supervised_lamp_task:purpose='SUPERVISED_LAMP_TASK'
    if args.extended_approach:purpose='SUPERVISED_EXTENDED_APPROACH'
    if args.supervised_grasp:purpose='SUPERVISED_GRASP_STAGE'
    if args.supervised_drawer_task:purpose='SUPERVISED_DRAWER_TASK'
    if args.gripper_reference_slew:
        purpose='SUPERVISED_LAMP_TASK_SLEW' if args.supervised_lamp_task else 'SUPERVISED_DRAWER_TASK_SLEW'
    if args.confirm_supervised_gripper_check:purpose='SUPERVISED_GRIPPER_CHECK'
    if args.confirm_supervised_gripper_check and args.max_chunks==3:purpose='SUPERVISED_GRIPPER_HOLD_CHECK'
    if args.confirm_supervised_gripper_check and args.gripper_direction=='open':
        purpose=('SUPERVISED_GRIPPER_OPEN_HOLD_CHECK' if args.max_chunks==3
                 else 'SUPERVISED_GRIPPER_OPEN_CHECK')
    grant_seconds=(120 if args.supervised_lamp_task else
                   (550 if args.action_row_dt_s == .4 else 300) if args.supervised_drawer_task else
                   30 if args.extended_approach or args.supervised_grasp else 10)
    if args.supervised_lamp_approach and args.action_row_dt_s == .4:
        grant_seconds = 20
    report = {'purpose':purpose, 'session_create_posts':0, 'session_delete_requests':0,
              'grant_written':False, 'max_chunks':args.max_chunks, 'action_rows':args.action_rows,
              'chunk_dt_s':args.action_row_dt_s}
    own_session = None
    reset_inflight = False
    gripper_bridge_stop = asyncio.Event()
    gripper_bridge_ready = asyncio.Event()
    gripper_bridge_task = None

    async def model_gripper_bridge(session_id):
        """Use the runtime's proven held-key gripper lane for the model's phase.

        The policy's arm actions remain on Dora.  Only ``KeyF``/``KeyH`` are
        emitted, at the UI heartbeat rate, and every phase transition is logged.
        The first-controller check prevents silently running as an observer.
        """
        control_url = args.url.replace('http://','ws://').replace('https://','wss://')+'/ws/control'
        prediction_path = args.policy_output/'predictions.jsonl'
        offset = 0
        phase = None
        last_held = None
        seq = 0
        async with websockets.connect(control_url, open_timeout=5) as control:
            hello = json.loads(await asyncio.wait_for(control.recv(), timeout=5))
            if (hello.get('t') != 'hello' or hello.get('role') != 'controller'
                    or hello.get('session_id') != session_id):
                raise RuntimeError(f'Gripper bridge did not receive controller role: {hello}')
            gripper_bridge_ready.set()
            with (args.output/'gripper_bridge.jsonl').open('x') as audit:
                while not gripper_bridge_stop.is_set():
                    if prediction_path.exists():
                        with prediction_path.open() as stream:
                            stream.seek(offset)
                            for line in stream:
                                try:
                                    item = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                if item.get('publication_requested'):
                                    phase = item.get('lamp_gripper_phase')
                            offset = stream.tell()
                    held = (['KeyF'] if phase in {'closing','closed'} else
                            ['KeyH'] if phase == 'opening' else [])
                    seq += 1
                    await control.send(json.dumps({
                        't':'keys', 'seq':seq, 'ts':time.time(), 'held':held,
                    }))
                    if held != last_held:
                        audit.write(json.dumps({
                            't_mono':time.monotonic(), 'phase':phase, 'held':held,
                        })+'\n')
                        audit.flush()
                        last_held = held
                    await asyncio.sleep(.04)
                seq += 1
                await control.send(json.dumps({
                    't':'keys', 'seq':seq, 'ts':time.time(), 'held':[],
                }))
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
            if args.model_gripper_key_bridge and before.get('controller_connected'):
                raise RuntimeError(
                    'Control websocket is already occupied; close the Cockpit/control '
                    'client before starting the model gripper bridge')
            ext = before.get('external') or {}
            if (not ext.get('policy_attached') or ext.get('policy_id') != args.expected_policy_id
                    or ext.get('policy_arms') != ['grip'] or ext.get('spec_age_s',99) > 1.5
                    or abs(float(ext.get('policy_rate_hz',0))-1/args.action_row_dt_s) > 1e-6):
                raise RuntimeError('Expected fresh grip-only policy is not attached')
            if (args.policy_output/'consumed_authorization.json').exists():
                raise RuntimeError('Policy output already contains a consumed authorization')
            body = {'mode':'inference', 'kind':'hardware', 'arms':['grip','view'],
                    'frames':{'grip':'arm_base:grip','view':'arm_base:view'},
                    'digital_twin_scene':'mavis_v2', 'start_from':'keep_current',
                    'speed_scale':.6 if args.supervised_lamp_task else .1,
                    'policy_source':'external', 'policy':None,
                    'task':'Lamp Assembling' if (args.supervised_lamp_approach or args.supervised_lamp_task)
                    else 'Drawer Assembling'}
            if current_session(args.url) is not None:
                raise RuntimeError('A session appeared during preflight')
            report['request'] = body
            report['session_create_posts'] += 1
            print(f'Starting approved hardware session; at most {args.max_chunks} chunks of {args.action_rows} guarded rows', flush=True)
            created = await asyncio.to_thread(request, args.url+'/api/session', 'POST', body)
            own_session = created['session_id']
            report['created'] = created
            if args.model_gripper_key_bridge:
                gripper_bridge_task = asyncio.create_task(model_gripper_bridge(own_session))
                ready_wait = asyncio.create_task(gripper_bridge_ready.wait())
                done, _ = await asyncio.wait(
                    {gripper_bridge_task, ready_wait}, timeout=5,
                    return_when=asyncio.FIRST_COMPLETED)
                if gripper_bridge_task in done:
                    await gripper_bridge_task
                if ready_wait not in done:
                    ready_wait.cancel()
                    raise TimeoutError('Gripper bridge did not become controller')
                report['model_gripper_key_bridge'] = True
            if args.initialize_lamp_start:
                from apollo_lamp_initialization import (
                    LAMP_REPO, LAMP_EPISODE, validate_recorded_start,
                    validate_reset_telemetry, validate_reset_transit_telemetry)
                info = request(args.url+f'/api/datasets/{LAMP_REPO}/episodes/{LAMP_EPISODE}/playback')
                validate_recorded_start(info)
                report['recorded_initial'] = info

                async def fresh_running(deadline, *, reset_transit=False):
                    while time.monotonic() < deadline:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
                        if ((msg.get('session') or {}).get('session_id') != own_session
                                or time.monotonic()-float(msg.get('ts', 0)) > .25):
                            continue
                        state = (msg.get('session') or {}).get('state')
                        if state in {'fault', 'recovering', 'teardown'}:
                            raise RuntimeError('Session fault during initialization')
                        if state == 'running':
                            (validate_reset_transit_telemetry if reset_transit
                             else validate_reset_telemetry)(msg, own_session)
                            return msg
                    raise TimeoutError('No fresh healthy initialization telemetry')

                # A prior interrupted initialization may leave the arm at any
                # point on the reviewed replay-final to recorded-start path.
                # Admit that corridor here; the post-reset handoff below still
                # requires the strict recorded endpoint tolerance.
                before_reset = await fresh_running(time.monotonic()+20, reset_transit=True)
                report['before_recorded_reset'] = before_reset
                reset_inflight = True
                print('Restoring the approved lamp recording start through the existing planned reset API; policy inactive', flush=True)
                operation = asyncio.create_task(asyncio.to_thread(
                    request, args.url+'/api/session/playback', 'POST',
                    {'repo_id': LAMP_REPO, 'episode_id': LAMP_EPISODE, 'action': 'goto_initial'}))
                reset_deadline = time.monotonic()+35
                with (args.output/'initialization_telemetry.jsonl').open('x') as trace:
                    while not operation.done():
                        msg = await fresh_running(reset_deadline, reset_transit=True)
                        trace.write(json.dumps(msg)+'\n'); trace.flush()
                    reset_result = await operation
                    report['recorded_reset_result'] = reset_result
                    if not reset_result.get('ok'):
                        raise RuntimeError(f'Recorded-start reset refused: {reset_result}')
                    # Drain the planned handback event before admitting model inputs.
                    settle_until = time.monotonic()+1.5
                    while time.monotonic() < settle_until:
                        msg = await fresh_running(settle_until+2)
                        trace.write(json.dumps(msg)+'\n'); trace.flush()
                validate_reset_telemetry(msg, own_session, arrived=True)
                reset_inflight = False
                report['after_recorded_reset'] = msg
                review_path = args.lamp_task_review if args.supervised_lamp_task else args.lamp_approach_review
                review = json.loads(review_path.read_text())
                receipt = {'purpose': 'LAMP_RECORDED_START_VERIFIED', 'session_id': own_session,
                           'epoch': created['epoch'], 'review_id': review['review_id'],
                           'expires_t_mono': time.monotonic()+15}
                receipt_source = args.output/'initialization_ready_source.json'
                with receipt_source.open('x') as f:
                    f.write(json.dumps(receipt, indent=2)+'\n')
                os.link(receipt_source, args.output/'initialization_ready.json')
                print('Recorded starting pose verified; enabling the bounded policy trial without restarting hardware', flush=True)
            grant = {'purpose':purpose, 'session_id':own_session,
                     'epoch':created['epoch'], 'policy_id':args.expected_policy_id,
                     'action_rows':args.action_rows, 'max_chunks':args.max_chunks,
                     'chunk_dt_s':args.action_row_dt_s,
                     'expires_t_mono':time.monotonic()+grant_seconds}
            if args.confirm_supervised_bounded_rollout or purpose in {
                    'SUPERVISED_GRIPPER_HOLD_CHECK','SUPERVISED_GRIPPER_OPEN_HOLD_CHECK'}:
                grant.update(total_translation_path_m=.25 if args.extended_approach or args.supervised_grasp else
                             (.08 if args.supervised_approach else .01),
                             total_rotation_path_rad=.15)
                if args.supervised_grasp:
                    grant.update(first_gripper_target_change=.1, row_gripper_target_change=.06,
                                 prefix_gripper_target_path=.45, total_gripper_target_path=1.)
                else:
                    grant.update(total_gripper_change=.1)
            if args.supervised_approach or args.supervised_grasp:
                grant.update(prefix_translation_path_m=.04,row_translation_m=.006)
            if args.supervised_lamp_approach:
                from apollo_lamp_approach import LAMP_APPROACH_LIMITS
                review = json.loads(args.lamp_approach_review.read_text())
                grant.update(lamp_review_id=review['review_id'], lamp_approach_limits=LAMP_APPROACH_LIMITS)
                grant['wire_action_space'] = 'abs_ee' if args.lamp_absolute_execution else 'delta_ee'
            if args.supervised_lamp_task:
                review = json.loads(args.lamp_task_review.read_text())
                grant.pop('total_gripper_change', None)
                grant.update(LAMP_TASK_LIMITS)
                grant.update(lamp_review_id=review['review_id'],
                             lamp_task_limits=LAMP_TASK_LIMITS,
                             wire_action_space=('abs_ee' if args.lamp_absolute_execution
                                                else 'delta_ee'))
            if args.supervised_drawer_task:
                grant.pop('total_gripper_change', None)
                grant.update(DRAWER_TASK_LIMITS)
            if args.gripper_reference_slew:
                grant.update(gripper_reference_slew=True,
                             gripper_reference_max_step=.05 if args.supervised_lamp_task else .06)
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
                    if args.lamp_absolute_execution:
                        reference = report.get('after_recorded_reset', report['before'])
                        expected_rails = {x['arm_id']: x['rail_pos_m'] for x in reference['arms']}
                        if any(abs(x['rail_pos_m']-expected_rails[x['arm_id']]) > .0002
                               for x in message['arms']):
                            raise RuntimeError('Unexpected rail movement during absolute approach')
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
            if gripper_bridge_task is not None:
                gripper_bridge_stop.set()
                if not gripper_bridge_task.done():
                    try:
                        await asyncio.wait_for(gripper_bridge_task, timeout=2)
                    except Exception as exc:
                        report['gripper_bridge_error'] = repr(exc)
                elif not gripper_bridge_task.cancelled():
                    exc = gripper_bridge_task.exception()
                    if exc is not None:
                        report['gripper_bridge_error'] = repr(exc)
            if own_session:
                active = current_session(args.url)
                if active and active['session_id'] == own_session:
                    try:
                        if reset_inflight:
                            request(args.url+'/api/session/playback', 'POST',
                                    {'repo_id':'bc_demo/lamp_assembling',
                                     'episode_id':'20260911T204950.391Z-538dc8', 'action':'stop'})
                    finally:
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
