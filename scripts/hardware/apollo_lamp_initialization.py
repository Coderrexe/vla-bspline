"""Checks for one operator-approved recorded-start reset; no hardware imports."""
import json
from pathlib import Path

import numpy as np

LAMP_REPO = 'bc_demo/lamp_assembling'
LAMP_EPISODE = '20260911T204950.391Z-538dc8'
LAMP_INITIAL = {
    'grip': {'q': [-.27, .888827, .471966, 2.263026, -.275096, 1.457, -1.559163],
             'rail_pos_m': .636, 'gripper_open_frac': .976190447807312},
    'view': {'q': [.491, .111898, .055438, .954684, 1.115541, .830775, -1.976937],
             'rail_pos_m': 0., 'gripper_open_frac': 1.},
}

# Trial 009's reviewed 24-row, absolute-EE control remained inside the lamp
# approach box but changed one manipulation-arm joint by 0.197896 rad. Admit
# that known high-clearance endpoint to the planned reset. The perception arm
# never moved and retains the original 0.08 rad pre-reset bound. Arrival at the
# recorded start remains the much tighter 0.003 rad check below.
LAMP_PRE_RESET_Q_TOLERANCE = {'grip': .22, 'view': .08}
LAMP_VERIFIED_REPLAY_FINAL = {
    'q': [.007457, .988845, .510106, 2.269397, -.338061, 1.362147, -1.259047],
    'rail_pos_m': .636,
    'tcp_min_m': [.64, .16, .10], 'tcp_max_m': [.69, .21, .15],
}


def validate_reset_review(review):
    if (review.get('operator_confirmed_recorded_start_reset') is not True
            or review.get('reset_repo_id') != LAMP_REPO
            or review.get('reset_episode_id') != LAMP_EPISODE):
        raise ValueError('Recorded-start reset requires its own operator clearance')


def validate_recorded_start(info):
    if (info.get('repo_id') != LAMP_REPO or info.get('episode_id') != LAMP_EPISODE
            or info.get('frames') != 521 or info.get('fps') != 25.):
        raise ValueError('Wrong lamp recording for initialization')
    arms = {x['arm_id']: x for x in info['arms']}
    if set(arms) != set(LAMP_INITIAL):
        raise ValueError('Wrong recorded arms')
    for name, expected in LAMP_INITIAL.items():
        for key, value in expected.items():
            actual = np.asarray(arms[name][key])
            if actual.shape != np.asarray(value).shape or not np.allclose(actual, value, atol=1e-6, rtol=0):
                raise ValueError(f'Recorded starting configuration changed: {name}/{key}')


def validate_reset_telemetry(message, session_id, *, arrived=False):
    s = message.get('session') or {}
    if s.get('session_id') != session_id or s.get('state') != 'running':
        raise ValueError('Initialization requires the same healthy running session')
    if message.get('collision', {}).get('blocked'):
        raise ValueError('Collision guard blocked initialization')
    if (message.get('external') or {}).get('action_age_s') is not None:
        raise ValueError('Unexpected learned action during initialization')
    arms = {x['arm_id']: x for x in message['arms']}
    if set(arms) != set(LAMP_INITIAL):
        raise ValueError('Unexpected arm set during initialization')
    for name, expected in LAMP_INITIAL.items():
        arm = arms[name]
        q = np.asarray(arm['q'], dtype=float)
        rail = float(arm['rail_pos_m'])
        opening = float(arm['gripper_open_frac'])
        q_tolerance = .003 if arrived else LAMP_PRE_RESET_Q_TOLERANCE[name]
        q_ok = np.max(np.abs(q[:7]-expected['q'])) <= q_tolerance
        if not arrived and name == 'grip':
            # The reference absolute replay ended at this recorded, observed
            # high-clearance withdrawal pose. Admit that exact endpoint to the
            # planned return-to-start operation; do not broaden other states.
            q_ok = q_ok or np.max(np.abs(q[:7]-LAMP_VERIFIED_REPLAY_FINAL['q'])) <= .03
        if (arm.get('error_code') or arm.get('stale') or arm.get('recovering')
                or q.shape not in ((7,), (8,)) or not np.isfinite(q).all()
                or not np.isfinite(rail) or not .85 <= opening <= 1
                or not q_ok
                or abs(rail-expected['rail_pos_m']) > (.0015 if arrived else .003)):
            raise ValueError(f'{name} is outside the reviewed near-start reset region')
    xyz = np.asarray(arms['grip']['ee_pose']['position'], dtype=float)
    initial_box = (np.all(xyz >= [.63, -.05, .12]) and np.all(xyz <= [.70, .025, .22]))
    replay_box = (not arrived and np.all(xyz >= LAMP_VERIFIED_REPLAY_FINAL['tcp_min_m'])
                  and np.all(xyz <= LAMP_VERIFIED_REPLAY_FINAL['tcp_max_m']))
    if xyz.shape != (3,) or not np.isfinite(xyz).all() or not (initial_box or replay_box):
        raise ValueError('Initialization left its high-clearance TCP region')


def validate_reset_transit_telemetry(message, session_id):
    """Validate the existing API's planned replay-final to initial transition.

    The reset interpolates between two reviewed recorded configurations.  It is
    incorrect to require every intermediate sample to be close to either
    endpoint.  This check instead admits only their per-joint corridor with a
    small tracking margin, while retaining health, rail, gripper, collision,
    and high-clearance TCP checks.  Learned publication is still gated until
    ``validate_reset_telemetry(..., arrived=True)`` succeeds.
    """
    s = message.get('session') or {}
    if s.get('session_id') != session_id or s.get('state') != 'running':
        raise ValueError('Initialization transit requires the same healthy session')
    if message.get('collision', {}).get('blocked'):
        raise ValueError('Collision guard blocked initialization transit')
    if (message.get('external') or {}).get('action_age_s') is not None:
        raise ValueError('Unexpected learned action during initialization transit')
    arms = {x['arm_id']: x for x in message['arms']}
    if set(arms) != set(LAMP_INITIAL):
        raise ValueError('Unexpected arm set during initialization transit')
    for name, expected in LAMP_INITIAL.items():
        arm = arms[name]
        q = np.asarray(arm['q'], dtype=float)
        rail = float(arm['rail_pos_m'])
        opening = float(arm['gripper_open_frac'])
        start_q = np.asarray(expected['q'], dtype=float)
        if name == 'grip':
            other_q = np.asarray(LAMP_VERIFIED_REPLAY_FINAL['q'], dtype=float)
            q_min = np.minimum(start_q, other_q)-.04
            q_max = np.maximum(start_q, other_q)+.04
        else:
            q_min, q_max = start_q-.03, start_q+.03
        if (arm.get('error_code') or arm.get('stale') or arm.get('recovering')
                or q.shape not in ((7,), (8,)) or not np.isfinite(q).all()
                or np.any(q[:7] < q_min) or np.any(q[:7] > q_max)
                or not np.isfinite(rail) or abs(rail-expected['rail_pos_m']) > .003
                or not .85 <= opening <= 1):
            raise ValueError(f'{name} left the reviewed reset transit corridor')
    xyz = np.asarray(arms['grip']['ee_pose']['position'], dtype=float)
    if (xyz.shape != (3,) or not np.isfinite(xyz).all()
            or np.any(xyz < [.62, -.06, .08]) or np.any(xyz > [.71, .22, .23])):
        raise ValueError('Initialization transit left its high-clearance TCP corridor')


def initialization_ready(path, session, review_id, now):
    path = Path(path)
    if not path.exists():
        return False
    if path.is_symlink() or path.stat().st_size > 4096:
        raise ValueError('Invalid initialization receipt')
    receipt = json.loads(path.read_text())
    if (receipt.get('purpose') != 'LAMP_RECORDED_START_VERIFIED'
            or receipt.get('session_id') != session.get('session_id')
            or receipt.get('epoch') != session.get('epoch')
            or receipt.get('review_id') != review_id
            or not 0 < float(receipt['expires_t_mono'])-now <= 15):
        raise ValueError('Initialization receipt is stale or belongs to another session')
    return True
