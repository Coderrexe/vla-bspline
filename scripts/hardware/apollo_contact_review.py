"""Local launcher hold following reported contact; not a hardware safety device.

The marker is deployed beside our scripts. Its presence blocks our motion
launchers, even if malformed or a dangling symlink. There is deliberately no
command-line bypass. Read-only/offline diagnosis remains available. Clearing a
hold requires a separately reviewed lab recovery and geometry validation.
"""
import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path


CONTACT_REVIEW_MARKER = Path(__file__).with_name('apollo_contact_review_required.json')


def require_contact_review_clear(marker=None, *, lamp_approach_review=None,
                                 lamp_task_review=None, policy_id=None):
    if lamp_approach_review is not None and lamp_task_review is not None:
        raise RuntimeError('Choose one lamp review scope')
    lamp_review = lamp_task_review if lamp_task_review is not None else lamp_approach_review
    if lamp_review is not None:
        # A named, reviewed high-clearance test is not a general release of the
        # contact hold. Only the dedicated lamp approach launch paths use this.
        path = Path(lamp_review)
        if path.is_symlink() or path.stat().st_size > 4096:
            raise RuntimeError('Invalid lamp review file')
        review = json.loads(path.read_text())
        held = CONTACT_REVIEW_MARKER if marker is None else Path(marker)
        if (held.is_symlink() or not held.is_file()
                or hashlib.sha256(held.read_bytes()).hexdigest() != review.get('contact_marker_sha256')):
            raise RuntimeError('Contact hold changed; lamp review cannot be reused')
        now = datetime.now(timezone.utc)
        start = datetime.fromisoformat(review['confirmed_at_utc'])
        end = datetime.fromisoformat(review['valid_until_utc'])
        expected_scope = ('SUPERVISED_LAMP_TASK' if lamp_task_review is not None
                          else 'SUPERVISED_LAMP_APPROACH')
        if (review.get('scope') != expected_scope
                or review.get('operator_confirmed_lamp_clearance') is not True
                or review.get('operator_present') is not True
                or (lamp_task_review is not None and
                    review.get('successful_absolute_replay_verified') is not True)
                or not review.get('review_id')
                or policy_id not in review.get('allowed_policy_ids', [])
                or not start <= now < end or (end-start).total_seconds() > 3600):
            raise RuntimeError('Lamp approach clearance review is absent, mismatched or expired')
        return review
    path = CONTACT_REVIEW_MARKER if marker is None else Path(marker)
    # lstat distinguishes a missing marker from unreadable storage. Access
    # errors must fail closed rather than look like an absent hold.
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    raise RuntimeError(
        f'MOTION HELD: operator-reported contact requires lab review ({path}). '
        'Inspect hardware, model the actual work surface, and validate fingertip '
        'clearance before another learned motion trial. No CLI bypass is provided.'
    )
