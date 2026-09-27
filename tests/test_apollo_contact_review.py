"""Offline tests of our launcher hold; no robot/runtime transport imports."""
import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from apollo_contact_review import require_contact_review_clear


def test_absent_marker(tmp_path):
    require_contact_review_clear(tmp_path/'absent.json')


@pytest.mark.parametrize('contents', ['', '{}', 'invalid JSON', '{"status":"clear"}'])
def test_any_present_marker_blocks(tmp_path, contents):
    marker = tmp_path/'hold.json'
    marker.write_text(contents)
    with pytest.raises(RuntimeError, match='MOTION HELD'):
        require_contact_review_clear(marker)


def test_dangling_symlink_blocks(tmp_path):
    marker = tmp_path/'hold.json'
    marker.symlink_to(tmp_path/'missing-target')
    with pytest.raises(RuntimeError, match='MOTION HELD'):
        require_contact_review_clear(marker)


def test_unreadable_storage_fails_closed(tmp_path):
    with patch('apollo_contact_review.os.lstat', side_effect=PermissionError):
        with pytest.raises(PermissionError):
            require_contact_review_clear(tmp_path/'hold.json')


def lamp_review(tmp_path):
    import hashlib
    import json
    from datetime import datetime, timezone, timedelta
    marker = tmp_path/'hold.json'
    marker.write_text('{"status":"CONTACT_REVIEW_REQUIRED"}')
    now = datetime.now(timezone.utc)
    review = {'scope': 'SUPERVISED_LAMP_APPROACH', 'review_id': 'lamp-test',
              'operator_confirmed_lamp_clearance': True, 'operator_present': True,
              'confirmed_at_utc': (now-timedelta(seconds=10)).isoformat(),
              'valid_until_utc': (now+timedelta(minutes=10)).isoformat(),
              'allowed_policy_ids': ['lamp'],
              'contact_marker_sha256': hashlib.sha256(marker.read_bytes()).hexdigest()}
    path = tmp_path/'review.json'
    path.write_text(json.dumps(review))
    return marker, path, review


def test_lamp_review_does_not_release_general_hold(tmp_path):
    marker, path, review = lamp_review(tmp_path)
    assert require_contact_review_clear(marker, lamp_approach_review=path, policy_id='lamp') == review
    with pytest.raises(RuntimeError, match='MOTION HELD'):
        require_contact_review_clear(marker)
    with pytest.raises(RuntimeError, match='mismatched'):
        require_contact_review_clear(marker, lamp_approach_review=path, policy_id='drawer')


@pytest.mark.parametrize('field,value', [
    ('scope', 'FULL_TASK'), ('operator_present', False),
    ('operator_confirmed_lamp_clearance', False), ('review_id', ''),
    ('valid_until_utc', '2020-01-01T00:00:00+00:00'),
    ('valid_until_utc', '2099-01-01T00:00:00+00:00'),
])
def test_lamp_review_rejects_wrong_scope_or_expiry(tmp_path, field, value):
    import json
    marker, path, review = lamp_review(tmp_path)
    path.write_text(json.dumps(review | {field: value}))
    with pytest.raises(RuntimeError):
        require_contact_review_clear(marker, lamp_approach_review=path, policy_id='lamp')


def test_new_contact_invalidates_lamp_review(tmp_path):
    marker, path, _ = lamp_review(tmp_path)
    marker.write_text('{"status":"NEW_CONTACT"}')
    with pytest.raises(RuntimeError, match='changed'):
        require_contact_review_clear(marker, lamp_approach_review=path, policy_id='lamp')


@pytest.mark.parametrize('script,conditional', [
    ('run_apollo_one_row_session.py', False),
    ('run_apollo_dora.py', True),
    ('run_apollo_gripper_check.py', False),
    ('check_apollo_shadow_session.py', False),
])
def test_launcher_hold_precedes_transport_and_side_effects(script, conditional):
    # Parse only the launcher's source: actual module imports could
    # import Dora/runtime transport packages, which are forbidden in this test.
    import apollo_contact_review
    root = Path(apollo_contact_review.__file__).parent
    tree = ast.parse((root/script).read_text())
    main = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == 'main')
    statements = main.body
    parse_index = next(i for i, n in enumerate(statements)
                       if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                       and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'parse_args')
    check = statements[parse_index+1]
    if conditional:
        assert isinstance(check, ast.If)
        check = check.body[0]
    assert isinstance(check, ast.Expr) and isinstance(check.value, ast.Call)
    assert check.value.func.id == 'require_contact_review_clear'
