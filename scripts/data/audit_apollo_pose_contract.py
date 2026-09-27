"""Numeric pose-convention audit using archived joints and the xArm MJCF chain.

No hardware imports/connections. Kinematics only: no motion authorization and no
physical calibration claim. Both corrected TCP and legacy flange features are
tested; mixed/unknown episodes are reported rather than silently converted.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def flange_fk(q, xml):
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != 7 or not np.isfinite(q).all():
        raise ValueError('Expected finite seven-joint rows')
    body = xml.find('.//body[@name="link_base"]')
    pos = np.zeros((len(q), 3))
    rot = np.tile(np.eye(3), (len(q), 1, 1))
    for i in range(7):
        body = body.find(f'body[@name="link{i+1}"]')
        if body is None or any(k in body.attrib for k in ('euler', 'axisangle', 'xyaxes', 'zaxis')):
            raise ValueError('Unexpected MJCF link chain')
        off = np.fromstring(body.get('pos', '0 0 0'), sep=' ')
        quat = np.fromstring(body.get('quat', '1 0 0 0'), sep=' ')
        static_rot = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        joint = body.find(f'joint[@name="joint{i+1}"]')
        if joint is None or joint.get('type', 'hinge') != 'hinge' or joint.get('pos', '0 0 0') != '0 0 0':
            raise ValueError('Unexpected joint type or pivot')
        axis = np.fromstring(joint.get('axis', '0 0 1'), sep=' ')
        pos += np.einsum('nij,j->ni', rot, off)
        rot = rot @ static_rot @ Rotation.from_rotvec(q[:, i, None]*axis).as_matrix()
    return pos, rot


def summarize(x):
    return dict(zip(('median', 'p95', 'max'), map(float, np.quantile(x, [.5, .95, 1]))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--mjcf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require', choices=['corrected_tcp', 'legacy_flange'])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # The production MJCF has '--' inside comments. MuJoCo accepts it, but
    # ElementTree rejects those comments; discard comments in memory only.
    xml = ET.fromstring(re.sub(r'<!--.*?-->', '', args.mjcf.read_text(), flags=re.S))
    default = xml.find('./default/default[@class="xarm7"]/joint')
    if default is None or default.get('axis') != '0 0 1' or xml.find('compiler').get('angle') != 'radian':
        raise ValueError('Unsupported MJCF defaults')
    records = []
    for path in sorted((args.source/'episodes').glob('*/frames.parquet')):
        state = np.asarray(pq.read_table(path, columns=['observation.state'])['observation.state'].to_pylist())
        if state.shape[1:] != (32,) or not np.isfinite(state).all():
            raise ValueError('Invalid recorded state')
        candidates = {'corrected_tcp': [], 'legacy_flange': []}
        for start, gripper in ((0, True), (16, False)):
            pos, rot = flange_fk(state[:, start:start+7], xml)
            tcp = pos + (np.einsum('nij,j->ni', rot, [0., 0., .172]) if gripper else 0.)
            tcp_rot = rot @ np.diag([-1., -1., 1.]) if gripper else rot
            legacy_rot = Rotation.from_euler('XYZ', Rotation.from_matrix(rot).as_euler('xyz')).as_matrix()
            observed_rot = Rotation.from_quat(state[:, start+np.array([13, 14, 15, 12])])
            for name, expected_pos, expected_rot in [('corrected_tcp', tcp, tcp_rot), ('legacy_flange', pos, legacy_rot)]:
                pe = np.linalg.norm(state[:, start+9:start+12]-expected_pos, axis=1)*1000
                rotation_error = (observed_rot*Rotation.from_matrix(expected_rot).inv()).magnitude()
                candidates[name].append({'arm': 'grip' if gripper else 'view',
                                         'position_error_mm': summarize(pe), 'rotation_error_rad': summarize(rotation_error)})
        passing = [name for name, arms in candidates.items() if all(
            a['position_error_mm']['p95'] < 2. and a['rotation_error_rad']['p95'] < .02 for a in arms)]
        records.append({'episode': path.parent.name, 'frames': len(state),
                        'parquet_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'classification': passing[0] if len(passing) == 1 else 'unknown',
                        'candidates': candidates})
    if not records:
        raise ValueError('No episodes')
    classifications = sorted({r['classification'] for r in records})
    report = {'purpose': 'OFFLINE_POSE_FEATURE_CONVENTION_NOT_HARDWARE_CALIBRATION',
              'source': str(args.source), 'mjcf_sha256': hashlib.sha256(args.mjcf.read_bytes()).hexdigest(),
              'frames': sum(r['frames'] for r in records), 'episodes': records,
              'classifications': classifications,
              'contract': classifications[0] if len(classifications) == 1 else 'mixed',
              'classification_tolerance': 'per-arm p95 position < 2 mm and rotation < .02 rad'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes'}, indent=2))
    if args.require and report['contract'] != args.require:
        raise ValueError(f"Expected {args.require}, found {report['contract']}")


if __name__ == '__main__':
    main()
