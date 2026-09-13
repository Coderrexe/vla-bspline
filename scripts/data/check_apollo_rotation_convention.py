"""Compare recorded rotational commands with measured orientation increments.

Read-only diagnostic: it does not modify targets or configure a robot. A strong
match is evidence for a convention, not a substitute for checking Apollo's
command application code with the mentor before physical execution.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exclude-episode', action='append', default=[])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    conventions = ['base_frame_rotvec', 'body_frame_rotvec', 'extrinsic_xyz_euler_increment',
                   'intrinsic_xyz_euler_increment', 'absolute_rotvec_increment']
    grouped = {(lag, mode): [[], []] for lag in [0, 1, 2] for mode in conventions}
    count = 0
    for path in sorted((args.source/'episodes').glob('*/frames.parquet')):
        if path.parent.name in args.exclude_episode:
            continue
        table = pq.read_table(path, columns=['action', 'observation.state', 'wallclock_ns'])
        command = np.asarray(table['action'].to_pylist())[:, 3:6]
        quaternion = np.asarray(table['observation.state'].to_pylist())[:, 12:16]
        gap = np.diff(np.asarray(table['wallclock_ns'].to_pylist(), dtype=np.int64))/1e9
        rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])
        euler = rotation.as_euler('xyz')
        intrinsic_euler = rotation.as_euler('XYZ')
        differences = {
            'base_frame_rotvec': (rotation[1:]*rotation[:-1].inv()).as_rotvec(),
            'body_frame_rotvec': (rotation[:-1].inv()*rotation[1:]).as_rotvec(),
            'extrinsic_xyz_euler_increment': (np.diff(euler, axis=0)+np.pi)%(2*np.pi)-np.pi,
            'intrinsic_xyz_euler_increment': (np.diff(intrinsic_euler, axis=0)+np.pi)%(2*np.pi)-np.pi,
            'absolute_rotvec_increment': np.diff(rotation.as_rotvec(), axis=0),
        }
        for lag in [0, 1, 2]:
            ix = np.arange(len(command)-1-lag)
            good = np.linalg.norm(command[ix], axis=1) > 1e-4
            for shift in range(lag+1):
                good &= (gap[ix+shift] >= .02) & (gap[ix+shift] <= .08)
            for mode, delta in differences.items():
                grouped[lag, mode][0].append(command[ix[good]])
                grouped[lag, mode][1].append(delta[(ix+lag)[good]])
        count += 1
    rows = []
    for (lag, mode), (xs, ys) in grouped.items():
        x, y = np.concatenate(xs), np.concatenate(ys)
        xx, yy, xy = (x*x).sum(), (y*y).sum(), (x*y).sum()
        rows.append({'state_increment_lag_frames': lag, 'convention': mode, 'samples': len(x),
                     'uncentered_cosine': float(xy/np.sqrt(max(xx*yy, 1e-20))),
                     'fitted_state_increment_per_command_scale': float(xy/max(xx, 1e-20)),
                     'unscaled_component_rmse_rad': float(np.sqrt(np.mean((y-x)**2))),
                     'axis_cosines': ((x*y).sum(0)/np.sqrt(np.maximum((x*x).sum(0)*(y*y).sum(0), 1e-20))).tolist()})
    report = {'source': str(args.source), 'episodes': count, 'excluded_episodes': args.exclude_episode,
              'quaternion_order_recorded': 'wxyz', 'frame_gap_range_s': [.02, .08],
              'minimum_command_rotation_norm_rad': 1e-4, 'comparisons': rows,
              'interpretation': 'Empirical convention check only; preserve targets and confirm runtime semantics before execution'}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
