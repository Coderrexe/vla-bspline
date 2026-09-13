"""Compare legacy training recordings with the lab's TCP-backfilled copies.

Reads numeric data only. Does not change either recording, any model, or any
robot state. Residuals include SDK/joint sample asynchrony, not just float error.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from apollo_legacy_state import current_tcp_to_training_state, STATE_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-root', type=Path, required=True)
    p.add_argument('--corrected-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {'comparison':'old SDK features vs adapter(corrected TCP features)',
              'excluded_from_training':['20260911T005454.988Z-2ad672'], 'tasks':{}}
    for task in ['cabinet_assembling','drawer_assembling']:
        rows, episodes, anomalies = [], [], []
        for original in sorted((args.original_root/task/'episodes').glob('*/frames.parquet')):
            episode = original.parent.name
            if episode in report['excluded_from_training']:
                continue
            corrected = args.corrected_root/task/episode/'frames.parquet'
            a,b = [pq.read_table(path,columns=['observation.state','action'])
                   for path in [original,corrected]]
            old,new = [np.asarray(t['observation.state'].to_pylist(),dtype=np.float32) for t in [a,b]]
            if old.shape != new.shape or old.shape[1] != 32:
                raise ValueError(f'Incompatible state arrays: {episode}')
            if not np.array_equal(np.asarray(a['action'].to_pylist()),
                                  np.asarray(b['action'].to_pylist()),equal_nan=True):
                raise ValueError(f'Backfill changed action targets: {episode}')
            reconstructed = current_tcp_to_training_state(new)
            errors = np.abs(old-reconstructed)
            maxima = errors.max(1)
            rows.append(maxima)
            episodes.append(episode)
            for frame in np.flatnonzero(maxima > 1e-3):
                field = int(np.argmax(errors[frame]))
                anomalies.append({'episode':episode,'frame':int(frame),'field':STATE_NAMES[field],
                                  'original':float(old[frame,field]),
                                  'reconstructed':float(reconstructed[frame,field]),
                                  'absolute_difference':float(maxima[frame])})
        if not rows:
            raise ValueError(f'No comparisons for {task}')
        maxima = np.concatenate(rows)
        report['tasks'][task] = {'episodes':len(episodes),'frames':len(maxima),
            'action_targets_identical':True,
            'median_max_field_difference':float(np.median(maxima)),
            'p99_max_field_difference':float(np.quantile(maxima,.99)),
            'rows_above_1e_4':int((maxima>1e-4).sum()),
            'rows_above_1e_3':int((maxima>1e-3).sum()),
            'rows_above_1e_2':int((maxima>1e-2).sum()),
            'anomalies':sorted(anomalies,key=lambda r:-r['absolute_difference'])}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{x:v for x,v in d.items() if x != 'anomalies'}
                      for k,d in report['tasks'].items()},indent=2))


if __name__ == '__main__':
    main()
