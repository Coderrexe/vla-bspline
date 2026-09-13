"""Offline numerical diagnostics on a single saved fixture; no robot/Dora API."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from apollo_predictor import ApolloPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    model = ApolloPredictor(args.checkpoint, device='cuda:0')
    with np.load(args.checkpoint/'prediction_fixture.npz') as f:
        state, view, grip, expected = [f[k].copy() for k in
                                      ['state','view_wrist','grip_wrist','expected_action']]
        seed = int(f['seed'])
    report = {'torch':torch.__version__, 'cuda':torch.version.cuda,
              'gpu':torch.cuda.get_device_name(),
              'parameter_dtypes':dict(Counter(str(p.dtype) for p in model.policy.parameters())),
              'defaults':{'matmul_precision':torch.get_float32_matmul_precision(),
                          'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                          'cudnn_tf32':torch.backends.cudnn.allow_tf32}, 'variants':[]}
    outputs = {'expected':expected}
    for name, tf32, cudnn in [('default',None,None),('strict_fp32',False,False),
                              ('tf32_matmul',True,False),('strict_repeat',False,False)]:
        if tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = cudnn
        model.reset()
        torch.manual_seed(seed)
        actual = model.predict_chunk(state,view,grip)
        outputs[name] = actual
        report['variants'].append({'name':name,
            'reference_matches_original_tolerance':bool(np.allclose(actual,expected,atol=1e-5,rtol=.005)),
            'max_reference_difference_by_channel':np.abs(actual-expected).max(0).tolist(),
            'max_difference_from_default':float(np.abs(actual-outputs['default']).max())})
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),**outputs)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
