"""CPU contracts for the installed production waypoint interpolation baseline."""
from types import SimpleNamespace
from unittest.mock import patch
import json

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla_spline.smolvla_interp import SmolVLAInterpPolicy


def main():
    torch.manual_seed(7)
    raw = torch.randn(1, 50, 7) * .02
    raw[..., 6] = torch.where(torch.arange(50) < 25, -1., 1.)
    mu = torch.tensor([.2, -.1, .3, -.2, .1, .15, .1])
    sigma = torch.tensor([.4, .5, .2, .8, .3, .7, .9])
    normalized = (raw - mu) / sigma
    policy = SmolVLAInterpPolicy.__new__(SmolVLAInterpPolicy)
    torch.nn.Module.__init__(policy)
    policy._act_mu, policy._act_sigma = mu, sigma
    errors = {}
    with patch.object(SmolVLAPolicy, "_get_action_chunk", return_value=normalized):
        for h in [25, 30, 50, 100]:
            policy.config = SimpleNamespace(exec_horizon=h, feasibility_stretch=False)
            out = policy._get_action_chunk({})
            decoded = out * sigma + mu
            assert out.shape == (1, h, 7)
            assert torch.isfinite(out).all()
            err = (decoded[..., :6].sum(1) - raw[..., :6].sum(1)).abs().max().item()
            assert err < 2e-6, (h, err)
            assert (decoded[:, :h // 2, 6] < 0).all()
            assert (decoded[:, h // 2 + 1:, 6] > 0).all()
            if h == 50:
                assert torch.equal(out, normalized), "native identity must be exact"
            errors[str(h)] = err
    print(json.dumps({"passed": True, "endpoint_max_abs_error": errors,
                      "native_identity_exact": True, "nonzero_normalization_mean": True}))


if __name__ == "__main__":
    main()
