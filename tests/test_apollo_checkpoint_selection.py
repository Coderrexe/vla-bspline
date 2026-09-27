import numpy as np
import pytest

from select_apollo_lamp_checkpoint import score_predictions


def test_perfect_is_zero_and_hold_is_one():
    rng = np.random.default_rng(6)
    target = rng.normal(0, .01, (10, 8, 16))
    target[:, :, 6] = .4
    opening = np.ones(10)*.8
    perfect = np.repeat(target[:, None], 3, axis=1)
    assert score_predictions(perfect, target, opening)['score'] == 0
    hold = np.zeros_like(perfect); hold[:, :, :, 6] = .8
    assert score_predictions(hold, target, opening)['score'] == pytest.approx(1)


def test_invalid_or_zero_reference_refused():
    with pytest.raises(ValueError, match='Degenerate'):
        score_predictions(np.zeros((2, 3, 8, 16)), np.zeros((2, 8, 16)), np.zeros(2))
    with pytest.raises(ValueError, match='Invalid'):
        score_predictions(np.zeros((2, 2, 8, 16)), np.zeros((2, 8, 16)), np.zeros(2))
