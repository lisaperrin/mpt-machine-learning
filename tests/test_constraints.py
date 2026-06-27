import numpy as np
import pytest

from portfolio_optimization.utils.constraints import (
    ConstraintError,
    project_weights_to_bounds,
    validate_weight_bounds,
)


def test_validate_weight_bounds_accepts_feasible_bounds():
    min_weight, max_weight = validate_weight_bounds(5, 0.05, 0.40)
    assert min_weight == 0.05
    assert max_weight == 0.40


def test_validate_weight_bounds_rejects_infeasible_min_weight():
    with pytest.raises(ConstraintError):
        validate_weight_bounds(3, 0.40, 0.60)


def test_validate_weight_bounds_rejects_infeasible_max_weight():
    with pytest.raises(ConstraintError):
        validate_weight_bounds(3, 0.01, 0.30)


def test_project_weights_preserves_sum_and_bounds():
    projected = project_weights_to_bounds([0.90, 0.05, 0.03, 0.02], 0.10, 0.50)

    assert abs(projected.sum() - 1.0) < 1e-8
    assert np.all(projected >= 0.10 - 1e-8)
    assert np.all(projected <= 0.50 + 1e-8)
