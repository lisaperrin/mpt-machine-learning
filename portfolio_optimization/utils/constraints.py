from typing import Dict, Iterable, Tuple

import numpy as np


class ConstraintError(ValueError):
    """Raised when portfolio constraints are mathematically infeasible."""


def validate_weight_bounds(
    n_assets: int,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> Tuple[float, float]:
    if n_assets <= 0:
        raise ConstraintError("Need at least one asset")
    if min_weight < 0:
        raise ConstraintError("min_weight must be non-negative")
    if max_weight <= 0:
        raise ConstraintError("max_weight must be positive")
    if min_weight > max_weight:
        raise ConstraintError("min_weight cannot be greater than max_weight")
    if n_assets * min_weight > 1.0 + 1e-12:
        raise ConstraintError(
            f"min_weight={min_weight:.4f} is infeasible for {n_assets} assets "
            f"(minimum total weight would be {n_assets * min_weight:.4f})"
        )
    if n_assets * max_weight < 1.0 - 1e-12:
        raise ConstraintError(
            f"max_weight={max_weight:.4f} is infeasible for {n_assets} assets "
            f"(maximum total weight would be {n_assets * max_weight:.4f})"
        )
    return float(min_weight), float(max_weight)


def bounds_from_constraints(
    n_assets: int,
    constraints: Dict | None,
    default_min: float = 0.0,
    default_max: float = 1.0,
) -> Tuple[float, float]:
    constraints = constraints or {}
    min_weight = constraints.get("min_weight", default_min)
    max_weight = constraints.get("max_weight", default_max)
    return validate_weight_bounds(n_assets, min_weight, max_weight)


def project_weights_to_bounds(
    weights: Iterable[float],
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    tolerance: float = 1e-10,
) -> np.ndarray:
    weights_array = np.asarray(list(weights), dtype=float)
    n_assets = len(weights_array)
    min_weight, max_weight = validate_weight_bounds(n_assets, min_weight, max_weight)

    if not np.all(np.isfinite(weights_array)) or weights_array.sum() <= 0:
        weights_array = np.ones(n_assets) / n_assets
    else:
        weights_array = np.maximum(weights_array, 0)
        weights_array = weights_array / weights_array.sum()

    projected = np.clip(weights_array, min_weight, max_weight)

    for _ in range(n_assets + 1):
        residual = 1.0 - projected.sum()
        if abs(residual) <= tolerance:
            break

        if residual > 0:
            eligible = projected < max_weight - tolerance
            capacity = max_weight - projected[eligible]
            total_capacity = capacity.sum()
            if total_capacity <= tolerance:
                raise ConstraintError("Cannot add residual weight within max_weight constraints")
            projected[eligible] += capacity / total_capacity * residual
        else:
            eligible = projected > min_weight + tolerance
            removable = projected[eligible] - min_weight
            total_removable = removable.sum()
            if total_removable <= tolerance:
                raise ConstraintError("Cannot remove residual weight within min_weight constraints")
            projected[eligible] -= removable / total_removable * (-residual)

        projected = np.clip(projected, min_weight, max_weight)

    residual = 1.0 - projected.sum()
    if abs(residual) > 1e-8:
        raise ConstraintError("Could not project weights to feasible bounds")

    projected += residual / n_assets
    return np.clip(projected, min_weight, max_weight)
