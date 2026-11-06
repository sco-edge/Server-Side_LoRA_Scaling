import numpy as np
import pytest


def _approx_loss(z1, zdot, y, s):
    Z = z1 + (s - 1.0) * zdot
    m = Z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(Z - m).sum(axis=1))
    return float((lse - Z[np.arange(len(y)), y]).mean())


def _problem(seed, n=200, k=2):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    z1 = rng.normal(0.0, 2.0, (n, k))
    z1[np.arange(n), y] += 0.5
    zdot = rng.normal(0.0, 0.5, (n, k))
    zdot[np.arange(n), y] += 0.3
    return z1, zdot, y


def _grid_minimizer(z1, zdot, y, hi=20.0, n=20001):
    grid = np.linspace(1e-3, hi, n)
    losses = [_approx_loss(z1, zdot, y, s) for s in grid]
    return grid[int(np.argmin(losses))]


def test_matches_grid_minimizer_binary(sls):
    z1, zdot, y = _problem(0)
    assert sls.scaling_factor(z1, zdot, y) == pytest.approx(_grid_minimizer(z1, zdot, y), abs=2e-3)


@pytest.mark.parametrize("k", [4, 5, 6])
def test_matches_grid_minimizer_multiclass(sls, k):
    z1, zdot, y = _problem(1, k=k)
    assert sls.scaling_factor(z1, zdot, y) == pytest.approx(_grid_minimizer(z1, zdot, y), abs=2e-3)
