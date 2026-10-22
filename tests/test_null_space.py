"""Synthetic regressions for the preserved activation row space."""

import pytest
import torch

from src.null_space import (
    apply_null_space_constrained_projection,
    apply_null_space_projection,
    compute_null_space_projector_svd,
)


def _affine_activations():
    activations = torch.zeros(20, 8)
    activations[:, 0] = 3.0
    activations[::2, 1] = 2.0
    activations[1::2, 1] = -2.0
    return activations


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_repeated_nonzero_activations_retain_their_mean_direction(seed):
    torch.manual_seed(seed)
    activations = torch.zeros(16, 8)
    activations[:, 0] = 4.0

    basis, singular_values = compute_null_space_projector_svd(activations, rank_ratio=0.95, min_null_dim=1)

    assert basis.shape == (8, 1)
    assert singular_values.shape == (1,)
    projector = apply_null_space_projection(torch.eye(8), basis)
    torch.testing.assert_close(projector @ activations.T, torch.zeros(8, 16), atol=1e-5, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_projector_protects_both_the_mean_and_variation(seed):
    torch.manual_seed(seed)
    activations = _affine_activations()

    basis, _ = compute_null_space_projector_svd(activations, rank_ratio=0.95, min_null_dim=1)

    assert basis.shape == (8, 2)
    projector = apply_null_space_projection(torch.eye(8), basis)
    torch.testing.assert_close(projector @ activations.T, torch.zeros(8, 20), atol=1e-5, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_constrained_update_preserves_outputs_for_original_activations(seed):
    torch.manual_seed(seed)
    activations = _affine_activations()
    weight = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 10
    direction = torch.ones(8)
    basis, _ = compute_null_space_projector_svd(activations, rank_ratio=0.95, min_null_dim=1)

    updated = apply_null_space_constrained_projection(weight, direction, basis, preserve_norm=False)

    torch.testing.assert_close(updated @ activations.T, weight @ activations.T, atol=1e-5, rtol=0)
    assert not torch.allclose(updated, weight)


def test_zero_mean_activations_still_have_an_orthonormal_row_basis():
    torch.manual_seed(0)
    activations = _affine_activations()
    activations[:, 0] = 0.0

    basis, _ = compute_null_space_projector_svd(activations, rank_ratio=0.95, min_null_dim=1)

    assert basis.shape == (8, 1)
    torch.testing.assert_close(basis.T @ basis, torch.eye(1), atol=1e-5, rtol=0)
    projector = apply_null_space_projection(torch.eye(8), basis)
    torch.testing.assert_close(projector @ activations.T, torch.zeros(8, 20), atol=1e-5, rtol=0)


@pytest.mark.parametrize("fallback", ["cpu", "eigh", "retry"])
def test_fallback_decompositions_also_preserve_the_activation_mean(monkeypatch, fallback):
    torch.manual_seed(0)
    original_svd = torch.svd_lowrank
    failures = 1 if fallback == "cpu" else 2
    calls = 0

    def fail_initial_svd_calls(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= failures:
            raise RuntimeError("injected SVD failure")
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(torch, "svd_lowrank", fail_initial_svd_calls)
    if fallback == "retry":

        def fail_eigh(*args, **kwargs):
            raise RuntimeError("injected eigendecomposition failure")

        monkeypatch.setattr(torch.linalg, "eigh", fail_eigh)
    activations = _affine_activations()

    basis, _ = compute_null_space_projector_svd(activations, rank_ratio=0.95, min_null_dim=1)

    assert calls == (2 if fallback in {"cpu", "eigh"} else 3)
    assert basis.shape == (8, 2)
    projector = apply_null_space_projection(torch.eye(8), basis)
    torch.testing.assert_close(projector @ activations.T, torch.zeros(8, 20), atol=1e-5, rtol=0)
