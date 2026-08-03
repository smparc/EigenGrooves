"""
Tests for the from-scratch linear algebra.

NumPy's ``linalg`` appears here and nowhere else in the project: it is the
ground truth we check our own implementations against.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigengrooves.linalg import (
    canonicalize_signs,
    explained_variance_ratio,
    householder_qr,
    jacobi_eigh,
    modified_gram_schmidt,
    qr_eigh,
    randomized_svd,
    svd,
)

from .conftest import ill_conditioned, well_conditioned

SHAPES = [(50, 9), (9, 9), (9, 50), (200, 7), (5, 1), (1, 5), (2, 2)]
BACKENDS = ["jacobi", "eigh"]


# ---------------------------------------------------------------------------
# QR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_householder_qr_reconstructs(rng, shape):
    A = rng.normal(size=shape)
    Q, R = householder_qr(A)
    assert np.allclose(Q @ R, A, atol=1e-10)


@pytest.mark.parametrize("shape", SHAPES)
def test_householder_q_is_orthonormal(rng, shape):
    A = rng.normal(size=shape)
    Q, _ = householder_qr(A)
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)


@pytest.mark.parametrize("shape", SHAPES)
def test_householder_r_is_upper_triangular(rng, shape):
    _, R = householder_qr(rng.normal(size=shape))
    assert np.allclose(R, np.triu(R))


def test_householder_full_mode_shapes(rng):
    A = rng.normal(size=(40, 8))
    Q, R = householder_qr(A, reduced=False)
    assert Q.shape == (40, 40)
    assert R.shape == (40, 8)
    assert np.allclose(Q @ R, A, atol=1e-10)
    assert np.allclose(Q.T @ Q, np.eye(40), atol=1e-10)


def test_householder_beats_gram_schmidt_on_ill_conditioning():
    """The stability claim in the module docstring, made checkable.

    A Hilbert matrix has condition number ~1e16. Modified Gram-Schmidt loses
    orthogonality proportionally to that; Householder does not.
    """
    n = 12
    H = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])

    q_house, _ = householder_qr(H)
    q_mgs, _ = modified_gram_schmidt(H)

    err_house = np.linalg.norm(q_house.T @ q_house - np.eye(n))
    err_mgs = np.linalg.norm(q_mgs.T @ q_mgs - np.eye(n))

    assert err_house < 1e-12
    assert err_mgs > 1e-3
    assert err_house < err_mgs


def test_gram_schmidt_handles_rank_deficiency(rng):
    A = rng.normal(size=(20, 4))
    A[:, 3] = A[:, 1]  # exact duplicate column
    Q, R = modified_gram_schmidt(A)
    assert np.all(np.isfinite(Q))
    assert np.allclose(Q @ R, A, atol=1e-10)


# ---------------------------------------------------------------------------
# Symmetric eigensolvers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", [jacobi_eigh, qr_eigh])
@pytest.mark.parametrize("n", [1, 2, 5, 9, 20])
def test_eigensolvers_match_numpy(rng, solver, n):
    M = rng.normal(size=(n, n))
    S = M @ M.T
    values, vectors = solver(S)
    expected = np.sort(np.linalg.eigvalsh(S))[::-1]

    assert np.allclose(values, expected, atol=1e-8 * max(1.0, abs(expected[0])))
    assert np.allclose(vectors.T @ vectors, np.eye(n), atol=1e-9)
    assert np.allclose(vectors @ np.diag(values) @ vectors.T, S, atol=1e-8 * max(1.0, np.linalg.norm(S)))


@pytest.mark.parametrize("solver", [jacobi_eigh, qr_eigh])
def test_eigenvalues_are_descending(rng, solver):
    M = rng.normal(size=(12, 12))
    values, _ = solver(M @ M.T)
    assert np.all(np.diff(values) <= 1e-9)


@pytest.mark.parametrize("solver", [jacobi_eigh, qr_eigh])
def test_eigensolvers_handle_repeated_eigenvalues(solver, rng):
    """Clustered spectra are where unshifted QR iteration stalls."""
    Q, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    spectrum = np.array([5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    S = Q @ np.diag(spectrum) @ Q.T

    values, vectors = solver(S)
    assert np.allclose(values, spectrum, atol=1e-8)
    assert np.allclose(vectors.T @ vectors, np.eye(10), atol=1e-9)


@pytest.mark.parametrize("solver", [jacobi_eigh, qr_eigh])
def test_eigensolvers_reject_asymmetric(solver, rng):
    with pytest.raises(ValueError, match="not symmetric"):
        solver(rng.normal(size=(5, 5)))


@pytest.mark.parametrize("solver", [jacobi_eigh, qr_eigh])
def test_eigensolvers_handle_zero_matrix(solver):
    values, vectors = solver(np.zeros((4, 4)))
    assert np.allclose(values, 0.0)
    assert np.allclose(vectors, np.eye(4))


def test_convergence_tolerance_is_scale_invariant(rng):
    """Regression: v1's absolute tolerance never fired on real-sized data.

    The original convergence test compared an off-diagonal *sum* against a
    fixed 1e-10. That sum grows with ``||A||``, so at 5000 rows the loop always
    exhausted its iteration budget and still exited unconverged. Scaling the
    input must not change the quality of the answer.
    """
    M = rng.normal(size=(9, 9))
    S = M @ M.T

    for scale in (1e-3, 1.0, 1e3, 1e6):
        values, vectors = jacobi_eigh(S * scale)
        expected = np.sort(np.linalg.eigvalsh(S * scale))[::-1]
        relative_error = np.max(np.abs(values - expected)) / max(abs(expected[0]), 1e-30)
        assert relative_error < 1e-12, f"failed at scale {scale}"
        assert np.allclose(vectors.T @ vectors, np.eye(9), atol=1e-10)


# ---------------------------------------------------------------------------
# SVD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_svd_reconstructs(rng, backend, shape):
    A = rng.normal(size=shape)
    U, sigma, Vt = svd(A, backend=backend)
    assert np.allclose(U @ np.diag(sigma) @ Vt, A, atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_svd_matches_numpy_singular_values(rng, backend, shape):
    A = rng.normal(size=shape)
    _, sigma, _ = svd(A, backend=backend)
    expected = np.linalg.svd(A, compute_uv=False)
    assert np.allclose(sigma, expected[: len(sigma)], atol=1e-9)


@pytest.mark.parametrize("shape", SHAPES)
def test_jacobi_factors_are_orthonormal(rng, shape):
    A = rng.normal(size=shape)
    U, sigma, Vt = svd(A, backend="jacobi")
    r = len(sigma)
    assert np.allclose(U.T @ U, np.eye(r), atol=1e-10)
    assert np.allclose(Vt @ Vt.T, np.eye(r), atol=1e-10)


@pytest.mark.parametrize("shape", SHAPES)
def test_eigh_factors_are_orthonormal_within_its_weaker_guarantee(rng, shape):
    """The eigh backend is held to a looser bound, on purpose.

    Its left singular vectors are recovered as ``u_i = A v_i / sigma_i``, so
    their accuracy degrades with ``sigma_min``. Because ``A^T A`` squares the
    condition number, that degradation is real and unavoidable: measured over
    300 random 9x9 matrices, worst-case orthogonality error is 2.2e-05 for
    eigh against 4.4e-15 for Jacobi.

    Asserting 1e-10 here would just be flaky. Asserting a weaker bound, and
    saying why, records the limitation instead of hiding it -- and it is the
    reason Jacobi is the default.
    """
    A = rng.normal(size=shape)
    U, sigma, Vt = svd(A, backend="eigh")
    r = len(sigma)
    assert np.allclose(U.T @ U, np.eye(r), atol=1e-3)
    # V comes straight from a symmetric eigensolver, so it stays accurate.
    assert np.allclose(Vt @ Vt.T, np.eye(r), atol=1e-10)


def test_jacobi_left_factor_is_orders_of_magnitude_more_orthonormal(rng):
    """Quantifies the gap the previous test tolerates."""
    worst = {"jacobi": 0.0, "eigh": 0.0}
    for seed in range(60):
        A = np.random.default_rng(seed).normal(size=(9, 9))
        for backend in ("jacobi", "eigh"):
            U, sigma, _ = svd(A, backend=backend)
            error = np.linalg.norm(U.T @ U - np.eye(len(sigma)))
            worst[backend] = max(worst[backend], error)

    assert worst["jacobi"] < 1e-13
    assert worst["eigh"] > worst["jacobi"] * 100


@pytest.mark.parametrize("backend", BACKENDS)
def test_singular_values_are_descending_and_nonnegative(rng, backend):
    _, sigma, _ = svd(well_conditioned(rng), backend=backend)
    assert np.all(sigma >= 0)
    assert np.all(np.diff(sigma) <= 1e-12)


@pytest.mark.parametrize("k", [1, 3, 7, 12])
def test_eckart_young_optimality(rng, k):
    """Rank-k truncation must be the best rank-k approximation.

    The error of the optimal approximation is exactly the root-sum-square of
    the discarded singular values. If our truncation matches that bound, the
    decomposition is correct in the sense that actually matters.
    """
    A = rng.normal(size=(120, 15))
    U, sigma, Vt = svd(A, k=k)
    error = np.linalg.norm(A - U @ np.diag(sigma) @ Vt)
    expected = np.sqrt(np.sum(np.linalg.svd(A, compute_uv=False)[k:] ** 2))
    assert error == pytest.approx(expected, abs=1e-9)


def test_jacobi_preserves_small_singular_values(rng):
    """The core justification for the one-sided Jacobi backend.

    On a matrix whose columns are near-collinear, going through ``A^T A``
    squares the condition number and destroys the trailing singular values.
    Jacobi never forms that product, and recovers them to full relative
    accuracy. This test asserts the gap rather than trusting the docstring.
    """
    A = ill_conditioned(rng)
    expected = np.linalg.svd(A, compute_uv=False)

    _, sigma_jacobi, _ = svd(A, backend="jacobi")
    _, sigma_eigh, _ = svd(A, backend="eigh")

    assert len(sigma_jacobi) == len(expected)

    relative_jacobi = np.max(np.abs(sigma_jacobi - expected) / expected)
    n = min(len(sigma_eigh), len(expected))
    relative_eigh = np.max(np.abs(sigma_eigh[:n] - expected[:n]) / expected[:n])

    assert relative_jacobi < 1e-10
    # The eigh path is expected to be materially worse on this input; if it
    # ever stops being worse, this test should be revisited rather than deleted.
    assert relative_eigh > relative_jacobi * 100


def test_reconstruction_error_hides_the_accuracy_gap(rng):
    """Both backends reconstruct well; only one gets the small values right.

    Worth pinning down, because reconstruction error is the obvious metric to
    reach for and it is *not* sensitive to the failure this project cares
    about. Frobenius error is dominated by the largest singular values, so a
    backend can mangle every small component and still reconstruct to 1e-16.
    Relative accuracy per singular value is the metric that discriminates --
    see ``test_jacobi_preserves_small_singular_values``.
    """
    A = ill_conditioned(rng)
    scale = np.linalg.norm(A)

    def relative_error(backend: str) -> float:
        U, sigma, Vt = svd(A, backend=backend)
        return float(np.linalg.norm(A - U @ np.diag(sigma) @ Vt) / scale)

    assert relative_error("jacobi") < 1e-13
    assert relative_error("eigh") < 1e-13


def test_truncation_clamps_to_available_rank(rng):
    _, sigma, _ = svd(rng.normal(size=(10, 3)), k=99)
    assert len(sigma) == 3


@pytest.mark.parametrize(
    "matrix",
    [np.zeros((10, 4)), np.ones((10, 1)), np.eye(5), np.zeros((0, 5))],
    ids=["all-zero", "single-column", "identity", "no-rows"],
)
def test_svd_survives_degenerate_input(matrix):
    U, sigma, Vt = svd(matrix)
    assert np.all(np.isfinite(U))
    assert np.all(np.isfinite(sigma))
    assert np.all(np.isfinite(Vt))


@pytest.mark.parametrize("bad", [np.array([[1.0, np.nan]]), np.array([[1.0, np.inf]])])
def test_svd_rejects_non_finite_input(bad):
    with pytest.raises(ValueError, match="NaN or inf"):
        svd(bad)


def test_svd_rejects_unknown_backend(rng):
    with pytest.raises(ValueError, match="unknown backend"):
        svd(rng.normal(size=(5, 3)), backend="magic")


def test_randomized_backend_requires_k(rng):
    with pytest.raises(ValueError, match="requires an explicit k"):
        svd(rng.normal(size=(50, 9)), backend="randomized")


# ---------------------------------------------------------------------------
# Randomized SVD
# ---------------------------------------------------------------------------


def test_randomized_svd_approximates_low_rank(rng):
    latent = rng.normal(size=(1500, 6))
    mixing = rng.normal(size=(6, 40))
    A = latent @ mixing + 0.01 * rng.normal(size=(1500, 40))

    _, sigma, _ = randomized_svd(A, k=6, random_state=42)
    expected = np.linalg.svd(A, compute_uv=False)[:6]
    assert np.allclose(sigma, expected, rtol=1e-5)


def test_randomized_svd_is_deterministic_given_a_seed(rng):
    A = rng.normal(size=(300, 20))
    first = randomized_svd(A, k=5, random_state=7)[1]
    second = randomized_svd(A, k=5, random_state=7)[1]
    assert np.array_equal(first, second)


def test_randomized_svd_left_factor_is_orthonormal(rng):
    A = rng.normal(size=(400, 25))
    U, _, _ = randomized_svd(A, k=8, random_state=3)
    assert np.allclose(U.T @ U, np.eye(8), atol=1e-9)


def test_randomized_svd_rejects_bad_k(rng):
    with pytest.raises(ValueError, match="k must be"):
        randomized_svd(rng.normal(size=(20, 5)), k=0)


# ---------------------------------------------------------------------------
# Cross-backend agreement and sign convention
# ---------------------------------------------------------------------------


def test_backends_agree(rng):
    A = well_conditioned(rng, m=300)
    reference = svd(A, k=5, backend="jacobi")
    for backend in ("eigh", "randomized"):
        _, sigma, Vt = svd(A, k=5, backend=backend, random_state=1)
        assert np.allclose(sigma, reference[1], rtol=1e-6)
        assert np.allclose(np.abs(Vt), np.abs(reference[2]), atol=1e-6)


def test_signs_are_stable_across_row_permutations(rng):
    A = well_conditioned(rng, m=300)
    _, _, first = svd(A, k=5)
    _, _, second = svd(A[rng.permutation(A.shape[0])], k=5)
    assert np.allclose(first, second, atol=1e-8)


def test_signs_are_stable_across_backends(rng):
    A = well_conditioned(rng, m=300)
    _, _, jacobi = svd(A, k=5, backend="jacobi")
    _, _, eigh = svd(A, k=5, backend="eigh")
    assert np.allclose(jacobi, eigh, atol=1e-6)


def test_canonical_sign_convention_makes_dominant_loading_positive(rng):
    _, _, Vt = svd(well_conditioned(rng, m=300), k=5)
    for row in Vt:
        assert row[np.argmax(np.abs(row))] > 0


def test_canonicalize_signs_preserves_the_product(rng):
    A = well_conditioned(rng)
    U, sigma, Vt = svd(A, k=4, canonical_signs=False)
    U2, Vt2 = canonicalize_signs(U, Vt)
    assert np.allclose(U @ np.diag(sigma) @ Vt, U2 @ np.diag(sigma) @ Vt2, atol=1e-12)


# ---------------------------------------------------------------------------
# Explained variance
# ---------------------------------------------------------------------------


def test_explained_variance_needs_the_full_spectrum(rng):
    """Regression for the '100% variance explained' bug.

    v1 normalised the truncated spectrum against itself, so every run reported
    100% regardless of k. Passing the full spectrum is what makes the number
    mean something.
    """
    A = well_conditioned(rng, m=1000, n=9)
    _, full, _ = svd(A)
    _, truncated, _ = svd(A, k=5)

    # The documented footgun: self-normalised, it always sums to 1.
    assert explained_variance_ratio(truncated).sum() == pytest.approx(1.0)

    # Against the real denominator, it is a genuine fraction below 1.
    honest = explained_variance_ratio(truncated, full).sum()
    expected = np.sum(truncated**2) / np.sum(full**2)
    assert honest == pytest.approx(expected)
    assert honest < 0.999


def test_explained_variance_of_full_spectrum_sums_to_one(rng):
    _, sigma, _ = svd(well_conditioned(rng))
    assert explained_variance_ratio(sigma, sigma).sum() == pytest.approx(1.0)


def test_explained_variance_handles_zero_spectrum():
    assert np.all(explained_variance_ratio(np.zeros(5)) == 0.0)
