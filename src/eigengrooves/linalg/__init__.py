"""From-scratch linear algebra: QR, symmetric eigensolvers, and SVD.

Everything in this package is implemented directly from the underlying
mathematics. NumPy is used for array storage and elementwise/BLAS-level
operations (``@``, ``norm``, slicing) but never for the decompositions
themselves -- ``np.linalg.svd``, ``np.linalg.eig`` and ``np.linalg.qr`` appear
only in the test suite, as ground truth.
"""

from .eigen import jacobi_eigh, qr_eigh, symmetric_eigh
from .jacobi_svd import jacobi_svd
from .qr import householder_qr, modified_gram_schmidt, qr
from .randomized import randomized_svd
from .svd import BACKENDS, canonicalize_signs, explained_variance_ratio, svd

__all__ = [
    "BACKENDS",
    "canonicalize_signs",
    "explained_variance_ratio",
    "householder_qr",
    "jacobi_eigh",
    "jacobi_svd",
    "modified_gram_schmidt",
    "qr",
    "qr_eigh",
    "randomized_svd",
    "svd",
    "symmetric_eigh",
]
