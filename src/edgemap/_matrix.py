"""Internal matrix helpers for sparse expression handling."""

import numpy as np
from scipy import sparse


def ensure_csc_matrix(
    X: np.ndarray | sparse.spmatrix,
) -> np.ndarray | sparse.spmatrix:
    """Return a CSC sparse matrix view for sparse inputs.

    Dense arrays are returned unchanged. Sparse CSC inputs are reused as-is.
    """
    if not sparse.issparse(X):
        return X
    if sparse.isspmatrix_csc(X):
        return X
    return X.tocsc(copy=False)
