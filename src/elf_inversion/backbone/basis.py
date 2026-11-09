"""Coordinates of reachable codes, shared by every model's forward map."""
import jax.numpy as jnp
import numpy as np


def row_basis(matrix):
    """Orthonormal basis of the numerically reachable row space (float32 rank)."""
    a = np.asarray(matrix, dtype=np.float64)
    _, singular, vh = np.linalg.svd(a, full_matrices=False)
    cutoff = np.finfo(np.float32).eps * max(a.shape) * (singular[0] if len(singular) else 0)
    rank = int(np.count_nonzero(singular > cutoff))
    return jnp.asarray(vh[:rank], dtype=jnp.float32), singular, cutoff
