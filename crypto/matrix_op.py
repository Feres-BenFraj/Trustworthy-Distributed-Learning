import numpy as np
from packed_shamir import _P

def _inv_mod(a: int, p: int = _P) -> int:
    a %= p
    if a == 0:
        raise ZeroDivisionError("No inverse for 0 mod p")
    return pow(a, p - 2, p)  # p is prime

def _mat_inv_mod(A: np.ndarray, p: int = _P) -> np.ndarray:
    """
    Invert a square matrix A over F_p using Gauss-Jordan elimination.
    """
    A = np.array(A, dtype=object) % p
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Matrix must be square")

    I = np.zeros((n, n), dtype=object)
    for i in range(n):
        I[i, i] = 1

    # Augment [A | I]
    Aug = np.concatenate([A, I], axis=1) % p

    # Gauss-Jordan
    for col in range(n):
        # find pivot
        pivot = None
        for r in range(col, n):
            if Aug[r, col] % p != 0:
                pivot = r
                break
        if pivot is None:
            raise ValueError("Matrix is singular mod p")

        # swap pivot row
        if pivot != col:
            Aug[[col, pivot]] = Aug[[pivot, col]]

        # normalize pivot row
        inv_piv = _inv_mod(int(Aug[col, col]), p)
        Aug[col, :] = (Aug[col, :] * inv_piv) % p

        # eliminate others
        for r in range(n):
            if r == col:
                continue
            factor = Aug[r, col] % p
            if factor != 0:
                Aug[r, :] = (Aug[r, :] - factor * Aug[col, :]) % p

    invA = Aug[:, n:] % p
    return invA

def _build_B(alpha_points: list[int], e_j: int, p: int = _P) -> np.ndarray:
    """
    B_{e_j}[i,k] = (alpha_k - e_j)^(i)  with i=0..n-1 (paper uses i-1 with 1-indexing)
    """
    n = len(alpha_points)
    B = np.zeros((n, n), dtype=object)
    for i in range(n):          # i = 0..n-1
        for k in range(n):      # k = 0..n-1
            B[i, k] = pow((alpha_points[k] - e_j) % p, i, p)
    return B 

def _chop_matrix(n: int, d: int, p: int = _P) -> np.ndarray:
    """
    Chop_d: keep only the first (d+1) 'coefficients' (degree <= d).
    We'll implement as a diagonal matrix with 1s on positions 0..d.
    """
    C = np.zeros((n, n), dtype=object)
    upto = min(d + 1, n)
    for i in range(upto):
        C[i, i] = 1
    return C 

