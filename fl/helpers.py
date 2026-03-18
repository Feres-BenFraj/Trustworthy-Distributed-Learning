# my-rflpa/fl/helpers.py
"""
Helper utilities for RFLPA client protocol (Rounds 3 & 4).

Helper functions reused from the original RFLPA algorithm (crypto_utils):
  - mod_inverse            (same modular inverse logic as original)
  - inverse_matrix_mod     (adapted from crypto_utils.inversematrix)
  - compute_B_ej_matrix    (paper definition; matches original implementation)
  - compute_chop_d_matrix  (paper definition; matches original implementation)
  - compute_disaggregation_matrices (builds on the above, mirrors original flow)
  - lagrange_interpolate_at_point  (mirrors original interpolation routine)
  - reed_solomon_decode           (wrapper around interpolation as in original)
"""

from __future__ import annotations

from typing import List, Sequence, Tuple
import os
import numpy as np
import gmpy2

_DEBUG_HELPERS = os.getenv("RFLPA_DEBUG_HELPERS", "").strip().lower() in {"1", "true", "yes", "y", "on"}

# Cache Lagrange coefficients keyed by (p, target, xs_tuple).
# In this project we decode thousands of blocks using the same x-points, so this
# avoids doing expensive modular inverses repeatedly.
_LAGRANGE_COEFF_CACHE: dict[tuple[int, int, tuple[int, ...]], list[int]] = {}


def _dprint(msg: str) -> None:
    if _DEBUG_HELPERS:
        print(msg)


_dprint("[DEBUG][helpers.py] Module loaded")


class RFLPAHelper:
    """Stateless helper methods used during protocol execution."""

    def __init__(self):
        _dprint("[DEBUG][helpers.py] RFLPAHelper instance created")

    # ---------------------------------------------------------------------
    # Basic finite-field helpers (reused from original RFLPA crypto_utils)
    # ---------------------------------------------------------------------
    @staticmethod
    def mod_inverse(a: int, p: int) -> int:
        _dprint(f"[DEBUG][helpers.py] mod_inverse called: a={a}, p={p}")
        a = a % p
        if a == 0:
            raise ZeroDivisionError("Cannot compute inverse of 0")
        result = int(gmpy2.invert(gmpy2.mpz(a), gmpy2.mpz(p)))
        _dprint(f"[DEBUG][helpers.py] mod_inverse result: {result}")
        return result

    @staticmethod
    def identity_matrix(n: int) -> np.ndarray:
        _dprint(f"[DEBUG][helpers.py] identity_matrix called: n={n}")
        return np.array([[1 if x == y else 0 for x in range(n)] for y in range(n)], dtype=object)

    @staticmethod
    def inverse_matrix_mod(matrix: np.ndarray, p: int) -> np.ndarray:
        """
        Matrix inverse over F_p using Gauss-Jordan elimination.
        Adapted from the original crypto_utils.inversematrix.
        """
        _dprint(f"[DEBUG][helpers.py] inverse_matrix_mod called: matrix shape={matrix.shape}, p={p}")
        n = matrix.shape[0]
        A = np.array([[int(matrix[i, j]) % p for j in range(n)] for i in range(n)], dtype=object)
        A_inv = RFLPAHelper.identity_matrix(n)

        for col in range(n):
            pivot_row = None
            for row in range(col, n):
                if int(A[row, col]) % p != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                raise ValueError(f"Matrix is singular at column {col}")

            if pivot_row != col:
                A[[col, pivot_row]] = A[[pivot_row, col]]
                A_inv[[col, pivot_row]] = A_inv[[pivot_row, col]]

            pivot_val = int(A[col, col]) % p
            pivot_inv = RFLPAHelper.mod_inverse(pivot_val, p)
            for j in range(n):
                A[col, j] = (int(A[col, j]) * pivot_inv) % p
                A_inv[col, j] = (int(A_inv[col, j]) * pivot_inv) % p

            for row in range(n):
                if row != col:
                    factor = int(A[row, col]) % p
                    for j in range(n):
                        A[row, j] = (int(A[row, j]) - factor * int(A[col, j])) % p
                        A_inv[row, j] = (int(A_inv[row, j]) - factor * int(A_inv[col, j])) % p

        _dprint("[DEBUG][helpers.py] inverse_matrix_mod completed")
        return A_inv

    # ---------------------------------------------------------------------
    # Matrices used in dot-product aggregation (paper Sec. 4.5)
    # ---------------------------------------------------------------------
    @staticmethod
    def compute_B_ej_matrix(n: int, alpha_points: Sequence[int], e_j: int, p: int) -> np.ndarray:
        _dprint(f"[DEBUG][helpers.py] compute_B_ej_matrix called: n={n}, e_j={e_j}, p={p}")
        B_ej = np.zeros((n, n), dtype=object)
        for i in range(n):
            for k in range(n):
                base = (int(alpha_points[k]) - int(e_j)) % p
                B_ej[i, k] = pow(base, i, p)
        _dprint("[DEBUG][helpers.py] compute_B_ej_matrix completed")
        return B_ej

    @staticmethod
    def compute_chop_d_matrix(n: int, d: int) -> np.ndarray:
        _dprint(f"[DEBUG][helpers.py] compute_chop_d_matrix called: n={n}, d={d}")
        Chop_d = np.zeros((n, n), dtype=object)
        for i in range(min(d, n)):
            Chop_d[i, i] = 1
        return Chop_d

    @staticmethod
    def compute_disaggregation_matrices(
        n: int,
        d: int,
        l: int,
        alpha_points: Sequence[int],
        e_points: Sequence[int],
        p: int,
    ) -> List[np.ndarray]:
        _dprint(f"[DEBUG][helpers.py] compute_disaggregation_matrices called: n={n}, d={d}, l={l}, p={p}")
        Chop_d = RFLPAHelper.compute_chop_d_matrix(n, d)
        matrices = []
        for j in range(l):
            e_j = e_points[j]
            B_ej = RFLPAHelper.compute_B_ej_matrix(n, alpha_points, e_j, p)
            B_ej_inv = RFLPAHelper.inverse_matrix_mod(B_ej, p)

            result = np.zeros((n, n), dtype=object)
            for row in range(n):
                for col in range(n):
                    val = 0
                    for k in range(n):
                        val = (val + int(B_ej_inv[row, k]) * int(Chop_d[k, col])) % p
                    result[row, col] = val
            matrices.append(result)
        _dprint(f"[DEBUG][helpers.py] compute_disaggregation_matrices completed: {len(matrices)} matrices")
        return matrices

    # ---------------------------------------------------------------------
    # Interpolation / decoding (paper Sec. 4.5 Step 4)
    # ---------------------------------------------------------------------
    @staticmethod
    def lagrange_interpolate_at_point(
        shares: Sequence[Tuple[int, int]],
        target_point: int,
        p: int,
    ) -> int:
        _dprint(f"[DEBUG][helpers.py] lagrange_interpolate_at_point called: {len(shares)} shares, target={target_point}, p={p}")
        k = len(shares)
        if k == 0:
            raise ValueError("Need at least one share for interpolation")

        xs = [int(s[0]) % p for s in shares]
        ys = [int(s[1]) % p for s in shares]
        target = int(target_point) % p

        cache_key = (int(p), int(target), tuple(xs))
        coeffs = _LAGRANGE_COEFF_CACHE.get(cache_key)
        if coeffs is None:
            coeffs = []
            for i in range(k):
                numerator = 1
                denominator = 1
                for j in range(k):
                    if j != i:
                        numerator = (numerator * (target - xs[j])) % p
                        denominator = (denominator * (xs[i] - xs[j])) % p
                if denominator == 0:
                    raise ValueError("Duplicate x values in shares")
                L_i = (numerator * RFLPAHelper.mod_inverse(denominator, p)) % p
                coeffs.append(int(L_i))
            _LAGRANGE_COEFF_CACHE[cache_key] = coeffs

        result = 0
        for i in range(k):
            result = (result + ys[i] * coeffs[i]) % p
        _dprint(f"[DEBUG][helpers.py] lagrange_interpolate_at_point result: {result}")
        return result

    @staticmethod
    def reed_solomon_decode(
        shares: Sequence[Tuple[int, int]],
        secret_point: int,
        p: int,
    ) -> int:
        _dprint(f"[DEBUG][helpers.py] reed_solomon_decode called: {len(shares)} shares, secret_point={secret_point}")
        result = RFLPAHelper.lagrange_interpolate_at_point(shares, secret_point, p)
        _dprint(f"[DEBUG][helpers.py] reed_solomon_decode result: {result}")
        return result


