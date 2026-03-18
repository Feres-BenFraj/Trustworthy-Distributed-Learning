# shamir.py
from __future__ import annotations


from typing import List, Tuple

from finite_field import FiniteField
from polynomial import Polynomial


_FIELD = FiniteField()
_P = _FIELD.p



def share_secret(s: int, n: int, t: int) -> List[Tuple[int, int]]:
    """
    Shamir secret sharing over GF(p).

    Args:
        s: The secret as an integer. Interpreted modulo p.
        n: Number of shares to generate.
        t: Threshold (minimum number of shares needed to reconstruct).
           We use a random polynomial of degree t-1 with constant term s.

    Returns:
        A list of (x, y) pairs, where x and y are integers in GF(p).
    """
    if t < 1:
        raise ValueError("Threshold t must be at least 1")
    if t > n:
        raise ValueError("Threshold t cannot be larger than n")
    if n >= _P:
        raise ValueError("Number of shares n must be < field prime p")

    secret = s % _P

    # Random degree-(t-1) polynomial with fixed constant term = secret
    poly = Polynomial.random_with_constant(
        field=_FIELD,
        degree=t - 1,
        constant_term=secret,
    )

    # Evaluate at x = 1, 2, ..., n
    shares: List[Tuple[int, int]] = []
    for x in range(1, n + 1):
        y = poly(x)  # Polynomial class should evaluate in the finite field
        shares.append((x, y))

    return shares


def reconstruct_secret(shares: List[Tuple[int, int]]) -> int:
    """
    Reconstruct the secret from Shamir shares using Lagrange interpolation at x = 0.
    Args:
        shares: A list of (x, y) points, with distinct x values, all in GF(p).
                Must contain at least t shares that were produced from the same polynomial.

    Returns:
        The reconstructed secret.
    """
    if not shares:
        raise ValueError("Need at least one share to reconstruct")

    xs, ys = zip(*shares) 

    # Build the interpolating polynomial f(x) over the field
    poly = Polynomial.lagrange_interpolate(
        field=_FIELD,
        xs=list(xs),
        ys=list(ys),
    )

    return poly(0)
