# rflpa/crypto/packed_shamir.py
from __future__ import annotations

import sys
import os
# Add the rflpa-main directory to path BEFORE importing
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from typing import List, Tuple, Sequence

# Try different import paths
try:
    from finite_field import FiniteField
except ImportError:
    from rflpa.finite_field import FiniteField

try:
    from polynomial import Polynomial
except ImportError:
    try:
        from crypto.polynomial import Polynomial
    except ImportError:
        from rflpa.crypto.polynomial import Polynomial

print("[DEBUG][packed_shamir.py] Module loaded")

_FIELD = FiniteField()
_P = _FIELD.p

print(f"[DEBUG][packed_shamir.py] Using prime p={_P}")


def pack_and_share( secrets: List[int], n: int, d: int, e_points: Sequence[int], alpha_points: Sequence[int],
) -> List[Tuple[int, int]]:
    """
    Packed Shamir secret sharing.

    Encode l secrets (s_1, ..., s_l) into one polynomial φ(x) of degree <= d
    such that φ(e_i) = s_i for each i (Eq. (14) in the paper).
    Then evaluate at α_j to obtain shares s_j = φ(α_j) (Eq. (15)).

    Args:
        secrets: list of l secrets [s_1, ..., s_l], integers interpreted mod p.
        n: total number of parties / shares to produce.
        d: maximum allowed degree of φ(x). Must satisfy l <= d + 1.
        e_points: evaluation points for the secrets, length l.
                  We enforce φ(e_i) = s_i.
        alpha_points: points where shares are evaluated, length n.
                      Each party j gets share (α_j, φ(α_j)).

    Returns:
        A list of length n of pairs (alpha_j, share_j), where share_j = φ(α_j).

    """
   # print(f"[DEBUG][packed_shamir.py] pack_and_share called: {len(secrets)} secrets, n={n}, d={d}")
    
    if len(secrets) != len(e_points):
        raise ValueError(f"secrets and e_points must have the same length: {len(secrets)}, {len(e_points)}")

    l = len(secrets)

    if l == 0:
        raise ValueError("Need at least one secret to pack")

    if l > d + 1:
        raise ValueError(f"Cannot pack {l} secrets into degree <= {d} polynomial; need l <= d + 1")

    if len(alpha_points) != n:
        raise ValueError("alpha_points length must equal n")

    if n >= _P:
        raise ValueError("Number of shares n must be < field prime p")

    p = _P

    xs = [e % p for e in e_points]
    ys = [s % p for s in secrets]

    # Distinctness checks
    if len(set(xs)) != l:
        raise ValueError("All e_points must be distinct modulo p")

    alphas = [a % p for a in alpha_points]
    if len(set(alphas)) != n:
        raise ValueError("All alpha_points must be distinct modulo p")

    # Interpolate φ such that φ(e_i) = s_i, Eq. (14).
    #print(f"[DEBUG][packed_shamir.py] interpolating polynomial")
    phi0 = Polynomial.lagrange_interpolate(
        field=_FIELD,
        xs=xs,
        ys=ys,
    )

    degree_random = d - len(secrets)

    if degree_random < 0:
        random = Polynomial.zero(_FIELD)
    else:
        random = Polynomial.random_with_constant(_FIELD, degree_random, 0)

    vanish = Polynomial.one(_FIELD)

    for e in e_points:
        factor = Polynomial(_FIELD, [(-e) % _FIELD.p, 1]) 
        vanish = vanish.mul(factor)

    phi = phi0.add(random.mul(vanish)) # Final Polynomial

    # Eq. (15): shares s_j = φ(α_j)
    shares: List[Tuple[int, int]] = []
    for alpha in alphas:
        share_val = phi(alpha)
        shares.append((alpha, share_val))

    #print(f"[DEBUG][packed_shamir.py] pack_and_share completed: {len(shares)} shares created")
    return shares


def reconstruct_packed(shares_subset: Sequence[Tuple[int, int]], e_points: Sequence[int], alpha_points: Sequence[int],
) -> List[int]:
    """
    Reconstruct the packed secrets from a subset of shares.

    Args:
        shares_subset: list of (alpha_j, share_j) pairs from (some of) the parties.
                       Must contain at least deg(φ)+1 points from the same φ.
        e_points: the secret-encoding points [e_1, ..., e_l].
        alpha_points: all α_j used in pack_and_share (for consistency checks).

    Returns:
        A list of secrets [s_1, ..., s_l] = [φ(e_1), ..., φ(e_l)].
    """
    print(f"[DEBUG][packed_shamir.py] reconstruct_packed called: {len(shares_subset)} shares, {len(e_points)} secrets")
    
    if not shares_subset:
        raise ValueError("Need at least one share to reconstruct")

    p = _P

    # Optional: check that the x-values in shares_subset are among the alpha_points
    alpha_set = {a % p for a in alpha_points}
    for (alpha, _) in shares_subset:
        if (alpha % p) not in alpha_set:
            raise ValueError("Share with alpha not in the provided alpha_points")

    xs, ys = zip(*shares_subset)
    xs = [x % p for x in xs]
    ys = [y % p for y in ys]

    # Interpolate 
    print(f"[DEBUG][packed_shamir.py] interpolating for reconstruction")
    phi = Polynomial.lagrange_interpolate(
        field=_FIELD,
        xs=xs,
        ys=ys,
    )

    # Recover each packed secret as φ(e_i)
    secrets: List[int] = []
    for e in e_points:
        val = phi(e % p)
        secrets.append(val % p)

    print(f"[DEBUG][packed_shamir.py] reconstruct_packed completed: {len(secrets)} secrets recovered")
    return secrets
