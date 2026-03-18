from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import secrets
import sys
import os

# Add the rflpa-main directory to path BEFORE importing
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Try different import paths
try:
    from finite_field import FiniteField
except ImportError:
    from rflpa.finite_field import FiniteField


@dataclass
class Polynomial:
    """
    Polynomial over a given FiniteField.
    """

    field: FiniteField
    coeffs: List[int]

    def __post_init__(self):
        # Normalize coefficients into the field and strip leading zeros
        p = self.field.p
        self.coeffs = [c % p for c in self.coeffs]
        self._strip_leading_zeros()

    def _strip_leading_zeros(self) -> None:
        """Remove highest-degree zero coefficients (but keep at least one)."""
        while len(self.coeffs) > 1 and self.coeffs[-1] == 0:
            self.coeffs.pop()

    @property
    def degree(self) -> int:
        """Return the degree of the polynomial."""
        return len(self.coeffs) - 1

    def __repr__(self) -> str:
        return f"Polynomial(coeffs={self.coeffs}, p={self.field.p})"

    # --------- evaluation ---------- #

    def __call__(self, x: int) -> int:
        """
        Evaluate the polynomial at x.
        """
        p = self.field.p
        x = x % p
        result = 0
        for c in reversed(self.coeffs):
            result = (result * x + c) % p
        return result

    # --------- arithmetic on polynomials ---------- #

    def add(self, other: Polynomial) -> Polynomial:
        assert self.field.p == other.field.p
        p = self.field.p
        max_len = max(len(self.coeffs), len(other.coeffs))
        a = self.coeffs + [0] * (max_len - len(self.coeffs))
        b = other.coeffs + [0] * (max_len - len(other.coeffs))
        return Polynomial(self.field, [(ai + bi) % p for ai, bi in zip(a, b)])

    def sub(self, other: Polynomial) -> Polynomial:
        assert self.field.p == other.field.p 
        p = self.field.p
        max_len = max(len(self.coeffs), len(other.coeffs))
        a = self.coeffs + [0] * (max_len - len(self.coeffs))
        b = other.coeffs + [0] * (max_len - len(other.coeffs))
        return Polynomial(self.field, [(ai - bi) % p for ai, bi in zip(a, b)])

    def mul(self, other: Polynomial) -> Polynomial:
        assert self.field.p == other.field.p
        p = self.field.p
        a, b = self.coeffs, other.coeffs
        res = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            if ai == 0:
                continue
            for j, bj in enumerate(b):
                if bj == 0:
                    continue
                res[i + j] = (res[i + j] + ai * bj) % p
        return Polynomial(self.field, res)

    def scale(self, k: int) -> Polynomial:
        p = self.field.p
        k %= p
        return Polynomial(self.field, [(k * c) % p for c in self.coeffs])

    # --------- Lagrange interpolation ---------- #
    @classmethod
    def zero(cls, field: FiniteField) -> Polynomial: 
        return cls(field, [0])
    @classmethod
    def one(cls, field: FiniteField) -> Polynomial: 
        return cls(field, [1])
    
    @classmethod
    def lagrange_interpolate(
        cls,
        field: FiniteField,
        xs: Sequence[int],
        ys: Sequence[int],
    ) -> Polynomial:
        """
        Basic Lagrange Interpolation.
        Assumes all xs are distinct modulo p.
        """
        if len(xs) != len(ys):
            raise ValueError("xs and ys must have the same length")
        if not xs:
            raise ValueError("Need at least one point for interpolation")

        p = field.p
        n = len(xs)
        xs = [x % p for x in xs]
        ys = [y % p for y in ys]

        # Check distinct x's
        if len(set(xs)) != n:
            raise ValueError("All x-values must be distinct modulo p")

        result = cls.zero(field)

        for j in range(n):
            xj, yj = xs[j], ys[j]

            
            # First build numerator polynomial N_j(x) = ∏_{m != j} (x - x_m)
            Lj_num = cls.one(field)   # starts as 1
            denom = 1

            for m in range(n):
                if m == j:
                    continue
                xm = xs[m]
                # Multiply numerator by (x - xm)
                factor = cls(field, [(-xm) % p, 1])  # (x - xm)
                Lj_num = Lj_num.mul(factor)

                # Update denominator: (xj - xm)
                denom = (denom * ((xj - xm) % p)) % p

            # Inverse of denom in the field: denom^(p-2) mod p
            if denom == 0:
                raise ZeroDivisionError("Interpolation denominator is zero; xs not distinct?")
            denom_inv = pow(denom, p - 2, p)

            # L_j(x) = N_j(x) * denom_inv
            Lj = Lj_num.scale(denom_inv)

            # Add contribution y_j * L_j(x)
            term = Lj.scale(yj)
            result = result.add(term)

        return result

    # --------- random polynomial generation ---------- #

    @classmethod
    def random_with_constant(
        cls,
        field: FiniteField,
        degree: int,
        constant_term: int,
    ) -> Polynomial:
        """
        Generate a random polynomial of given degree with fixed constant term.
        """
        if degree < 0:
            raise ValueError("Degree must be non-negative")

        p = field.p
        constant_term %= p

        if degree == 0:
            return cls(field, [constant_term])

        coeffs = [0] * (degree + 1)
        coeffs[0] = constant_term

        # Random coefficients for x^1 .. x^(degree-1)
        for i in range(1, degree):
            coeffs[i] = secrets.randbelow(p)

        leading = 0
        while leading == 0:
            leading = secrets.randbelow(p)
        coeffs[degree] = leading

        return cls(field, coeffs)
