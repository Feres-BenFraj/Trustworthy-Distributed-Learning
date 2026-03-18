from __future__ import annotations
from Crypto.Util.number import getPrime

# Fixed prime for consistent operations across all modules
# Using a safe 256-bit prime (generated once, used everywhere)
_DEFAULT_PRIME = 115792089237316195423570985008687907853269984665640564039457584007913129639747

class FiniteField:
    """
    Finite field for a prime p.
    Values are represented as Python integers modulo p.
    """

    def __init__(self, b: int = 256, use_fixed: bool = True):
        """
        :param b: bit length for the randomly generated prime modulus p.
        :param use_fixed: if True, use the fixed default prime for consistency.
        """
        if use_fixed:
            self.p = _DEFAULT_PRIME
        else:
            self.p = getPrime(b)

    # ---- basic helpers ---- #

    def normalize(self, x: int) -> int:
        """Reduce x modulo p into the canonical range [0, p-1]."""
        return x % self.p

    # ---- four basic operations ---- #

    def add(self, a: int, b: int) -> int:
        """(a + b) mod p"""
        return (a + b) % self.p

    def sub(self, a: int, b: int) -> int:
        """(a - b) mod p"""
        return (a - b) % self.p

    def mul(self, a: int, b: int) -> int:
        """(a * b) mod p"""
        return (a * b) % self.p

    def div(self, a: int, b: int) -> int:
        """
        (a / b) mod p, implemented as a * inv(b) mod p.
        Raises ZeroDivisionError if b == 0 mod p.
        """
        inv_b = self.inv(b)
        return (a * inv_b) % self.p

    # ---- extra useful operations ---- #

    def neg(self, a: int) -> int:
        """(-a) mod p"""
        return (-a) % self.p

    def inv(self, a: int) -> int:
        """
        Multiplicative inverse of a mod p using extended Euclidean algorithm.
        Raises ZeroDivisionError if a == 0 mod p.
        """
        a = a % self.p
        if a == 0:
            raise ZeroDivisionError("Cannot invert 0 in a field.")

        # Extended Euclidean algorithm
        t, new_t = 0, 1
        r, new_r = self.p, a
        while new_r != 0:
            q = r // new_r
            t, new_t = new_t, t - q * new_t
            r, new_r = new_r, r - q * new_r

        # r is now gcd(p, a); gcd must be 1 for invertible
        if r > 1:
            raise ZeroDivisionError(f"{a} has no inverse modulo {self.p}")

        if t < 0:
            t += self.p
        return t

    def pow(self, a: int, e: int) -> int:
        """
        a^e mod p with support for negative exponents (via inverses).
        """
        a = a % self.p
        if e < 0:
            a = self.inv(a)
            e = -e
        return pow(a, e, self.p)