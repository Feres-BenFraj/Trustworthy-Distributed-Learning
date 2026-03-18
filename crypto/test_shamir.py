# test_shamir.py
from itertools import combinations

from shamir import share_secret, reconstruct_secret
from packed_shamir import pack_and_share, reconstruct_packed


 # --------- test normal shamir ---------- #

def test_share_and_reconstruct_small_secret():
    secret = 123456789
    n = 5
    t = 3

    shares = share_secret(secret, n, t)

    # Try all size-t subsets
    for subset in combinations(shares, t):
        recovered = reconstruct_secret(list(subset))
        assert recovered == secret


def test_reconstruct_with_all_shares():
    secret = 42
    n = 10
    t = 4

    shares = share_secret(secret, n, t)
    recovered = reconstruct_secret(shares)
    assert recovered == secret


def test_multiple_random_secrets():
    # A few random secrets just to see nothing explodes
    for s in [0, 1, 2, 1234, 999999999999]:
        shares = share_secret(s, n=6, t=3)
        # Pick some arbitrary subset of size 3
        subset = list(shares[:3])
        recovered = reconstruct_secret(subset)
        assert recovered == s


 # --------- test packed shamir ---------- #

def test_share_and_reconstruct_small_packed():
    secrets = [123456789, 23456789, 34567891]

    n = 5
    t = 3

    e_points = [1, 2, 3]
    alpha_points = [10, 11, 12, 13, 14]

    shares = pack_and_share(secrets, n, t, e_points, alpha_points)

    # Try all size-t subsets
    for subset in combinations(shares, t):
        recovered = reconstruct_packed(list(subset), e_points, alpha_points)
        assert recovered == secrets

