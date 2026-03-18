# rflpa/agg/fltrust.py

from __future__ import annotations

from typing import Sequence, Union, Tuple, List
import numpy as np

print("[DEBUG][fl_trust.py] Module loaded")

ArrayLike = Union[np.ndarray, Sequence[float]]


def _to_1d_array(x: ArrayLike) -> np.ndarray:
    """
    Convert input to a 1-D float64 numpy array.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr

#client-side
def normalize_gradient( 
    client_grad: ArrayLike,
    g0_norm: float,
    eps: float = 1e-12,
) -> List[np.ndarray]:
    """
    Normalize client gradients w.r.t. server gradient.

    Parameters
    ----------
    client_grads:
        Sequence of client gradients g_i. Each can be any 1-D array-like.
    server_grad:
        Trusted server gradient g0 (from root dataset).
    eps:
        Small constant to avoid division by zero.

    Returns
    -------
    normed_grads:
        List of normalized gradients ḡ_i (same shape as g0).
    """
    print(f"[DEBUG][fl_trust.py] normalize_gradient called: g0_norm={g0_norm}")

    if g0_norm < eps:
        raise ValueError("Server gradient norm is zero; cannot normalize w.r.t. g0.")

    normed_grads: List[np.ndarray] = []
    for gi in client_grad:
        gi = _to_1d_array(gi)
        gi_norm = np.linalg.norm(gi)

        if gi_norm < eps:
            gbar_i = np.zeros_like(gi)
        else:
            # Normalization: ḡ_i = (||g0|| / ||g_i||) * g_i
            scale = g0_norm / gi_norm
            gbar_i = scale * gi

        normed_grads.append(gbar_i)
    
    print(f"[DEBUG][fl_trust.py] normalize_gradient completed: {len(normed_grads)} gradients normalized")
    return normed_grads

#client-side
def quantize_array(x: ArrayLike, q: int) -> np.ndarray:
    """
    Vectorized quantization Q(x) applied element-wise.

    Parameters
    ----------
    x : array-like of floats
        Values to quantize (e.g. normalized gradient ḡ_i).
    q : int
        Positive integer controlling precision (grid size 1/q).

    Returns
    -------
    qx : np.ndarray
        Quantized values with step size 1/q.
    """
    print(f"[DEBUG][fl_trust.py] quantize_array called: q={q}")

    if q <= 0:
        raise ValueError("q must be a positive integer")

    x = np.asarray(x, dtype=np.float64)
    y = q * x
    out = np.empty_like(x)

    pos_mask = x >= 0
    neg_mask = ~pos_mask

    out[pos_mask] = np.floor(y[pos_mask]) / q
    out[neg_mask] = (np.floor(y[neg_mask]) + 1.0) / q

    print(f"[DEBUG][fl_trust.py] quantize_array completed: shape={out.shape}")
    return out


def cosine_similarity(
    x: ArrayLike,
    y: ArrayLike,
    eps: float = 1e-12,
) -> float:
    """
    Compute cosine similarity cos(x, y) = <x, y> / (||x|| * ||y||).

    Returns 0.0 if either vector has near-zero norm.
    """
    x = _to_1d_array(x)
    y = _to_1d_array(y)

    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)

    if nx < eps or ny < eps:
        return 0.0

    result = float(np.dot(x, y) / (nx * ny))
    print(f"[DEBUG][fl_trust.py] cosine_similarity: result={result:.4f}")
    return result

#server-side
def compute_trust_score(
    client_grads: Sequence[ArrayLike],
    server_grad: ArrayLike,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute trust scores.

    Parameters
    ----------
    client_grads:
        Sequence of normalized client gradients g_i.
    server_grad:
        Trusted server gradient g0 (from root dataset).


    Returns
    -------
    trust_scores:
        1-D numpy array of trust scores TS_i (float64) of length len(client_grads).
    """
    print(f"[DEBUG][fl_trust.py] compute_trust_score called: {len(client_grads)} clients")
    
    g0 = _to_1d_array(server_grad)
    g0_norm = np.linalg.norm(g0)

    if g0_norm < eps:
        raise ValueError("Server gradient norm is zero; cannot normalize w.r.t. g0.")

    trust_scores: List[float] = []

    for idx, gi in client_grads:
        gi = _to_1d_array(gi)

        if gi.shape != g0.shape:
            raise ValueError(
                f"Shape mismatch: client gradient {gi.shape} vs server gradient {g0.shape}"
            )

        # compute cosine similarity using helper function
        cos_i = cosine_similarity(gi, g0, eps=eps)

        # Trust score: TS_i = max(0, cos_i)
        ts_i = max(0.0, cos_i)
        trust_scores.append(ts_i)
        print(f"[DEBUG][fl_trust.py] client {idx}: cosine={cos_i:.4f}, trust_score={ts_i:.4f}")

    result = np.asarray(trust_scores, dtype=np.float64)
    print(f"[DEBUG][fl_trust.py] compute_trust_score completed: scores={result}")
    return result



def fltrust_aggregate(
    client_grads: Sequence[ArrayLike],
    server_grad: ArrayLike,
    eps: float = 1e-12,
    q: int = 1000,
) -> np.ndarray:
    """
    Parameters
    ----------
    client_grads:
        Sequence of client gradients g_i.
    server_grad:
        Trusted server gradient g0 (from root dataset).
    eps:
        Small constant for numerical stability.
    q: 
        Quantization precision factor
    Returns
    -------
    g:
        Aggregated robust gradient as a 1-D float64 numpy array.
    """
    print(f"[DEBUG][fl_trust.py] fltrust_aggregate called: {len(client_grads)} clients, q={q}")
    
    g0 = _to_1d_array(server_grad)

    norm_client_grads = normalize_gradient(client_grads, g0, eps=eps)

    quant_client_grads: List[np.ndarray] = [
        quantize_array(gbar_i, q) for gbar_i in norm_client_grads
    ]

    trust_scores = compute_trust_score(quant_client_grads, g0, eps=eps)

    total_trust = float(trust_scores.sum())
    print(f"[DEBUG][fl_trust.py] total_trust={total_trust:.4f}")

    if total_trust < eps:
        print("[DEBUG][fl_trust.py] total trust near zero, returning server gradient")
        return g0.copy()

    # Weighted robust aggregation: sum_i TS_i * ḡ_i
    weighted_sum = np.zeros_like(g0)
    for gbar_i, ts_i in zip(norm_client_grads, trust_scores):
        if ts_i <= 0.0:
            continue
        weighted_sum += ts_i * gbar_i

    aggregated = weighted_sum / total_trust
    print(f"[DEBUG][fl_trust.py] fltrust_aggregate completed: result norm={np.linalg.norm(aggregated):.4f}")
    return aggregated
