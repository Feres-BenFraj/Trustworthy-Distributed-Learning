"""
RFLPA Server Utility Functions
Contains helper functions, state management, and cryptographic utilities for the server.
"""

from __future__ import annotations

import sys
import os

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import numpy as np
import torch
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any

# Import from fl folder (matches nodes.py pattern)
from fl.helpers import RFLPAHelper

# Import from agg folder (matches nodes.py pattern)
from agg.fl_trust import normalize_gradient, quantize_array

# Import from crypto folder (matches nodes.py pattern)
from crypto.packed_shamir import pack_and_share, _P

print("[DEBUG][utils_server.py] Module loaded")

# Use the same prime as packed_shamir for consistent finite field operations
DEFAULT_PRIME = _P


@dataclass
class ServerState:
    """Container for server state during protocol execution."""
    global_model: Any
    num_clients: int
    min_clients: int
    prime: int = DEFAULT_PRIME
    
    # Crypto parameters
    n: int = 0  # Number of parties (shares)
    d: int = 0  # Polynomial degree
    l: int = 0 # Packing factor 1
    p: int = 0 # Packing factor 2
    alpha_points: List[int] = field(default_factory=list)
    e_points: List[int] = field(default_factory=list)
    e_points_p: List[int] = field(default_factory=list)
    secret_point: int = 0
    
    # Client tracking sets
    U0: Set[int] = field(default_factory=set)  # Active clients
    U1: Set[int] = field(default_factory=set)  # Round 1 respondents
    U2: Set[int] = field(default_factory=set)  # Round 2 respondents
    U3: Set[int] = field(default_factory=set)  # Round 3 respondents
    U4: Set[int] = field(default_factory=set)  # Round 4 respondents
    
    # Server's reference gradient
    server_gradient: Optional[np.ndarray] = None
    server_gradient_norm: float = 0.0
    server_gradient_shares: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    
    # Round message storage
    round1_messages: Dict[int, Dict] = field(default_factory=dict)
    round2_messages: Dict[int, Dict] = field(default_factory=dict)
    round3_messages: Dict[int, Dict] = field(default_factory=dict)
    round4_messages: Dict[int, Any] = field(default_factory=dict)
    
    # Recovered values
    client_norms: Dict[int, float] = field(default_factory=dict)
    client_cosine_similarities: Dict[int, float] = field(default_factory=dict)
    trust_scores: Dict[int, float] = field(default_factory=dict)
    
    # Client keys
    client_public_keys: Dict[int, Any] = field(default_factory=dict)
    client_encryption_keys: Dict[Tuple[int, int], bytes] = field(default_factory=dict)
    
    def __post_init__(self):
        print(f"[DEBUG][utils_server.py] ServerState created: num_clients={self.num_clients}, min_clients={self.min_clients}, prime={self.prime}")


class ServerCryptoUtils:
    """Cryptographic utility functions for the server using utils folder functions."""
    
    def __init__(self, args):
        print("[DEBUG][utils_server.py] ServerCryptoUtils initialized")
        self.helper = RFLPAHelper()  # Use RFLPAHelper from fl.helpers
        self.args = args
    """
    def init_crypto_params(self, state: ServerState):
        Initialize cryptographic parameters for packed secret sharing.
        print("[DEBUG][utils_server.py] init_crypto_params called")
        n = state.num_clients
        # Degree d must satisfy: 2d + 1 <= n for security
        d = (n - 1) // 2
        # Packing factor l (number of secrets per polynomial)
        l = max(1, d // 2)
        
        state.n = n
        state.d = d
        state.l = l
        
        # Generate evaluation points (must be distinct and non-zero in F_p)
        p = state.prime
        state.alpha_points = [(i + 1) % p for i in range(n)]
        state.e_points = [-(i + 1) % p for i in range(l)]
        state.secret_point = 0
        
        print(f"[DEBUG][utils_server.py] crypto params: n={n}, d={d}, l={l}")
        print(f"[DEBUG][utils_server.py] alpha_points={state.alpha_points[:5]}... e_points={state.e_points}")
    """
    
    def init_crypto_params(self, state: ServerState):
        """Initialize cryptographic parameters for packed secret sharing (paper-consistent)."""
        print("[DEBUG][utils_server.py] init_crypto_params called")

        n = state.num_clients
        if n <= 0:
            raise ValueError("num_clients must be > 0")

        # Degree d: tolerate up to d malicious / dropped parties
        # Security condition: 2d + 1 <= n
        d = (n - 1) // 2

        # Packing factor l: number of secrets per polynomial
        # Must satisfy: d >= l - 1
        l = max(1, d // 2)

        if d < l - 1:
            raise ValueError(f"Invalid parameters: d={d} too small for l={l}")

        p = d + 1

        state.n = n
        state.d = d
        state.l = l
        state.p = p
        

        # --- Deterministic, disjoint points (RECOMMENDED) ---

        # Secret embedding points: exactly l points
        state.e_points = list(range(1, l + 1))
        state.e_points_p = list(range(1, p + 1))
        b = max(l,p)
        # Share evaluation points: exactly n points, disjoint from e_points
        start = b + 1
        state.alpha_points = list(range(start, start + n))

        # Secret point for final decoding - must match e_points[0] for packed Shamir
        # Secrets are embedded at e_points, so we reconstruct at e_points[0]
        state.secret_point = state.e_points[0]  # = 1

        # --- Sanity checks (DO NOT REMOVE) ---
        if len(state.e_points) != l:
            raise AssertionError("e_points length mismatch")
        if len(state.e_points_p) != p:
            raise AssertionError("e_points_p length mismatch")
        if len(state.alpha_points) != n:
            raise AssertionError("alpha_points length mismatch")
        if set(state.e_points) & set(state.alpha_points):
            raise AssertionError("e_points and alpha_points must be disjoint")
        if l + n + 1 >= state.prime:
            raise ValueError("Field prime p is too small for selected parameters")

        print(f"[DEBUG][utils_server.py] crypto params: n={n}, d={d}, l={l}, p={p}")
        print(f"[DEBUG][utils_server.py] e_points={state.e_points}")
        print(f"[DEBUG][utils_server.py] e_points_p={state.e_points_p}")
        print(f"[DEBUG][utils_server.py] alpha_points={state.alpha_points[:5]}...")
    
    def create_server_gradient_shares(self, state: ServerState):
        """
        Create packed secret shares v0 of server gradient for distribution.
        Uses pack_and_share from crypto.packed_shamir (same as nodes.py).
        """
        print("[DEBUG][utils_server.py] create_server_gradient_shares called")
        
        if state.server_gradient is None:
            raise RuntimeError("Server gradient not computed.")
        
        # Normalize gradient using fl_trust utility (same as nodes.py)
        grad = state.server_gradient
        norm = state.server_gradient_norm
        
        if norm > 0:
            # Use normalize_gradient from agg.fl_trust
            normalized_grad = normalize_gradient([grad], norm)[0]
        else:
            normalized_grad = grad
        
        # Quantize using quantize_array from agg.fl_trust (same as nodes.py)
        q = getattr(self.args, 'precision', 1000)
        quantized_grad = quantize_array(normalized_grad, q=q)
        
        print(f"[DEBUG][utils_server.py] gradient shape={grad.shape}, quantized={quantized_grad.shape}")
        
        # Pack into blocks and create shares for each client
        l = state.l
        # quantized_grad is on 1/q grid (e.g., 0.052 for q=1000)
        # We need to multiply by q to get integer-coded secrets (e.g., 52)
        secrets_list = np.rint(quantized_grad * q).astype(np.int64).tolist()
        blocks = [secrets_list[i:i+l] for i in range(0, len(secrets_list), l)]
        
        # Pad last block
        if blocks and len(blocks[-1]) < l:
            blocks[-1] += [0] * (l - len(blocks[-1]))
        
        print(f"[DEBUG][utils_server.py] created {len(blocks)} blocks of size {l}")
        
        # Create shares for each client using pack_and_share from crypto.packed_shamir
        n = state.n
        
        # Initialize shares dict for each client
        for client_id in range(n):
            state.server_gradient_shares[client_id] = []
        
        for block_idx, block in enumerate(blocks):
            # Use pack_and_share from crypto.packed_shamir (same as nodes.py)
            shares_for_block = pack_and_share(
                secrets=block,
                n=n,
                d=state.d,
                e_points=state.e_points,
                alpha_points=state.alpha_points,
            )
            for party_idx, share in enumerate(shares_for_block):
                state.server_gradient_shares[party_idx].append(share)
        
        print(f"[DEBUG][utils_server.py] created shares for {n} clients, {len(blocks)} blocks each")
    
    def recover_values_rs(self, shares_by_group: Dict[int, List[Tuple[int, int]]], state: ServerState) -> Dict[int, float]:
        """
        Recover values using Reed-Solomon decoding from RFLPAHelper.
        """
        print(f"[DEBUG][utils_server.py] recover_values_rs called: {len(shares_by_group)} groups")
        
        p = state.prime
        recovered = {}
        
        for group_idx, shares in shares_by_group.items():
            if len(shares) < state.min_clients:
                print(f"[DEBUG][utils_server.py] group {group_idx}: insufficient shares ({len(shares)})")
                continue
            
            # Use Lagrange interpolation at x=0 to recover secret
            points = [(s[0], int(s[1]) % p) for s in shares]
            
            try:
                # Use reed_solomon_decode from RFLPAHelper (same as nodes.py)
                secret = self.helper.reed_solomon_decode(
                    shares=points,
                    secret_point=state.secret_point,
                    p=p,
                )
                # Handle signed values (values near p are negative)
                if secret > p // 2:
                    secret = secret - p
                recovered[group_idx] = float(secret)
            except Exception as e:
                print(f"[DEBUG][utils_server.py] group {group_idx}: decode failed - {e}")
                recovered[group_idx] = 0.0
        
        print(f"[DEBUG][utils_server.py] recovered {len(recovered)} values")
        return recovered


class ServerModelUtils:
    """Model-related utility functions for the server."""
    
    @staticmethod
    def get_global_parameters(global_model) -> Dict[str, torch.Tensor]:
        """Return current global model parameters as state dict."""
        print("[DEBUG][utils_server.py] get_global_parameters called")
        return {name: param.clone() for name, param in global_model.state_dict().items()}
    
    @staticmethod
    def get_global_parameters_flat(global_model) -> np.ndarray:
        """Return flattened global model parameters."""
        print("[DEBUG][utils_server.py] get_global_parameters_flat called")
        params = []
        for param in global_model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        result = np.concatenate(params)
        print(f"[DEBUG][utils_server.py] flattened params shape={result.shape}")
        return result
    
    @staticmethod
    def update_global_model(global_model, gradient: np.ndarray, learning_rate: float = 1.0):
        """
        Update global model: w_t = w_{t-1} + gamma_t * delta_w
        
        Note: 'gradient' is actually the model update (new_params - old_params),
        so we ADD it to the global model (not subtract).
        """
        print(f"[DEBUG][utils_server.py] update_global_model called: lr={learning_rate}, gradient_shape={gradient.shape}")
        
        current_state = global_model.state_dict()
        
        # Reshape gradient to match model parameters
        idx = 0
        new_state = {}
        
        for name, param in global_model.named_parameters():
            num_params = param.numel()
            if idx + num_params <= len(gradient):
                param_update = gradient[idx:idx + num_params].reshape(param.shape)
                # ADD the update since gradient = new_params - old_params
                new_state[name] = current_state[name] + learning_rate * torch.tensor(
                    param_update, dtype=param.dtype, device=param.device
                )
            else:
                new_state[name] = current_state[name]
            idx += num_params
        
        # Update non-parameter buffers
        for name in current_state:
            if name not in new_state:
                new_state[name] = current_state[name]
        
        global_model.load_state_dict(new_state)
        print("[DEBUG][utils_server.py] global model updated")
