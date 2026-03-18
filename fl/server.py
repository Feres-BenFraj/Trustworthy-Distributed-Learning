#RFLPA Server Implementation
#Implements the server-side logic for the 4-round secure aggregation protocol
#with privacy-preserving verification. Works with Client in nodes.py.

#Protocol Overview:
#- Round 1: Collect encrypted gradient shares and commitments from clients
#- Round 2: Collect encrypted norm/cosine similarity shares  
#- Round 3: Collect final shares, compute trust scores via RS decoding
#- Round 4: Collect weighted aggregations and recover final gradient

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
from typing import Dict, List, Set, Tuple, Optional, Any

# Import utilities and round handlers from same folder
from fl.utils_server import (
    ServerState, ServerCryptoUtils, ServerModelUtils, DEFAULT_PRIME
)
from fl.round1_server import Round1Handler
from fl.round2_server import Round2Handler
from fl.round3_server import Round3Handler
from fl.round4_server import Round4Handler

# Import from fl folder (matches nodes.py pattern)
from fl.helpers import RFLPAHelper

# Import from agg folder (matches nodes.py pattern)
from agg.fl_trust import normalize_gradient, quantize_array

print("[DEBUG][server.py] Module loaded")


class RFLPAServer:
    
    #RFLPA Server implementing the 4-round secure aggregation protocol.
    
    #Works with Client from nodes.py to perform:
    #- Privacy-preserving gradient aggregation
    #- Trust score computation based on cosine similarity
    #- Byzantine-robust aggregation using FLTrust mechanism

    
    def __init__(
        self,
        args: Any,
        global_model: Any,
        num_clients: int,
        min_clients: int,
        prime: int = DEFAULT_PRIME,
    ):
        print(f"[DEBUG][server.py] RFLPAServer.__init__: num_clients={num_clients}, min_clients={min_clients}, prime={prime}")
        
        self.args = args
        self.helper = RFLPAHelper()  # Use RFLPAHelper from fl.helpers (same as nodes.py)
        
        self.state = ServerState(
            global_model=global_model,
            num_clients=num_clients,
            min_clients=min_clients,
            prime=prime,
        )
        
        # Initialize crypto utilities (uses RFLPAHelper internally)
        self.crypto_utils = ServerCryptoUtils(args)
        self.crypto_utils.init_crypto_params(self.state)
        
        # Initialize round handlers
        self.round1 = Round1Handler(self.state)
        self.round2 = Round2Handler(self.state)
        self.round3 = Round3Handler(self.state, self.crypto_utils)
        self.round4 = Round4Handler(self.state, self.crypto_utils, args)
        
        print("[DEBUG][server.py] RFLPAServer initialized")
    
    # =========================================================================
    # INITIALIZATION & SETUP
    # =========================================================================
    
    def set_active_clients(self, active_clients: Set[int]):
        print(f"[DEBUG][server.py] set_active_clients: {len(active_clients)} clients")
        self.state.U0 = active_clients.copy()
        self._reset_round_state()
    
    def _reset_round_state(self):
        print("[DEBUG][server.py] _reset_round_state called")
        self.state.U1 = set()
        self.state.U2 = set()
        self.state.U3 = set()
        self.state.U4 = set()
        self.state.round1_messages = {}
        self.state.round2_messages = {}
        self.state.round3_messages = {}
        self.state.round4_messages = {}
        self.state.trust_scores = {}
        self.state.client_norms = {}
        self.state.client_cosine_similarities = {}
    
    def compute_server_update(self, root_dataset: Any, train_fn: Any) -> Tuple[np.ndarray, float]:
        print("[DEBUG][server.py] compute_server_update called")
        
        # Get current model parameters
        current_params = self.get_global_parameters_flat()
        
        # Compute gradient using root dataset
        server_gradient = train_fn(current_params, root_dataset)
        server_gradient = np.asarray(server_gradient, dtype=np.float64).ravel()
        
        # Compute norm
        gradient_norm = float(np.linalg.norm(server_gradient))
        
        print(f"[DEBUG][server.py] server gradient shape={server_gradient.shape}, norm={gradient_norm:.4f}")
        
        # Store
        self.state.server_gradient = server_gradient
        self.state.server_gradient_norm = gradient_norm
        
        # Create packed secret shares of server gradient v0
        # Uses pack_and_share from crypto.packed_shamir (same as nodes.py)
        self.crypto_utils.create_server_gradient_shares(self.state)
        
        print("[DEBUG][server.py] compute_server_update completed")
        return server_gradient, gradient_norm
    
    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        print("[DEBUG][server.py] get_global_parameters called")
        #Return current global model parameters as state dict.
        return ServerModelUtils.get_global_parameters(self.state.global_model)
    
    def get_global_parameters_flat(self) -> np.ndarray:
        print("[DEBUG][server.py] get_global_parameters_flat called")
        #Return flattened global model parameters.
        return ServerModelUtils.get_global_parameters_flat(self.state.global_model)
    
    def get_client_round_data(self, client_id: int) -> Dict[str, Any]:
        print(f"[DEBUG][server.py] get_client_round_data: client_id={client_id}")
        #Get data to send to client at start of round.
        
        #Returns global model, v0 share for this client, and ||g0||.
        return {
            'global_model': self.get_global_parameters_flat(),
            'v0_share': self.state.server_gradient_shares.get(client_id, []),
            'g0_norm': self.state.server_gradient_norm,
        }
    
    def register_client_public_key(self, client_id: int, public_key: Any):
        print(f"[DEBUG][server.py] register_client_public_key: client_id={client_id}")
        """Register client's public key for signature verification."""
        self.state.client_public_keys[client_id] = public_key
    
    # =========================================================================
    # ROUND 1: Collect gradient shares
    # =========================================================================
    
    def receive_round1_message(self, client_id: int, message: Dict) -> bool:
        print(f"[DEBUG][server.py] receive_round1_message: client_id={client_id}")
        """Receive Round 1 message from client."""
        return self.round1.receive_message(client_id, message)
    
    def is_round1_complete(self) -> bool:
        result = self.round1.is_complete()
        print(f"[DEBUG][server.py] is_round1_complete: {result}")
        """Check if at least K clients responded in Round 1."""
        return result
    
    def prepare_round1_response(self, target_client: int) -> Optional[Dict]:
        print(f"[DEBUG][server.py] prepare_round1_response: target_client={target_client}")
        """Prepare Round 1 response for client j."""
        return self.round1.prepare_response(target_client)
    
    # =========================================================================
    # ROUND 2: Collect partial norm/cosine similarity shares
    # =========================================================================
    
    def receive_round2_message(self, client_id: int, message: Dict) -> bool:
        print(f"[DEBUG][server.py] receive_round2_message: client_id={client_id}")
        """Receive Round 2 message from client."""
        return self.round2.receive_message(client_id, message)
    
    def is_round2_complete(self) -> bool:
        result = self.round2.is_complete()
        print(f"[DEBUG][server.py] is_round2_complete: {result}")
        """Check if at least K clients responded in Round 2."""
        return result
    
    def prepare_round2_response(self, target_client: int) -> Optional[Dict]:
        print(f"[DEBUG][server.py] prepare_round2_response: target_client={target_client}")
        """Prepare Round 2 response for client j."""
        return self.round2.prepare_response(target_client)
    
    # =========================================================================
    # ROUND 3: Collect final shares and compute trust scores
    # =========================================================================
    
    def receive_round3_message(self, client_id: int, message: Dict) -> bool:
        print(f"[DEBUG][server.py] receive_round3_message: client_id={client_id}")
        """
        Receive Round 3 message from client.
        
        Accepts output from nodes.py get_round3_output():
            {"cs_final": [...], "nr_final": [...]}
        """
        return self.round3.receive_message(client_id, message)
    
    def is_round3_complete(self) -> bool:
        result = self.round3.is_complete()
        print(f"[DEBUG][server.py] is_round3_complete: {result}")
        """Check if at least K clients responded in Round 3."""
        return result
    
    def process_round3_and_compute_trust_scores(self) -> Dict[int, float]:
        print("[DEBUG][server.py] process_round3_and_compute_trust_scores called")
        """Process Round 3 messages and compute trust scores."""
        return self.round3.process_and_compute_trust_scores()
    
    def broadcast_trust_scores(self) -> Dict[int, float]:
        print("[DEBUG][server.py] broadcast_trust_scores called")
        """
        Broadcast trust scores to all clients in U3.
        
        Clients receive via nodes.py receive_trust_scores() method.
        """
        return self.round3.broadcast_trust_scores()
    
    # =========================================================================
    # ROUND 4: Collect aggregated shares and recover global gradient
    # =========================================================================
    
    def receive_round4_message(self, client_id: int, aggregated_shares: List[Tuple[int, int]]) -> bool:
        print(f"[DEBUG][server.py] receive_round4_message: client_id={client_id}")
        """
        Receive Round 4 message from client.
        
        Accepts output from nodes.py get_round4_output():
            [(block_idx, aggregated_share_value), ...]
        """
        return self.round4.receive_message(client_id, aggregated_shares)
    
    def is_round4_complete(self) -> bool:
        result = self.round4.is_complete()
        print(f"[DEBUG][server.py] is_round4_complete: {result}")
        """Check if at least K clients responded in Round 4."""
        return result
    
    def recover_global_gradient(self) -> Optional[np.ndarray]:
        print("[DEBUG][server.py] recover_global_gradient called")
        """
        Recover global aggregated gradient g using Reed-Solomon decoding.
        Uses RFLPAHelper.reed_solomon_decode (same as nodes.py).
        """
        return self.round4.recover_global_gradient()
    
    def update_global_model(self, gradient: np.ndarray, learning_rate: float = 1.0):
        print(f"[DEBUG][server.py] update_global_model: lr={learning_rate}")
        """Update global model: w_t = w_{t-1} - gamma_t * g"""
        ServerModelUtils.update_global_model(self.state.global_model, gradient, learning_rate)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_crypto_params(self) -> Dict[str, Any]:
        print("[DEBUG][server.py] get_crypto_params called")
        """
        Get cryptographic parameters for clients.
        
        These parameters are used by nodes.py methods like share_gradient(),
        round3_dot_product_aggregation(), etc.
        """
        return {
            'n': self.state.n,
            'd': self.state.d,
            'l': self.state.l,
            'p': self.state.p,
            'alpha_points': self.state.alpha_points,
            'e_points': self.state.e_points,
            'e_points_p': self.state.e_points_p,
            'secret_point': self.state.secret_point,
            'prime': self.state.prime,
        }
    
    def get_respondent_sets(self) -> Dict[str, Set[int]]:
        print("[DEBUG][server.py] get_respondent_sets called")
        """Return all respondent sets for debugging/logging."""
        return {
            'U0': self.state.U0.copy(),
            'U1': self.state.U1.copy(),
            'U2': self.state.U2.copy(),
            'U3': self.state.U3.copy(),
            'U4': self.state.U4.copy(),
        }
    
    def get_client_count(self, round_num: int) -> int:
        counts = {
            0: len(self.state.U0),
            1: len(self.state.U1),
            2: len(self.state.U2),
            3: len(self.state.U3),
            4: len(self.state.U4),
        }
        count = counts.get(round_num, 0)
        print(f"[DEBUG][server.py] get_client_count: round={round_num}, count={count}")
        """Get number of responding clients for a specific round."""
        return count
    
    @property
    def trust_scores(self) -> Dict[int, float]:
        """Return computed trust scores."""
        return self.state.trust_scores
    
    @property
    def global_model(self):
        """Return global model."""
        return self.state.global_model


class RFLPAServerSimple:
    """
    Simplified RFLPA Server for testing without full cryptographic operations.
    Useful for debugging and attack simulations.
    """
    
    def __init__(self, args: Any, global_model: Any, num_clients: int, min_clients: int):
        print(f"[DEBUG][server.py] RFLPAServerSimple.__init__: num_clients={num_clients}, min_clients={min_clients}")
        self.args = args
        self.global_model = global_model
        self.num_clients = num_clients
        self.min_clients = min_clients
        self.helper = RFLPAHelper()  # Use RFLPAHelper from fl.helpers
        
        self.client_updates: Dict[int, Dict] = {}
        self.trust_scores: Dict[int, float] = {}
        self.server_gradient: Optional[np.ndarray] = None
        self.server_gradient_norm: float = 0.0
    
    def set_server_gradient(self, gradient: np.ndarray):
        print(f"[DEBUG][server.py] RFLPAServerSimple.set_server_gradient: shape={gradient.shape}")
        #Set reference gradient for trust score computation.
        self.server_gradient = np.asarray(gradient, dtype=np.float64).ravel()
        self.server_gradient_norm = float(np.linalg.norm(self.server_gradient))
        print(f"[DEBUG][server.py] server_gradient_norm={self.server_gradient_norm:.4f}")
    
    def receive_client_update(self, client_id: int, gradient: np.ndarray):
        print(f"[DEBUG][server.py] RFLPAServerSimple.receive_client_update: client_id={client_id}")
        #Receive and store a client update.
        grad = np.asarray(gradient, dtype=np.float64).ravel()
        self.client_updates[client_id] = {
            'gradient': grad,
            'norm': float(np.linalg.norm(grad)),
        }
    
    def compute_trust_scores(self) -> Dict[int, float]:
        print("[DEBUG][server.py] RFLPAServerSimple.compute_trust_scores called")
        #Compute trust scores based on cosine similarity with server gradient.
        if self.server_gradient is None:
            return {}
        
        for client_id, data in self.client_updates.items():
            client_grad = data['gradient']
            client_norm = data['norm']
            
            if client_norm > 0 and self.server_gradient_norm > 0:
                dot_product = np.dot(client_grad, self.server_gradient)
                cosine_sim = dot_product / (client_norm * self.server_gradient_norm)
                self.trust_scores[client_id] = max(0.0, float(cosine_sim))
            else:
                self.trust_scores[client_id] = 0.0
            
            print(f"[DEBUG][server.py] client_id={client_id}, trust_score={self.trust_scores[client_id]:.4f}")
        
        return self.trust_scores
    
    def aggregate(self) -> np.ndarray:
        print("[DEBUG][server.py] RFLPAServerSimple.aggregate called")
        #Perform weighted aggregation based on trust scores.
        if not self.trust_scores:
            self.compute_trust_scores()
        
        total_trust = sum(self.trust_scores.values())
        print(f"[DEBUG][server.py] total_trust={total_trust:.4f}")
        
        if total_trust == 0:
            # Fallback to uniform weights
            weights = {cid: 1.0 / len(self.client_updates) for cid in self.client_updates}
        else:
            weights = {cid: ts / total_trust for cid, ts in self.trust_scores.items()}
        
        aggregated = None
        for client_id, data in self.client_updates.items():
            grad = data['gradient']
            w = weights.get(client_id, 0.0)
            
            if aggregated is None:
                aggregated = grad * w
            else:
                aggregated = aggregated + grad * w
        
        result = aggregated if aggregated is not None else np.zeros(1)
        print(f"[DEBUG][server.py] aggregated gradient norm={np.linalg.norm(result):.4f}")
        return result
    
    def update_global_model(self, gradient: np.ndarray, learning_rate: float = 1.0):
        print(f"[DEBUG][server.py] RFLPAServerSimple.update_global_model: lr={learning_rate}")
        # Apply aggregated update to global model.
        # Note: 'gradient' is actually model update (new_params - old_params),
        # so we ADD it (not subtract).
        idx = 0
        for param in self.global_model.parameters():
            num_params = param.numel()
            if idx + num_params <= len(gradient):
                param_update = gradient[idx:idx + num_params].reshape(param.shape)
                # ADD the update since gradient = new_params - old_params
                param.data += learning_rate * torch.tensor(
                    param_update, dtype=param.dtype, device=param.device
                )
            idx += num_params
        print("[DEBUG][server.py] RFLPAServerSimple model updated")
    
    def reset(self):
        print("[DEBUG][server.py] RFLPAServerSimple.reset called")
        """Reset server state for new round."""
        self.client_updates = {}
        self.trust_scores = {}
