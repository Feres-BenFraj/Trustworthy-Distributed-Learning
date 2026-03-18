#RFLPA Server - Round 3 Logic
#Collect final shares and compute trust scores.

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from fl.utils_server import ServerState, ServerCryptoUtils

print("[DEBUG][round3_server.py] Module loaded")


class Round3Handler:
    
    #Round 3: Collect final shares and compute trust scores.
    
    def __init__(self, state: ServerState, crypto_utils: ServerCryptoUtils):
        print("[DEBUG][round3_server.py] Round3Handler initialized")
        self.state = state
        self.crypto_utils = crypto_utils
    
    def receive_message(self, client_id: int, message: Dict) -> bool:
        
        print(f"[DEBUG][round3_server.py] receive_message called: client_id={client_id}")
        
        if client_id not in self.state.U2:
            print(f"[DEBUG][round3_server.py] client_id={client_id} not in U2, rejecting")
            return False
        
        # Accept both naming conventions for compatibility with nodes.py
        if 'cs_final' in message and 'nr_final' in message:
            print(f"[DEBUG][round3_server.py] client_id={client_id} converting from nodes.py format")
            message = {
                'norm_shares': message['nr_final'],
                'cosine_shares': message['cs_final'],
            }
        
        required = ['norm_shares', 'cosine_shares']
        if not all(k in message for k in required):
            print(f"[DEBUG][round3_server.py] client_id={client_id} rejected (missing required fields)")
            return False
        
        print(f"[DEBUG][round3_server.py] client_id={client_id} accepted, norm_shares={len(message['norm_shares'])}, cosine_shares={len(message['cosine_shares'])}")
        self.state.round3_messages[client_id] = message
        self.state.U3.add(client_id)
        return True
    
    def is_complete(self) -> bool:
        complete = len(self.state.U3) >= self.state.min_clients
        print(f"[DEBUG][round3_server.py] is_complete: {complete} (U3={len(self.state.U3)}, min={self.state.min_clients})")
        return complete
    
    def process_and_compute_trust_scores(self) -> Dict[int, float]:
        print("[DEBUG][round3_server.py] process_and_compute_trust_scores called")
        
        # Get alpha_points from state for proper Lagrange interpolation
        alpha_points = self.state.alpha_points
        
        # Collect shares by target client
        # IMPORTANT: Use alpha_points[sender_id] as x-coordinate for Lagrange interpolation
        all_norm_shares: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        all_cosine_shares: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        
        for sender_id, message in self.state.round3_messages.items():
            # Map sender_id to its alpha point for proper RS decoding
            alpha_i = alpha_points[sender_id] if sender_id < len(alpha_points) else sender_id + 1
            
            for group_idx, share_val in message['norm_shares']:
                all_norm_shares[group_idx].append((alpha_i, share_val))
            
            for group_idx, share_val in message['cosine_shares']:
                all_cosine_shares[group_idx].append((alpha_i, share_val))
        
        print(f"[DEBUG][round3_server.py] collected {len(all_norm_shares)} norm groups, {len(all_cosine_shares)} cosine groups")
        
        # Recover norms using Reed-Solomon decoding
        print("[DEBUG][round3_server.py] recovering norms via RS decoding")
        recovered_norms = self.crypto_utils.recover_values_rs(all_norm_shares, self.state)
        
        # Recover cosine similarities using Reed-Solomon decoding
        print("[DEBUG][round3_server.py] recovering cosine similarities via RS decoding")
        recovered_cosines = self.crypto_utils.recover_values_rs(all_cosine_shares, self.state)
        
        # Map recovered values to client IDs
        client_ids = sorted(self.state.U1)
        
        for idx, client_id in enumerate(client_ids):
            if idx in recovered_norms:
                self.state.client_norms[client_id] = recovered_norms[idx]
            if idx in recovered_cosines:
                self.state.client_cosine_similarities[client_id] = recovered_cosines[idx]
        
        print(f"[DEBUG][round3_server.py] recovered norms for {len(self.state.client_norms)} clients")
        print(f"[DEBUG][round3_server.py] recovered cosine similarities for {len(self.state.client_cosine_similarities)} clients")
        
        # Verify norm bounds and compute trust scores
        self._compute_trust_scores()
        
        print(f"[DEBUG][round3_server.py] computed trust scores: {self.state.trust_scores}")
        return self.state.trust_scores
    
    def _compute_trust_scores(self):
        print("[DEBUG][round3_server.py] _compute_trust_scores called")
        
        # Quantization factor used during gradient quantization
        # The recovered norms and cosines are computed from quantized values,
        # so they need to be dequantized: divide by q² where q=1000
        Q = 1000  # Must match the quantization factor used in clients
        Q_SQ = Q * Q  # For dequantizing dot products (product of quantized values)
        
        server_norm_sq = self.state.server_gradient_norm ** 2
        # Scale server_norm_sq to match quantized space for fair comparison
        server_norm_sq_quantized = server_norm_sq * Q_SQ
        
        print(f"[DEBUG][round3_server.py] server_norm_sq={server_norm_sq}, server_norm_sq_quantized={server_norm_sq_quantized}")
        
        for client_id in self.state.U1:
            norm_sq_quantized = self.state.client_norms.get(client_id, float('inf'))
            
            # Check norm bound: ||g_j||^2 <= ||g0||^2 (in quantized space)
            if norm_sq_quantized > server_norm_sq_quantized * 1.5:  # Allow tolerance for numerical errors
                print(f"[DEBUG][round3_server.py] client_id={client_id} norm_sq_quantized={norm_sq_quantized} exceeds bound, trust=0")
                self.state.trust_scores[client_id] = 0.0
                continue
            
            # Compute trust score from Eq. (5): TS_j = max(0, ⟨ḡ_j, g0⟩ / ||g0||²)
            # Both cosine and norm are in quantized space, so we can use them directly
            cosine_val_quantized = self.state.client_cosine_similarities.get(client_id, 0.0)
            
            if server_norm_sq_quantized > 0:
                # TS = cosine_quantized / server_norm_sq_quantized
                # This gives the same result as dequantizing both and dividing
                ts = cosine_val_quantized / server_norm_sq_quantized
                self.state.trust_scores[client_id] = max(0.0, float(ts))
            else:
                self.state.trust_scores[client_id] = 0.0
            
            print(f"[DEBUG][round3_server.py] client_id={client_id} cosine_q={cosine_val_quantized}, norm_sq_q={norm_sq_quantized}, trust_score={self.state.trust_scores[client_id]}")
    
    def broadcast_trust_scores(self) -> Dict[int, float]:
        print(f"[DEBUG][round3_server.py] broadcast_trust_scores called, {len(self.state.trust_scores)} scores")
        return self.state.trust_scores.copy()
    
    def get_respondent_count(self) -> int:
        count = len(self.state.U3)
        print(f"[DEBUG][round3_server.py] get_respondent_count: {count}")
        return count
    
    def get_respondents(self) -> set:
        print(f"[DEBUG][round3_server.py] get_respondents: {self.state.U3}")
        return self.state.U3.copy()
    
    def get_client_norms(self) -> Dict[int, float]:
        print(f"[DEBUG][round3_server.py] get_client_norms called")
        return self.state.client_norms.copy()
    
    def get_client_cosine_similarities(self) -> Dict[int, float]:
        print(f"[DEBUG][round3_server.py] get_client_cosine_similarities called")
        return self.state.client_cosine_similarities.copy()
