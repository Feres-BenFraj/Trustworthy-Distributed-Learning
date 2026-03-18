#RFLPA Server - Round 4 Logic
#Collect aggregated shares and recover global gradient.

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from fl.utils_server import ServerState, ServerCryptoUtils

print("[DEBUG][round4_server.py] Module loaded")


class Round4Handler:
    
    #Round 4: Collect aggregated shares and recover global gradient.
    
    #Message from client contains:
    #    - aggregated_shares: <g>_i from Eq. (6) - client's local aggregation
    #      This comes from client's get_round4_output() in nodes.py
    
    
    def __init__(self, state: ServerState, crypto_utils: ServerCryptoUtils, args: Any):
        print("[DEBUG][round4_server.py] Round4Handler initialized")
        self.state = state
        self.crypto_utils = crypto_utils
        self.args = args
    
    def receive_message(self, client_id: int, aggregated_shares: List[Tuple[int, int]]) -> bool:
        
        print(f"[DEBUG][round4_server.py] receive_message called: client_id={client_id}, shares_count={len(aggregated_shares)}")
        
        if client_id not in self.state.U3:
            print(f"[DEBUG][round4_server.py] client_id={client_id} not in U3, rejecting")
            return False
        
        self.state.round4_messages[client_id] = aggregated_shares
        self.state.U4.add(client_id)
        print(f"[DEBUG][round4_server.py] client_id={client_id} accepted, U4 size={len(self.state.U4)}")
        return True
    
    def is_complete(self) -> bool:
        complete = len(self.state.U4) >= self.state.min_clients
        print(f"[DEBUG][round4_server.py] is_complete: {complete} (U4={len(self.state.U4)}, min={self.state.min_clients})")
        return complete
    
    def recover_global_gradient(self) -> Optional[np.ndarray]:
        
        print("[DEBUG][round4_server.py] recover_global_gradient called")
        
        if not self.is_complete():
            print("[DEBUG][round4_server.py] not enough responses, returning None")
            return None
        
        p = self.state.prime
        
        # Determine number of blocks from first message
        first_client = next(iter(self.state.round4_messages.keys()))
        num_blocks = len(self.state.round4_messages[first_client])
        print(f"[DEBUG][round4_server.py] num_blocks={num_blocks}, prime={p}")
        
        # Recover each block
        recovered_blocks = []
        
        for block_idx in range(num_blocks):
            # Collect shares for this block
            # Use alpha_points for proper Lagrange interpolation
            shares = []
            for client_id, agg_shares in self.state.round4_messages.items():
                if block_idx < len(agg_shares):
                    _, share_val = agg_shares[block_idx]
                    # Use the client's alpha point for proper RS decoding
                    alpha_point = self.state.alpha_points[client_id] if client_id < len(self.state.alpha_points) else client_id + 1
                    shares.append((alpha_point, int(share_val) % p))
            
            if len(shares) >= self.state.min_clients:
                try:
                    # Decode using reed_solomon_decode from RFLPAHelper (same as nodes.py)
                    block_value = self.crypto_utils.helper.reed_solomon_decode(
                        shares=shares,
                        secret_point=self.state.secret_point,
                        p=p,
                    )
                    # Handle signed values
                    if block_value > p // 2:
                        block_value = block_value - p
                    recovered_blocks.append(float(block_value))
                except Exception as e:
                    print(f"[DEBUG][round4_server.py] block {block_idx} decode failed: {e}")
                    recovered_blocks.append(0.0)
            else:
                print(f"[DEBUG][round4_server.py] block {block_idx} insufficient shares")
                recovered_blocks.append(0.0)
        
        # Convert to numpy array and dequantize
        # The recovered values are: Σ (ts_scaled[j] * quantized_grad[j])
        # where ts_scaled sums to WEIGHT_PRECISION (10000) and quantized values
        # are multiplied by q (1000). So we divide by both.
        q = getattr(self.args, 'precision', 1000)
        WEIGHT_PRECISION = 10000  # Must match nodes.py
        
        gradient = np.array(recovered_blocks, dtype=np.float64) / (q * WEIGHT_PRECISION)
        
        print(f"[DEBUG][round4_server.py] recovered gradient shape={gradient.shape}, norm={np.linalg.norm(gradient):.4f}")
        return gradient
    
    def get_respondent_count(self) -> int:
        count = len(self.state.U4)
        print(f"[DEBUG][round4_server.py] get_respondent_count: {count}")
        return count
    
    def get_respondents(self) -> set:
        print(f"[DEBUG][round4_server.py] get_respondents: {self.state.U4}")
        return self.state.U4.copy()
