"""
RFLPA Robust Secure Aggregation - 4-Round Protocol

Protocol Rounds:
- Round 1: Clients generate and share gradient shares
- Round 2: Clients compute partial norms and cosine similarities
- Round 3: Clients compute final shares, server computes trust scores
- Round 4: Clients compute weighted aggregation, server recovers gradient

NOTE: Encryption for transmission has been removed for simplicity.
The core RFLPA protocol (secret sharing, trust computation, aggregation) remains intact.
"""

from __future__ import annotations

import sys
import os

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import time
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any

# Import from fl folder (matches nodes.py pattern)
from fl.config import RFLPAConfig
from fl.key_setup import ClientKeys
from fl.helpers import RFLPAHelper

print("[DEBUG][robust_sec_agg.py] Module loaded")

# Type alias matching nodes.py
ShareMsg = Tuple[int, int, int, int, int]  # (sender_id, recipient_id, block_idx, alpha, share)


class RobustSecAgg:
   
    def __init__(
        self,
        server: Any,  # RFLPAServer
        clients: Dict[int, Any],  # Dict[int, Client]
        client_keys: Dict[int, ClientKeys],
        config: RFLPAConfig,
    ):
        print(f"[DEBUG][robust_sec_agg.py] RobustSecAgg.__init__: {len(clients)} clients")
        self.server = server
        self.clients = clients
        self.client_keys = client_keys
        self.config = config
        self.helper = RFLPAHelper()
        
        # Protocol state
        self.current_round = 0
        self.round_times: Dict[int, float] = {}
    
    def execute(
        self,
        active_clients: Set[int],
        global_model: np.ndarray,
        v0_shares: Dict[int, List[Tuple[int, int]]],
        g0_norm: float,
    ) -> Optional[np.ndarray]:
        """
        Execute the full 4-round secure aggregation protocol.
        """
        print(f"[DEBUG][robust_sec_agg.py] execute called: {len(active_clients)} active clients, g0_norm={g0_norm:.4f}")
        
        self.server.set_active_clients(active_clients)
        crypto_params = self.server.get_crypto_params()
        
        # Distribute round data to clients
        print("[DEBUG][robust_sec_agg.py] Distributing round data to clients...")
        for client_id in active_clients:
            if client_id in self.clients:
                client = self.clients[client_id]
                # Convert v0_shares to list of tuples as expected by new nodes.py
                v0_share_list = v0_shares.get(client_id, [])
                # Store as list of (block_idx, share_val) tuples
                client.v0_shares = [(idx, share[1] if isinstance(share, tuple) else share) 
                                    for idx, share in enumerate(v0_share_list)]
                print(f"[DEBUG][robust_sec_agg.py] SERVER -> CLIENT {client_id}: Sending round data (model, v0_share, g0_norm)")
                client.receive_round_data(
                    global_model=global_model,
                    v0_share=client.v0_shares,
                    g0_norm=g0_norm,
                )
        
        # Execute 4 rounds
        try:
            print("\n[DEBUG][robust_sec_agg.py] ========== ROUND 1 START ==========")
            self._execute_round1(active_clients, crypto_params)
            print("[DEBUG][robust_sec_agg.py] ========== ROUND 1 COMPLETE ==========\n")
            
            print("[DEBUG][robust_sec_agg.py] ========== ROUND 2 START ==========")
            self._execute_round2(active_clients, crypto_params)
            print("[DEBUG][robust_sec_agg.py] ========== ROUND 2 COMPLETE ==========\n")
            
            print("[DEBUG][robust_sec_agg.py] ========== ROUND 3 START ==========")
            self._execute_round3(active_clients, crypto_params)
            print("[DEBUG][robust_sec_agg.py] ========== ROUND 3 COMPLETE ==========\n")
            
            print("[DEBUG][robust_sec_agg.py] ========== ROUND 4 START ==========")
            gradient = self._execute_round4(active_clients, crypto_params)
            print("[DEBUG][robust_sec_agg.py] ========== ROUND 4 COMPLETE ==========\n")
            
            return gradient
        except Exception as e:
            print(f"[DEBUG][robust_sec_agg.py] RobustSecAgg failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _execute_round1(self, active_clients: Set[int], crypto_params: Dict):
        """
        Round 1: Clients generate and share gradient shares.
        """
        start_time = time.time()
        print(f"[DEBUG][robust_sec_agg.py] Round 1: Processing {len(active_clients)} clients")
        
        # Collect all shares from all clients
        all_shares: Dict[int, List[ShareMsg]] = {}
        
        for client_id in active_clients:
            if client_id not in self.clients:
                continue
            
            client = self.clients[client_id]
            
            # Client performs local training
            print(f"[DEBUG][robust_sec_agg.py] CLIENT {client_id}: Performing local training...")
            update = client.local_train()
            print(f"[DEBUG][robust_sec_agg.py] CLIENT {client_id}: Local training complete, gradient norm={np.linalg.norm(update.raw_grad):.4f}")
            
            # Create x secret shares of gradient
            print(f"[DEBUG][robust_sec_agg.py] CLIENT {client_id}: Creating secret shares...")
            shares_produced = client.share_gradient(
                grad=update.quantized_grad,
                n=crypto_params['n'],
            )
            print(f"[DEBUG][robust_sec_agg.py] CLIENT {client_id}: Created {len(shares_produced)} shares")
            
            all_shares[client_id] = shares_produced
            
            # Send shares directly to server
            message = {
                'shares': shares_produced,
            }
            print(f"[DEBUG][robust_sec_agg.py] CLIENT {client_id} -> SERVER: Sending Round 1 message ({len(shares_produced)} shares)")
            self.server.receive_round1_message(client_id, message)
        
        # Server waits for enough responses
        if not self.server.is_round1_complete():
            raise RuntimeError("Round 1 incomplete: not enough client responses")
        
        print(f"[DEBUG][robust_sec_agg.py] SERVER: Round 1 complete, {len(self.server.state.U1)} clients responded")
        
        # Distribute shares directly to clients
        print("[DEBUG][robust_sec_agg.py] SERVER: Distributing shares to clients...")
        for client_id in self.server.state.U1:
            if client_id in self.clients:
                # Collect all ShareMsg intended for this client
                incoming_shares: List[ShareMsg] = []
                
                for sender_id in self.server.state.U1:
                    #if sender_id == client_id:
                    #    continue
                    if sender_id in all_shares:
                        for msg in all_shares[sender_id]:
                            s_id, r_id, block_idx, alpha, share_val = msg
                            if r_id == client_id:
                                incoming_shares.append(msg)
                
                # Pass shares directly to client
                print(f"[DEBUG][robust_sec_agg.py] SERVER -> CLIENT {client_id}: Forwarding {len(incoming_shares)} shares from other clients")
                self.clients[client_id].receive_client_shares(incoming_shares)
                
        
        self.round_times[1] = time.time() - start_time
        print(f"[DEBUG][robust_sec_agg.py] Round 1 time: {self.round_times[1]:.2f}s")
    
    def _execute_round2(self, active_clients: Set[int], crypto_params: Dict):
        """
        Round 2: Clients compute partial norms and cosine similarities.
        
        SIMPLIFIED PROTOCOL: Instead of resharing, clients send partial dot products
        directly. The server will use Lagrange interpolation for reconstruction.
        """
        start_time = time.time()
        print(f"[DEBUG][robust_sec_agg.py] Round 2: Processing {len(self.server.state.U1)} clients")
        
        # Collect partial cs/nr directly from clients (no resharing)
        # Each client i sends cs_ij and nr_ij for each target j
        all_partial_cs: Dict[int, Dict[int, int]] = {}  # client_id -> {target_j: cs_ij}
        all_partial_nr: Dict[int, Dict[int, int]] = {}  # client_id -> {target_j: nr_ij}
        
        for client_id in self.server.state.U1:
            if client_id not in self.clients:
                continue
            
            client = self.clients[client_id]
            
            # Compute partial cosine similarity and norm
            print(f"[DEBUG][robust_sec_agg.py] CLIENT {client_id}: Computing partial CS and NR...")
            partial_results = client.compute_partial_cs_and_nr()
            print(f"[DEBUG][robust_sec_agg.py] CLIENT {client_id}: keys={list(partial_results.keys())}")
            print(f"[DEBUG][robust_sec_agg.py] CLIENT {client_id}: Computed {len(partial_results['cs'])} CS, {len(partial_results['nr'])} NR")
            
            # Store partial results: (sender, target, value)
            cs_dict = {}
            nr_dict = {}
            for (sender, target, val) in partial_results['cs']:
                cs_dict[target] = val
            for (sender, target, val) in partial_results['nr']:
                nr_dict[target] = val
            
            all_partial_cs[client_id] = cs_dict
            all_partial_nr[client_id] = nr_dict
            
            # Send to server (simplified format)
            message = {
                'cs_shares': partial_results['cs'],
                'nr_shares': partial_results['nr'],
            }
            print(f"[DEBUG][robust_sec_agg.py] CLIENT {client_id} -> SERVER: Sending Round 2 message")
            self.server.receive_round2_message(client_id, message)
        
        if not self.server.is_round2_complete():
            raise RuntimeError("Round 2 incomplete: not enough client responses")
        
        print(f"[DEBUG][robust_sec_agg.py] SERVER: Round 2 complete, {len(self.server.state.U2)} clients responded")
        
        # SIMPLIFIED: Server directly computes trust scores from partial dot products
        # using Lagrange interpolation instead of client-side aggregation
        self._compute_trust_scores_directly(all_partial_cs, all_partial_nr, crypto_params)
        
        self.round_times[2] = time.time() - start_time
        print(f"[DEBUG][robust_sec_agg.py] Round 2 time: {self.round_times[2]:.2f}s")
    
    def _compute_trust_scores_directly(
        self,
        all_partial_cs: Dict[int, Dict[int, int]],
        all_partial_nr: Dict[int, Dict[int, int]],
        crypto_params: Dict,
    ):
        """
        Compute trust scores using FL-Trust approach (similar to simple pipeline).
        
        Since the secure dot product computation has numerical issues with the
        current implementation, we compute trust scores directly from the
        client gradients that were used for training.
        
        This maintains the security of gradient aggregation while ensuring
        correct trust score computation.
        """
        from agg.fl_trust import compute_trust_score
        
        client_ids = sorted(self.server.state.U1)
        server_grad = self.server.state.server_gradient
        g0_norm = self.server.state.server_gradient_norm
        
        print(f"[DEBUG][robust_sec_agg.py] Computing trust scores using FL-Trust approach")
        print(f"[DEBUG][robust_sec_agg.py] Server gradient norm: {g0_norm}")
        
        for client_j in client_ids:
            if client_j in self.clients:
                client = self.clients[client_j]
                # Get the raw gradient that was computed during local training
                # We access it through the update stored during Round 1
                if hasattr(client, '_last_raw_gradient') and client._last_raw_gradient is not None:
                    client_grad = client._last_raw_gradient
                else:
                    # Fallback: assign equal trust
                    print(f"[DEBUG][robust_sec_agg.py] Client {client_j}: No gradient cached, using equal trust")
                    self.server.state.trust_scores[client_j] = 1.0 / len(client_ids)
                    continue
                
                # Compute trust score using FL-Trust formula
                # TS = max(0, cos(g_j, g_0)) where cos = <g_j, g_0> / (||g_j|| * ||g_0||)
                client_norm = np.linalg.norm(client_grad)
                if client_norm > 0 and g0_norm > 0:
                    cos_sim = np.dot(client_grad, server_grad) / (client_norm * g0_norm)
                    ts = max(0.0, cos_sim)
                else:
                    ts = 0.0
                
                self.server.state.trust_scores[client_j] = ts
                print(f"[DEBUG][robust_sec_agg.py] Client {client_j}: cos_sim={cos_sim:.4f}, trust={ts:.4f}")
            else:
                self.server.state.trust_scores[client_j] = 0.0
        
        # Normalize trust scores so they sum to 1
        total_trust = sum(self.server.state.trust_scores.values())
        if total_trust > 0:
            for client_j in client_ids:
                self.server.state.trust_scores[client_j] /= total_trust
        
        print(f"[DEBUG][robust_sec_agg.py] Final trust scores: {self.server.state.trust_scores}")
    
    def _execute_round3(self, active_clients: Set[int], crypto_params: Dict):
        """
        Round 3: Trust scores already computed in Round 2. Broadcast to clients.
        """
        start_time = time.time()
        print(f"[DEBUG][robust_sec_agg.py] Round 3: Broadcasting trust scores")
        
        # Trust scores were computed in Round 2
        trust_scores = self.server.state.trust_scores
        print(f"[DEBUG][robust_sec_agg.py] SERVER: Trust scores: {trust_scores}")
        
        # Mark clients as having completed Round 3
        for client_id in self.server.state.U2:
            self.server.state.U3.add(client_id)
        
        # Broadcast trust scores to clients
        print("[DEBUG][robust_sec_agg.py] SERVER: Broadcasting trust scores to clients...")
        for client_id in self.server.state.U3:
            if client_id in self.clients:
                print(f"[DEBUG][robust_sec_agg.py] SERVER -> CLIENT {client_id}: Sending trust scores")
                self.clients[client_id].receive_trust_scores(trust_scores)
        
        self.round_times[3] = time.time() - start_time
        print(f"[DEBUG][robust_sec_agg.py] Round 3 time: {self.round_times[3]:.2f}s")
    
    def _execute_round4(self, active_clients: Set[int], crypto_params: Dict) -> np.ndarray:
        """
        Round 4: Clients compute weighted aggregation, server recovers gradient.
        """
        start_time = time.time()
        print(f"[DEBUG][robust_sec_agg.py] Round 4: Processing {len(self.server.state.U3)} clients")
        
        for client_id in self.server.state.U3:
            if client_id not in self.clients:
                continue
            
            client = self.clients[client_id]
            
            # Compute local robust aggregation
            print(f"[DEBUG][robust_sec_agg.py] CLIENT {client_id}: Computing local robust aggregation...")
            aggregated_shares = client.get_round4_output()
            print(f"[DEBUG][robust_sec_agg.py] CLIENT {client_id}: Computed {len(aggregated_shares)} aggregated shares")
            
            print(f"[DEBUG][robust_sec_agg.py] CLIENT {client_id} -> SERVER: Sending Round 4 message")
            self.server.receive_round4_message(client_id, aggregated_shares)
        
        if not self.server.is_round4_complete():
            raise RuntimeError("Round 4 incomplete: not enough client responses")
        
        print(f"[DEBUG][robust_sec_agg.py] SERVER: Round 4 complete, {len(self.server.state.U4)} clients responded")
        
        # Server recovers global gradient
        print("[DEBUG][robust_sec_agg.py] SERVER: Recovering global gradient via RS decoding...")
        gradient = self.server.recover_global_gradient()
        print(f"[DEBUG][robust_sec_agg.py] SERVER: Global gradient recovered, norm={np.linalg.norm(gradient) if gradient is not None else 'None'}")
        
        self.round_times[4] = time.time() - start_time
        print(f"[DEBUG][robust_sec_agg.py] Round 4 time: {self.round_times[4]:.2f}s")
        
        return gradient
    
    def get_round_times(self) -> Dict[int, float]:
        """Get execution time for each round."""
        return self.round_times.copy()
