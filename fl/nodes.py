from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, List, Tuple
import numpy as np

from agg.fl_trust import normalize_gradient, quantize_array, compute_trust_score
from crypto.packed_shamir import pack_and_share, reconstruct_packed, _P
from crypto.matrix_op import _mat_inv_mod, _build_B, _chop_matrix
from fl.helpers import RFLPAHelper

print("[DEBUG][nodes.py] Module loaded")

Array = np.ndarray

# Type aliases for clarity
LocalTrainFn = Callable[[Array, Any], Array]
"""
Local training function for a client.

Args:
    local_dataset: the client's local dataset D_i (any type you like).

Returns:
    A 1-D numpy array representing the local gradient g_i computed on D_i,
    aligned in shape with `global_model` (same length).
"""

AggFn = Callable[[Sequence[Array], Dict[str, Any]], Array]
"""
Aggregation function, e.g. your FLTrust / RFLPA aggregation.

Signature:
    agg_fn(list_of_client_updates, context) -> global_gradient

`context` can contain:
    - "server_update": g0
    - "g0_norm": ||g0||
    - anything else needed by your robust SecAgg logic
"""

#(sender_id, recipient_id, block_idx, alpha, share )
ShareMsg = tuple[int, int, int, int, int]

# Output messages for partial dot-products
DPMsg = tuple[int, int, int]  # (i, j, cs_ij)

# we can use this for debugging purposes.
@dataclass
class ClientUpdate:
    raw_grad: np.ndarray
    normalized_grad: np.ndarray
    quantized_grad: np.ndarray

@dataclass
class Client:

#---------------------------------------INITIALIZATION-----------------------------------------------------------
   
    client_id: int
    dataset: Any 
    local_train_fn: LocalTrainFn

    #------ exchange specifc parameteres ------#

    l: int # TODO add check in constructor that l=e_points=p AND that l<=d AND p<=d
    p: int 
    d: int # degree bound for the Shamir polynomial.
    e_points: List[int] # points where secrets are enforced: φ(e_i) = s_i
    e_points_p: List[int] 
    alpha_points: List[int] # points where shares are evaluated: φ(α_j)
    alpha: int # this client's alpha point
    
    #------------------------------------------#
    recipients_ids: List[int]
    ### TODO still have to figure out how inititalize the alpha point correctly to ensure proper mapping to the recipients ###

    # per-round state received from server
    _current_global_model: Optional[Array] = field(default=None, init=False, repr=False)
    _current_v0_share: Optional[Any] = field(default=None, init=False, repr=False)
    v0_shares: list[tuple[int, int]]
    _current_g0_norm: Optional[float] = field(default=None, init=False, repr=False)

    # per-round shares received FROM other clients
    # key: sender_client_id  -> list of (alpha, share_val) over all blocks
    _incoming_shares: Dict[int, list[tuple[int, int]]] = field(
        default_factory=dict, init=False, repr=False
    )
    inbox_shares: list[ShareMsg]

    # per-round partial dot-product shares (cs_ij, nr_ij) that THIS client i holds
    partial_cs_nr: Dict[int, Dict[str, int]]
    
    # second-level secret shares we receive from other clients (re-shared cs/nr)
    inbox_cs: list[ShareMsg]
    inbox_nr: list[ShareMsg]

    _trust_scores: Optional[Dict[int, float]] = None
    _last_raw_gradient: Optional[np.ndarray] = None  # Cache for trust score computation
    helper = RFLPAHelper()

    def __post_init__(self):
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Created with l={self.l}, d={self.d}, p={self.p}")

#---------------------------------------COMMUNICATIONS-----------------------------------------------------------  
   
    # ---- per-round interaction with server and other clients ---- #
    def receive_round_data(     
        self,
        global_model: Array,
        v0_share: Any,
        g0_norm: float,
    ) -> None:
        """
        Called by the server at the beginning of each round.

        Stores:
        - global model w_{t-1}
        - this client's packed secret share v0^{(i)} of g0
        - ||g0||
        """
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id} <- SERVER: Receiving round data (model shape={global_model.shape}, g0_norm={g0_norm:.4f})")
        self._current_global_model = np.copy(global_model)
        self._current_v0_share = v0_share
        self._current_g0_norm = float(g0_norm)
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Round data stored")

    # ---- client i recieves shares computed gradients ---- #
    ### TODO add check for if all recipients have sent their shares ###
    def receive_client_shares(
        self,
        shares: list[ShareMsg],
    ) -> None:
        """
        Store shares sent to this client by another client.

        Parameters
        ----------
        from_client_id:
            ID of the sending client.
        shares_for_me:
            List of (alpha_j, share_val) pairs for all blocks of that client's gradient.
        """
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id} <- SERVER: Receiving {len(shares)} gradient shares from other clients")
        self.inbox_shares = shares
        # Count unique senders
        senders = set(msg[0] for msg in shares)
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Received {len(self.inbox_shares)} shares")
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Received shares from {len(senders)} senders")

    # ---- client i recieves re-shares of cs and nr  ---- #    
    ### TODO add check for if all recipients have sent their shares ###
    def receive_re_shares(
        self,
        shares_cs: list[ShareMsg],
        shares_nr: list[ShareMsg],
    ) -> None:
        """
        Store re-shared cs/nr sent to this client.

        """
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id} <- SERVER: Receiving reshared CS ({len(shares_cs)}) and NR ({len(shares_nr)}) shares")
        self.inbox_cs = shares_cs
        self.inbox_nr = shares_nr

#---------------------------------------COMPUTATIONS-----------------------------------------------------------
    
    # ---- compute the normalized and quantized updated gi ---- #
    def local_train(self) -> ClientUpdate:
        """
        Run local training using the most recent global model.

        Returns:
            Instance of Client Update: The raw actual update, the normalized update and the quantized update.
        """
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Starting local training...")
        if self._current_global_model is None:
            raise RuntimeError("Client has not received global model for this round.")

        if self._current_g0_norm is None:
            raise RuntimeError("Client has not received g0_norm from server for this round.")

        # Compute
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Computing gradient via local_train_fn...")
        grad = self.local_train_fn(self._current_global_model, self.dataset) #placeholder train function that returns the gradients
        grad = np.asarray(grad, dtype=np.float64)
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Raw gradient computed, shape={grad.shape}, norm={np.linalg.norm(grad):.4f}")
        
        # Cache raw gradient for trust score computation
        self._last_raw_gradient = grad.copy()

        # Normalize
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Normalizing gradient with g0_norm={self._current_g0_norm:.4f}")
        gbar_i = normalize_gradient(grad, self._current_g0_norm)
        
        # Quantize
        q = 1000
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Quantizing gradient with q={q}")
        q_gbar_i = quantize_array(gbar_i, q=q)
        
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Local training complete")
        return ClientUpdate(
            raw_grad=grad,
            normalized_grad=gbar_i,
            quantized_grad=q_gbar_i,
        )

    # ---- prepare the shares of the packed secret of the computed gradient ---- #
    def share_gradient(
        self,
        grad: Array,
        n: int
    )-> list[ShareMsg]:
        """
        Convert the client's final gradient into integer secrets and
        use the Shamir pack_and_share function to produce shares
        for all n parties.

        Parameters
        ----------
        grad : np.ndarray
            The client's final local gradient (already normalized + quantized).
        n : int
            Number of parties (clients) to create shares for.

        Returns
        -------
        shares_produced : list[ShareMsg]
        Flat list of share messages. Each recipient will later filter messages
        intended for it and group them by block_idx and sender_id.
        """
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Creating gradient shares for {n} parties...")
        
        grad_arr = np.asarray(grad, dtype=np.float64).ravel()

        # --- FIX: paper-style quantization produces multiples of 1/q ---
        q = 1000
        if q <= 0:
            raise ValueError("self.q must be a positive integer")

        if np.issubdtype(grad_arr.dtype, np.integer):
            # Already integer-coded (i.e., already multiplied by q somewhere upstream)
            grad_q = grad_arr.astype(np.int64)
        else:
            # Expect values on 1/q grid: q * grad_arr should be (almost) integers
            scaled = grad_arr * q
            if not np.all(np.isclose(scaled, np.rint(scaled))):
                raise ValueError(
                    "grad must be quantized to the 1/q grid before sharing "
                    "(i.e., q*grad must be integer-valued)."
                )
            # Integer-coded secrets to share: z = round(q * Q(ḡ))
            grad_q = np.rint(scaled).astype(np.int64)

        secrets = [int(x) % _P for x in grad_q.tolist()]
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: {len(secrets)} secrets to pack")

        # number of blocks of size l
        blocks = [
            secrets[i:i+self.l] for i in range(0, len(secrets), self.l)
        ]

        # pad last block if needed
        if len(blocks[-1]) < self.l:
            blocks[-1] += [0] * (self.l - len(blocks[-1]))

        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Created {len(blocks)} blocks of size {self.l}")

        # initialize output
        shares_produced: List[ShareMsg] = []

        for block_idx, block in enumerate(blocks):
            block_shares = pack_and_share(
                secrets=block,
                n=n,
                d=self.d,
                e_points=self.e_points,
                alpha_points=self.alpha_points,
            ) 

            for recipient_id, (alpha, share_val) in zip(self.recipients_ids, block_shares):
                msg = (self.client_id, recipient_id, block_idx, alpha, share_val)
                shares_produced.append(msg)

        print(f"[DEBUG][nodes.py] CLIENT {self.client_id} -> SERVER: Sending {len(shares_produced)} gradient shares")
        return shares_produced

    # ---- computing the corresponding cosine similarity and norms ---- #
    def compute_partial_cs_and_nr(self) -> Dict[str, List[DPMsg]]:
        """
        Compute this client's i-th shares of partial cosine similarity (cs^i_j)
        and partial gradient norm square (nr^i_j) for every other client j,
        using ONLY shares (Eq. 8):

            cs^i_j = sum_b v^j_{i,b} * v^0_{i,b}
            nr^i_j = sum_b v^j_{i,b} * v^j_{i,b}

        Inputs expected on self:
        - self.v0_shares: list[tuple[int,int]]  # [(block_idx, v0_share_val), ...]
        - self.inbox_shares: list[ShareMsg]     # incoming gradient shares from other clients

        Returns
        -------
        out : Dict[str, List[DPMsg]]
            {
            "cs": [(i, j, cs_ij), ...],
            "nr": [(i, j, nr_ij), ...]
            }

        Side effects
        ------------
        Populates self._partial_cs_nr[j] = {"cs_share": cs_ij, "nr_share": nr_ij}
        for the later resharing step.
        """
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Computing partial CS and NR...")

        if self.v0_shares is None:
            raise RuntimeError("Server share v0 has not been set for this client.")

        client_id = self.client_id
        v0_pairs = list(self.v0_shares)  # copy
        v0_pairs.sort(key=lambda t: t[0])

        v0_vals = [int(share) % _P for (_, share) in v0_pairs]
        num_blocks = len(v0_vals)
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: {num_blocks} v0 blocks")

        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: inbox_shares={len(self.inbox_shares)}")

        grad_by_sender: Dict[int, Dict[int, int]] = {} # this will create a dictionary of the shares sent by each user, each in a tuple (block_idx,share)

        for sender_id, recipient_id, block_idx, alpha, share_val in self.inbox_shares:
            if int(recipient_id) != client_id:
                continue  # not for me
            j = sender_id
            b = block_idx
            if b < 0 or b >= num_blocks:
                continue
            grad_by_sender.setdefault(j, {})[b] = int(share_val) % _P
        
        if not grad_by_sender:
            raise RuntimeError("No incoming gradient shares found from this client.")
        
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Processing shares from {len(grad_by_sender)} senders")
            
        out_cs: List[DPMsg] = []
        out_nr: List[DPMsg] = []
        partial_cache: Dict[int, Dict[str, int]] = {}

        for j, vj_map in grad_by_sender.items():
            # Require that we have all blocks for j
            if len(vj_map) != num_blocks:
                missing = [b for b in range(num_blocks) if b not in vj_map]
                raise ValueError(
                    f"Missing gradient-share blocks from client {j} to client {client_id}: {missing}"
                )

            cs_ij = 0
            nr_ij = 0
            for b in range(num_blocks):
                s_j = vj_map[b]
                s_0 = v0_vals[b]
                cs_ij = (cs_ij + s_j * s_0) % _P
                nr_ij = (nr_ij + s_j * s_j) % _P

            partial_cache[j] = {"cs_share": cs_ij, "nr_share": nr_ij}

            # Emit compact messages: (sender=i, receiver=j, share)
            out_cs.append((client_id, j, cs_ij))
            out_nr.append((client_id, j, nr_ij))

        # cache for re-sharing step (your existing downstream expects this)
        self.partial_cs_nr = partial_cache
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Computed {len(out_cs)} CS and {len(out_nr)} NR partial shares")

        return {"cs": out_cs, "nr": out_nr}

    # ---- computing the shares to be reshared of cs and nr ---- #
    def reshare_partial_cs_and_nr(
        self,
        n: int,
    ) -> Dict[str, list[ShareMsg]]:
        """
        Re-share this client's partial dot-product shares (cs_ij, nr_ij) to all recipients.
        
        IMPORTANT: Each target client j gets its own block_idx to ensure proper 
        reconstruction on the server side. We use block size 1 (not packed) so that
        each block corresponds to exactly one target client.
        
        Produces ShareMsg tuples:
            (sender_id, recipient_id, block_idx, alpha, share_val)
        
        where block_idx corresponds to the target client index.

        Returns
        -------
            out : Dict[str, List[ShareMsg]]
                {
                "cs": [ShareMsg, ShareMsg, ...],
                "nr": [ShareMsg, ShareMsg, ...],
                }
        """
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Resharing partial CS and NR...")
        
        if not self.partial_cs_nr:
            raise RuntimeError(
                "Partial cs/nr not computed yet. Call compute_partial_cs_and_nr() first."
            )
       
        out_cs: List[ShareMsg] = []
        out_nr: List[ShareMsg] = []

        # Share each target client's value separately (block size = 1)
        # This ensures block_idx maps directly to target client index
        for target_idx, j in enumerate(self.recipients_ids):
            entry = self.partial_cs_nr[j]
            cs_ij = int(entry["cs_share"]) % _P
            nr_ij = int(entry["nr_share"]) % _P
            
            # Create shares for cs_ij (single value, use e_points with just 1 point)
            cs_shares = pack_and_share(
                secrets=[cs_ij],
                n=n,
                d=self.d,
                e_points=self.e_points,  # Use e_points (size l=1) for single value
                alpha_points=self.alpha_points,
            )
            for recipient_id, (alpha, share_val) in zip(self.recipients_ids, cs_shares):
                msg = (self.client_id, recipient_id, target_idx, alpha, share_val)
                out_cs.append(msg)
            
            # Create shares for nr_ij (single value)
            nr_shares = pack_and_share(
                secrets=[nr_ij],
                n=n,
                d=self.d,
                e_points=self.e_points,  # Use e_points (size l=1) for single value
                alpha_points=self.alpha_points,
            )
            for recipient_id, (alpha, share_val) in zip(self.recipients_ids, nr_shares):
                msg = (self.client_id, recipient_id, target_idx, alpha, share_val)
                out_nr.append(msg)

        print(f"[DEBUG][nodes.py] CLIENT {self.client_id} -> SERVER: Sending {len(out_cs)} CS shares, {len(out_nr)} NR shares")
        return {
            "cs": out_cs,
            "nr": out_nr,
        }


    """
    RFLPA Client Implementation - Rounds 3 & 4

    This module implements the client-side logic for Rounds 3 and 4 of the RFLPA protocol
    as described in the paper.

    - Round 3: Dot Product Aggregation (Section 4.5, Eq. 10, 11, Reed-Solomon decoding)
    - Round 4: Local Robust Aggregation (Section 4.3, Eq. 6)
    """
    # =========================================================================
    # ROUND 3: DOT PRODUCT AGGREGATION (CLIENT SIDE)
    # Paper Section 4.5, Algorithm 3 Round 3
    # =========================================================================
    
    def receive_reshared_partial_dot_products(
        self,
        cs_shares: list[ShareMsg],
        nr_shares: list[ShareMsg],
    ) -> None:
        """
        Receive re-shared partial dot product shares from another client.
        
        From Algorithm 3, Round 3:
            "Receive (C || {c'_{ji}}_{j∈U2\\i} || {σ'_{ji}}_{j∈U2\\i}) from server"
        
        This method stores the decrypted re-shared values (after verification
        by collaborator's code).
        
        Parameters
        ----------
    
        cs_shares : list[ShareMsg]
            Re-shared cosine similarity shares for each group k
        nr_shares : list[ShareMsg]
            Re-shared norm square shares for each group k
        """
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id} <- SERVER: Receiving reshared dot products (CS={len(cs_shares)}, NR={len(nr_shares)})")
        self.inbox_cs = cs_shares
        self.inbox_nr = nr_shares
    
    
    def round3_dot_product_aggregation(
        self,
        n: int,
        d: int,
        l: int,
        alpha_points: Sequence[int],
        e_points: Sequence[int],
        secret_point: int,
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Perform Round 3 client-side computation: Aggregate reshared partial dot products.
        
        According to Algorithm 3 (Round 3) in the paper:
        - Each client i receives reshared CS/NR values from other clients
        - For each target client j (group), client i sums the shares it received
        - Client i sends (group_idx, aggregated_share) to the server
        - The server will use Lagrange interpolation to recover the final CS_j and NR_j values
        
        Parameters
        ----------
        n : int
            Number of parties
        d : int
            Polynomial degree
        l : int
            Packing factor
        alpha_points : Sequence[int]
            Share evaluation points [α_1, ..., α_n]
        e_points : Sequence[int]
            Secret encoding points [e_1, ..., e_l]
        secret_point : int
            Point where secrets are encoded (for server-side reconstruction)
        
        Returns
        -------
        final_shares : Dict[str, List[Tuple[int, int]]]
            {
                "cs_final": [(group_idx, aggregated_cs_share), ...],
                "nr_final": [(group_idx, aggregated_nr_share), ...]
            }
            These are sent to the server for reconstruction.
        """
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Starting Round 3 aggregation...")
        
        # Check that we have received reshared values
        if not self.inbox_cs or not self.inbox_nr:
            raise RuntimeError("No re-shared partial dot products received.")

        # Get this client's alpha point for identification
        my_alpha = self.alpha
        
        # =====================================================================
        # Process CS shares: Group by target (block_idx), sum contributions
        # =====================================================================
        cs_by_group: Dict[int, int] = defaultdict(int)
        
        for sender_id, recipient_id, block_idx, alpha, share_val in self.inbox_cs:
            if int(recipient_id) != int(self.client_id):
                continue
            # Sum all shares for this group (target client)
            cs_by_group[int(block_idx)] = (cs_by_group[int(block_idx)] + int(share_val)) % _P
        
        # =====================================================================
        # Process NR shares: Group by target (block_idx), sum contributions
        # =====================================================================
        nr_by_group: Dict[int, int] = defaultdict(int)
        
        for sender_id, recipient_id, block_idx, alpha, share_val in self.inbox_nr:
            if int(recipient_id) != int(self.client_id):
                continue
            # Sum all shares for this group (target client)
            nr_by_group[int(block_idx)] = (nr_by_group[int(block_idx)] + int(share_val)) % _P
        
        # =====================================================================
        # Format output: (group_idx, aggregated_share_value)
        # The server will collect these from all clients and use the alpha points
        # for Lagrange interpolation
        # =====================================================================
        cs_final: List[Tuple[int, int]] = []
        for group_idx in sorted(cs_by_group.keys()):
            cs_final.append((group_idx, cs_by_group[group_idx]))
        
        nr_final: List[Tuple[int, int]] = []
        for group_idx in sorted(nr_by_group.keys()):
            nr_final.append((group_idx, nr_by_group[group_idx]))
        
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Aggregated {len(cs_final)} CS groups, {len(nr_final)} NR groups")
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id} -> SERVER: Sending Round 3 output (CS={len(cs_final)}, NR={len(nr_final)})")
        
        return {
            "cs_final": cs_final,
            "nr_final": nr_final,
        }

    
    def get_round3_output(
        self,
        n: int,
        d: int,
        l: int,
        alpha_points: Sequence[int],
        e_points: Sequence[int],
        secret_point: int,
    ) -> Dict[str, List[Tuple[int, int]]]:
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: get_round3_output called")
        return self.round3_dot_product_aggregation(
            n=n,
            d=d,
            l=l,
            alpha_points=alpha_points,
            e_points=e_points,
            secret_point=secret_point,
        )

    # =========================================================================
    # ROUND 4: LOCAL ROBUST AGGREGATION (CLIENT SIDE)
    # Paper Section 4.3, Eq. 6, Algorithm 3 Round 4
    # =========================================================================
    
    def receive_trust_scores(self, trust_scores: Dict[int, float]) -> None:
        """
        Receive trust scores from server.
        
        From Algorithm 3, Round 4:
            Server "broadcasts the trust score {TS_j}_{j∈U1} to all users i ∈ U3"
        
        Parameters
        ----------
        trust_scores : Dict[int, float]
            Mapping from client_id to trust score TS_j
            Trust scores are computed by server using Eq. (5):
            TS_j = max(0, ⟨ḡ_j, g0⟩ / ||g0||²)
        """
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id} <- SERVER: Receiving trust scores for {len(trust_scores)} clients")
        self._trust_scores = dict(trust_scores)
        my_score = trust_scores.get(self.client_id, 0.0)
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: My trust score = {my_score:.4f}")


    def round4_local_robust_aggregation(self) -> List[Tuple[int, int]]:
        """
        Perform Round 4 client-side computation: Local Robust Aggregation.
        
        From Algorithm 3, Round 4:
            "Compute local aggregation ⟨g⟩_i from (6)"
        
        The global gradient aggregation formula (Eq. 6) is:
            g = (1 / Σ_{i=1}^{N} TS_i) · Σ_{i=1}^{N} TS_i · ḡ_i
        
        Applied locally to this client's shares:
            ⟨g⟩_i = (1 / Σ_j TS_j) · Σ_j TS_j · v^j_i
        
        where v^j_i is this client's share of client j's normalized gradient.
        
        The computation is performed in the finite field F_p.
        Trust scores are scaled to integers to enable modular arithmetic.
        
        Returns
        -------
        aggregated_shares : List[Tuple[int, int]]
            [(block_idx, aggregated_share_value), ...]
            This represents ⟨g⟩_i to be sent to the server.
        """
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Starting Round 4 local robust aggregation...")
        
        if self._trust_scores is None:
            raise RuntimeError("Trust scores not received from server.")

        # Use canonical storage already populated for gradient shares
        if not self.inbox_shares:
            raise RuntimeError("No gradient shares received from other clients.")

        # =====================================================================
        # Step 1: Filter to only clients with positive trust scores
        # From Eq. (5): TS_i = max(0, cos_sim), so negative ones are already 0
        # =====================================================================
        active_clients = {
            cid: ts for cid, ts in self._trust_scores.items() if ts > 0
        }
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: {len(active_clients)} clients with positive trust scores")

        if not active_clients:
            raise ValueError("All trust scores are zero or negative.")
        # =====================================================================
        # Step 2: Compute total trust score Σ TS_j
        # =====================================================================
        total_trust = sum(active_clients.values())
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Total trust = {total_trust:.4f}")

        if total_trust <= 0:
            raise ValueError("Total trust score is zero.")
        
        # =====================================================================
        # Step 3: Scale trust scores to integers for finite field arithmetic
        # We use a precision factor to maintain accuracy
        # IMPORTANT: We use WEIGHT_PRECISION = 10000 to keep values small enough
        # to avoid overflow when multiplied with quantized gradients (q=1000)
        # The server will divide by WEIGHT_PRECISION after reconstruction
        # =====================================================================
        WEIGHT_PRECISION = 10000

        # Normalize trust scores so they sum exactly to WEIGHT_PRECISION
        # This avoids issues with modular arithmetic
        ts_scaled: Dict[int, int] = {}
        remaining = WEIGHT_PRECISION
        sorted_clients = sorted(active_clients.items(), key=lambda x: x[1], reverse=True)
        
        for i, (cid, ts) in enumerate(sorted_clients):
            if i == len(sorted_clients) - 1:
                # Last client gets the remainder to ensure exact sum
                ts_scaled[cid] = remaining
            else:
                scaled_val = int(round(ts * WEIGHT_PRECISION))
                ts_scaled[cid] = scaled_val
                remaining -= scaled_val
        
        total_trust_scaled = sum(ts_scaled.values())
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Weights sum = {total_trust_scaled}")

        if total_trust_scaled == 0:
            raise ValueError("Scaled total trust is zero.")
        # =====================================================================
        # Step 4: Determine number of blocks from any client's shares
        # =====================================================================

        # Determine number of blocks from messages addressed to this client
        my_msgs = [m for m in self.inbox_shares if int(m[1]) == int(self.client_id)]
        if not my_msgs:
            raise RuntimeError("No gradient shares received from other clients.")
        num_blocks = max(int(m[2]) for m in my_msgs) + 1
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Aggregating {num_blocks} blocks")

        # =====================================================================
        # Step 5: Compute weighted sum Σ_j TS_j · v^j_i for each block
        # =====================================================================
        weighted_sum = [0] * num_blocks

        # Aggregate directly from inbox_shares (ShareMsg) to avoid duplicate structures
        for sender_id, recipient_id, block_idx, alpha, share_val in self.inbox_shares:
            if int(recipient_id) != int(self.client_id):
                continue

            ts_j = ts_scaled.get(int(sender_id), 0)

            if ts_j <= 0:
                continue

            if int(block_idx) >= num_blocks:
                break

            contribution = (ts_j * int(share_val)) % _P
            weighted_sum[int(block_idx)] = (weighted_sum[int(block_idx)] + contribution) % _P

        # =====================================================================
        # Step 6: Send weighted sum directly (NO modular division)
        # The server will divide by WEIGHT_PRECISION after reconstruction
        # This avoids issues with modular arithmetic not matching real division
        # =====================================================================
        # NOTE: We do NOT divide by total_trust_scaled here because modular
        # inverse division in finite fields doesn't work like real division
        # when the numerator isn't exactly divisible by the denominator.
        # Instead, we send the raw weighted sum: Σ (ts_scaled[j] * v^j_i)
        # The server will divide by WEIGHT_PRECISION (=10000) after recovery.

        aggregated_shares: List[Tuple[int, int]] = []
        for block_idx in range(num_blocks):
            # Send raw weighted sum - server divides by WEIGHT_PRECISION
            final_share = weighted_sum[block_idx] % _P
            aggregated_shares.append((block_idx, final_share))

        print(f"[DEBUG][nodes.py] CLIENT {self.client_id} -> SERVER: Sending {len(aggregated_shares)} aggregated shares")
        return aggregated_shares


    def get_round4_output(self) -> List[Tuple[int, int]]:
        """
        Execute Round 4 and return output to send to server.
        
        From Algorithm 3, Round 4:
            "Compute local aggregation ⟨g⟩_i from (6), and send to the server"
        
        Returns
        -------
        aggregated_shares : List[Tuple[int, int]]
            The aggregated gradient shares ⟨g⟩_i to be sent to server
        """
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: get_round4_output called")
        return self.round4_local_robust_aggregation()


    def clear_round_state(self) -> None:
        """
        Clear all per-round state to prepare for next iteration.
        """
        print(f"[DEBUG][nodes.py] CLIENT {self.client_id}: Clearing round state")
        # Clear canonical inboxes; do not reference duplicate storages.
        self.inbox_shares.clear()
        self.inbox_cs.clear()
        self.inbox_nr.clear()
        self._trust_scores = None

    
    @property
    def v0_share(self) -> Any:
        return self._current_v0_share

    @property
    def g0_norm(self) -> Optional[float]:
        return self._current_g0_norm

    @property
    def trust_scores(self) -> Optional[Dict[int, float]]:
        """Return received trust scores."""
        return self._trust_scores
    
    @property
    def incoming_gradient_shares(self) -> list[ShareMsg]:
        """Return gradient shares received from other clients."""
        return self.inbox_shares