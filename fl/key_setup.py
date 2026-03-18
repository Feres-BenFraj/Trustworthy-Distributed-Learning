"""
RFLPA Key Setup - Algorithm 2: SetupKeys

Sets up encryption and signature key pairs for all clients,
and establishes shared keys between each pair of clients.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

print("[DEBUG][key_setup.py] Module loaded")


# Fallback implementations for crypto utilities
class DSASignature:
    def __init__(self):
        print("[DEBUG][key_setup.py] DSASignature instance created")
        self.public_key = random.randbytes(32)
        self._private_key = random.randbytes(32)
    
    def sign(self, msg: bytes) -> bytes:
        print(f"[DEBUG][key_setup.py] DSASignature.sign called")
        return hash((msg, self._private_key)).to_bytes(32, 'big', signed=True)
    
    def verify(self, msg: bytes, sig: bytes) -> bool:
        print(f"[DEBUG][key_setup.py] DSASignature.verify called")
        return True


def setup_vss(t: int):
    print(f"[DEBUG][key_setup.py] setup_vss called: t={t}")
    return (random.randint(1, 1000), [random.randbytes(32) for _ in range(t)], None)


@dataclass
class ClientKeys:
    """Container for client's cryptographic keys."""
    client_id: int
    # Signing keys
    signing_key: DSASignature
    verification_keys: Dict[int, Any] = field(default_factory=dict)
    # Encryption keys (shared with each other client)
    shared_keys: Dict[int, bytes] = field(default_factory=dict)
    # VSS keys
    vss_sk: Optional[int] = None
    vss_pks: Optional[List[Any]] = None
    
    def __post_init__(self):
        print(f"[DEBUG][key_setup.py] ClientKeys created for client_id={self.client_id}")


class KeySetup:
    """
    Implements Algorithm 2: SetupKeys
    
    Sets up encryption and signature key pairs for all clients,
    and establishes shared keys between each pair of clients.
    
    Protocol Steps:
    1. Each client generates signing key pair (sk_i, vk_i)
    2. Clients exchange verification keys
    3. Clients perform key exchange to establish shared encryption keys
    4. Setup VSS keys for secret sharing
    """
    
    def __init__(self, num_clients: int, security_param: int = 128):
        """
        Initialize key setup.
        
        Args:
            num_clients: Total number of clients N
            security_param: Security parameter κ (default 128 bits)
        """
        print(f"[DEBUG][key_setup.py] KeySetup.__init__ called: num_clients={num_clients}, security_param={security_param}")
        self.num_clients = num_clients
        self.security_param = security_param
        self.client_keys: Dict[int, ClientKeys] = {}
    
    def setup_all_keys(self) -> Dict[int, ClientKeys]:
        """
        Execute full key setup protocol (Algorithm 2).
        
        Returns:
            Dict mapping client_id -> ClientKeys
        """
        print("[DEBUG][key_setup.py] KeySetup.setup_all_keys called")
        
        # Step 1: Each client generates signing key pair
        print("[DEBUG][key_setup.py] Step 1: Generating signing keys")
        self._generate_signing_keys()
        
        # Step 2: Distribute verification keys to all clients
        print("[DEBUG][key_setup.py] Step 2: Distributing verification keys")
        self._distribute_verification_keys()
        
        # Step 3: Generate key exchange pairs and compute shared keys
        print("[DEBUG][key_setup.py] Step 3: Establishing shared keys")
        self._establish_shared_keys()
        
        # Step 4: Setup VSS keys for each client
        print("[DEBUG][key_setup.py] Step 4: Setting up VSS keys")
        self._setup_vss_keys()
        
        print(f"[DEBUG][key_setup.py] KeySetup.setup_all_keys completed: {len(self.client_keys)} clients")
        return self.client_keys
    
    def _generate_signing_keys(self):
        """Step 1: Each client generates signing key pair."""
        print(f"[DEBUG][key_setup.py] _generate_signing_keys: generating for {self.num_clients} clients")
        for i in range(self.num_clients):
            signing_key = DSASignature()
            self.client_keys[i] = ClientKeys(
                client_id=i,
                signing_key=signing_key,
            )
        print(f"[DEBUG][key_setup.py] _generate_signing_keys completed")
    
    def _distribute_verification_keys(self):
        """Step 2: Distribute verification keys to all clients."""
        print(f"[DEBUG][key_setup.py] _distribute_verification_keys called")
        for i in range(self.num_clients):
            for j in range(self.num_clients):
                if i != j:
                    self.client_keys[i].verification_keys[j] = \
                        self.client_keys[j].signing_key.public_key
        print(f"[DEBUG][key_setup.py] _distribute_verification_keys completed")
    
    def _establish_shared_keys(self):
        """Step 3: Generate key exchange pairs and compute shared keys."""
        print(f"[DEBUG][key_setup.py] _establish_shared_keys called")
        # Using Diffie-Hellman for key agreement
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                # In practice, this would use proper DH exchange
                # For simulation, we generate a shared random key
                shared_key = self._generate_shared_key()
                self.client_keys[i].shared_keys[j] = shared_key
                self.client_keys[j].shared_keys[i] = shared_key
        print(f"[DEBUG][key_setup.py] _establish_shared_keys completed")
    
    def _setup_vss_keys(self):
        """Step 4: Setup VSS keys for each client."""
        print(f"[DEBUG][key_setup.py] _setup_vss_keys called")
        for i in range(self.num_clients):
            sk, pks, _ = setup_vss(t=self.num_clients // 2)
            self.client_keys[i].vss_sk = sk
            self.client_keys[i].vss_pks = pks
        print(f"[DEBUG][key_setup.py] _setup_vss_keys completed")
    
    def _generate_shared_key(self) -> bytes:
        """Generate a random shared key for encryption."""
        return random.randbytes(32)
    
    def get_client_keys(self, client_id: int) -> Optional[ClientKeys]:
        """Get keys for a specific client."""
        print(f"[DEBUG][key_setup.py] get_client_keys called: client_id={client_id}")
        return self.client_keys.get(client_id)
