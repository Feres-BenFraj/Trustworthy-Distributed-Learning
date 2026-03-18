"""
RFLPA Configuration

Contains configuration dataclass and default constants for the RFLPA protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Import the fixed prime from crypto module for consistency
from crypto.packed_shamir import _P

# Default prime for finite field operations (must match crypto modules)
DEFAULT_PRIME = _P

print("[DEBUG][config.py] Module loaded")


@dataclass
class RFLPAConfig:
    """Configuration for RFLPA pipeline."""
    
    # Client configuration
    num_clients: int = 10
    min_clients: int = 7
    
    # Training parameters
    num_iterations: int = 100
    learning_rate: float = 0.01
    local_epochs: int = 5
    batch_size: int = 32
    
    # Cryptographic parameters
    prime: int = DEFAULT_PRIME
    precision: int = 1000  # Quantization precision q
    security_param: int = 128
    clip: float = 10.0  # Gradient clipping threshold
    
    # Attack simulation
    attack_type: str = "none"  # "none", "label_flip", "scaling", "badnet", "backdoor"
    attack_prop: float = 0.0  # Proportion of malicious clients
    num_classes: int = 10  # Number of label classes
    
    # Label-flip attack options
    flip_mode: str = "random"  # "random" or "targeted"
    target_label: int = 0  # Target label for targeted flip
    
    # Scaling attack options
    scale_factor: float = 10.0  # Multiplier for scaling attack
    scale_mode: str = "full"  # "full" or "partial"
    partial_ratio: float = 0.5  # Fraction of params scaled when partial
    
    # Backdoor attack options
    trigger_pattern: str = "pixel"  # "pixel", "patch", or "pattern"
    trigger_size: int = 3  # Size of trigger in pixels
    trigger_value: float = 1.0  # Intensity of trigger
    trigger_label: int = 0  # Target label for backdoor samples
    poison_ratio: float = 0.5  # Fraction of local data poisoned
    
    # Data configuration
    dataset: str = "Mnist"
    data_distribution: str = "iid"  # "iid" or "non_iid"
    alpha: float = 0.5  # Dirichlet alpha for non-iid
    
    # Device
    device: str = "cpu"
    
    def __post_init__(self):
        print(f"[DEBUG][config.py] RFLPAConfig initialized:")
        print(f"  - num_clients: {self.num_clients}")
        print(f"  - min_clients: {self.min_clients}")
        print(f"  - num_iterations: {self.num_iterations}")
        print(f"  - attack_type: {self.attack_type}, attack_prop: {self.attack_prop}")
        print(f"  - dataset: {self.dataset}, distribution: {self.data_distribution}")
        print(f"  - device: {self.device}")
