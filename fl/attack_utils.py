"""
Attack Utilities for RFLPA Federated Learning

  - Label-Flip: Malicious clients flip labels to degrade model accuracy.
  - Scaling: Malicious clients scale their model updates to dominate aggregation.
  - Backdoor (BadNet): Malicious clients inject a trigger pattern into data
    so the global model learns a backdoor mapping.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AttackConfig:
    """Configuration that fully describes an attack."""

    attack_type: str = "none"          # none | label_flip | scaling | backdoor
    attack_prop: float = 0.0           # fraction of clients that are malicious
    num_classes: int = 10              # number of label classes

    # --- Label-Flip options ---
    flip_mode: str = "random"          # random | targeted
    target_label: int = 0             # used when flip_mode == "targeted"

    # --- Scaling options ---
    scale_factor: float = 10.0        # multiplier applied to malicious updates
    scale_mode: str = "full"           # full | partial (partial scales a random
                                       #   subset of parameters)
    partial_ratio: float = 0.5        # fraction of params scaled when partial

    # --- Backdoor / BadNet options ---
    trigger_pattern: str = "pixel"     # pixel | patch | pattern
    trigger_size: int = 3              # size (pixels) of the trigger patch
    trigger_value: float = 1.0        # intensity of the trigger
    trigger_label: int = 0            # target label for backdoor samples
    poison_ratio: float = 0.5         # fraction of local data poisoned


# =============================================================================
# Base class
# =============================================================================

class BaseAttack:
    """Base class — no-op by default."""

    def __init__(self, config: AttackConfig):
        self.config = config

    def _is_malicious(self, client_id: int, num_clients: int) -> bool:
        """Deterministically decide whether *client_id* is malicious."""
        num_mal = max(1, int(math.ceil(self.config.attack_prop * num_clients)))
        return client_id < num_mal

    def apply_data_attack(
        self,
        dataset: Dataset,
        client_id: int,
        num_clients: int,
    ) -> Dataset:
        """Return (possibly poisoned) dataset for *client_id*."""
        return dataset

    def apply_model_attack(
        self,
        local_updates: Dict[str, torch.Tensor],
        global_params: Dict[str, torch.Tensor],
        client_id: int,
        num_clients: int,
    ) -> Dict[str, torch.Tensor]:
        """Return (possibly poisoned) model update for *client_id*."""
        return local_updates


# =============================================================================
# Label-Flip Attack
# =============================================================================

class _LabelFlippedDataset(Dataset):
    """Wraps a dataset and flips its labels."""

    def __init__(self, base_dataset: Dataset, num_classes: int,
                 mode: str = "random", target_label: int = 0):
        self.base = base_dataset
        self.num_classes = num_classes
        self.mode = mode
        self.target_label = target_label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data, label = self.base[idx]
        if self.mode == "targeted":
            new_label = self.target_label
        else:
            new_label = (label + random.randint(1, self.num_classes - 1)) % self.num_classes
        return data, new_label


class LabelFlipAttack(BaseAttack):
    """Malicious clients train on flipped labels."""

    def apply_data_attack(self, dataset, client_id, num_clients):
        if not self._is_malicious(client_id, num_clients):
            return dataset
        return _LabelFlippedDataset(
            dataset,
            num_classes=self.config.num_classes,
            mode=self.config.flip_mode,
            target_label=self.config.target_label,
        )


# =============================================================================
# Scaling Attack
# =============================================================================

class ScalingAttack(BaseAttack):
    """Malicious clients scale their model updates to dominate aggregation."""

    def apply_model_attack(self, local_updates, global_params,
                           client_id, num_clients):
        if not self._is_malicious(client_id, num_clients):
            return local_updates

        factor = self.config.scale_factor
        poisoned = {}

        if self.config.scale_mode == "partial":
            keys = list(local_updates.keys())
            n_scale = max(1, int(len(keys) * self.config.partial_ratio))
            scale_keys = set(random.sample(keys, n_scale))
        else:
            scale_keys = set(local_updates.keys())

        for k, v in local_updates.items():
            if k in scale_keys:
                poisoned[k] = v * factor
            else:
                poisoned[k] = v

        return poisoned


# =============================================================================
# Backdoor (BadNet) Attack
# =============================================================================

class _BackdoorDataset(Dataset):
    """Wraps a dataset and injects a trigger into a fraction of samples."""

    def __init__(self, base_dataset: Dataset, config: AttackConfig):
        self.base = base_dataset
        self.config = config
        self.num_poison = max(1, int(len(base_dataset) * config.poison_ratio))
        self._poison_indices = set(random.sample(
            range(len(base_dataset)),
            min(self.num_poison, len(base_dataset))
        ))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data, label = self.base[idx]
        if idx in self._poison_indices:
            data = self._stamp_trigger(data.clone())
            label = self.config.trigger_label
        return data, label

    def _stamp_trigger(self, img: torch.Tensor) -> torch.Tensor:
        """Stamp a trigger onto *img* (C×H×W)."""
        c, h, w = img.shape
        s = self.config.trigger_size
        val = self.config.trigger_value
        pattern = self.config.trigger_pattern

        if pattern == "pixel":
            img[:, h - 1, w - 1] = val
        elif pattern == "patch":
            r_start = max(0, h - s)
            c_start = max(0, w - s)
            img[:, r_start:h, c_start:w] = val
        elif pattern == "pattern":
            r_start = max(0, h - s)
            c_start = max(0, w - s)
            for ri in range(r_start, h):
                for ci in range(c_start, w):
                    if (ri + ci) % 2 == 0:
                        img[:, ri, ci] = val
                    else:
                        img[:, ri, ci] = -val
        else:
            img[:, h - 1, w - 1] = val

        return img


class BackdoorAttack(BaseAttack):
    """Malicious clients inject a backdoor trigger into their local data."""

    def apply_data_attack(self, dataset, client_id, num_clients):
        if not self._is_malicious(client_id, num_clients):
            return dataset
        return _BackdoorDataset(dataset, self.config)

    def apply_model_attack(self, local_updates, global_params,
                           client_id, num_clients):
        """Boost backdoor updates with mild scaling."""
        if not self._is_malicious(client_id, num_clients):
            return local_updates
        return {k: v * 2.0 for k, v in local_updates.items()}


# =============================================================================
# Factory
# =============================================================================

_ATTACK_REGISTRY: Dict[str, type] = {
    "none": BaseAttack,
    "label_flip": LabelFlipAttack,
    "scaling": ScalingAttack,
    "backdoor": BackdoorAttack,
    "badnet": BackdoorAttack,
}


def create_attack(config: AttackConfig) -> BaseAttack:
    """Instantiate an attack from *config*."""
    key = config.attack_type.lower().strip()
    cls = _ATTACK_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown attack type '{config.attack_type}'. "
            f"Available: {list(_ATTACK_REGISTRY.keys())}"
        )
    return cls(config)


def get_available_attacks() -> List[str]:
    """Return list of registered attack names."""
    return list(_ATTACK_REGISTRY.keys())