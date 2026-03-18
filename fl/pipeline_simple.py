"""
RFLPA Simplified Pipeline

A simplified version of the RFLPA pipeline without full cryptographic operations.
Useful for quick testing, debugging, and attack simulation experiments.
"""

from __future__ import annotations

import sys
import os

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Set, Any
from torch.utils.data import DataLoader, Subset

# Import from fl folder (matches nodes.py pattern)
from fl.config import RFLPAConfig
from fl.attack_utils import AttackConfig, create_attack, BaseAttack

print("[DEBUG][pipeline_simple.py] Module loaded")

# Fallback data label mapping
data2label = {"Mnist": 10, "Cifar10": 10, "Cifar100": 100}


def iid_partition(data, n):
    print(f"[DEBUG][pipeline_simple.py] iid_partition: {len(data)} samples -> {n} clients")
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    return {i: set(indices[i::n]) for i in range(n)}


def non_iid_partition(data, n, alpha):
    print(f"[DEBUG][pipeline_simple.py] non_iid_partition: {len(data)} samples -> {n} clients, alpha={alpha}")
    return iid_partition(data, n)


class RFLPAServerSimple:
    """
    Simplified RFLPA Server for testing without full cryptographic operations.
    """
    
    def __init__(self, args: Any, global_model: Any, num_clients: int, min_clients: int):
        print(f"[DEBUG][pipeline_simple.py] RFLPAServerSimple.__init__: num_clients={num_clients}, min_clients={min_clients}")
        self.args = args
        self.global_model = global_model
        self.num_clients = num_clients
        self.min_clients = min_clients
        
        self.client_updates: Dict[int, Dict] = {}
        self.trust_scores: Dict[int, float] = {}
        self.server_gradient: np.ndarray = None
        self.server_gradient_norm: float = 0.0
    
    def set_server_gradient(self, gradient: np.ndarray):
        """Set reference gradient for trust score computation."""
        print(f"[DEBUG][pipeline_simple.py] SERVER: set_server_gradient, shape={gradient.shape}")
        self.server_gradient = np.asarray(gradient, dtype=np.float64).ravel()
        self.server_gradient_norm = float(np.linalg.norm(self.server_gradient))
        print(f"[DEBUG][pipeline_simple.py] SERVER: gradient_norm={self.server_gradient_norm:.4f}")
    
    def receive_client_update(self, client_id: int, gradient: np.ndarray):
        """Receive and store a client update."""
        print(f"[DEBUG][pipeline_simple.py] SERVER <- CLIENT {client_id}: Receiving update")
        grad = np.asarray(gradient, dtype=np.float64).ravel()
        self.client_updates[client_id] = {
            'gradient': grad,
            'norm': float(np.linalg.norm(grad)),
        }
        print(f"[DEBUG][pipeline_simple.py] SERVER: Stored update from client {client_id}, norm={self.client_updates[client_id]['norm']:.4f}")
    
    def compute_trust_scores(self) -> Dict[int, float]:
        """Compute trust scores based on cosine similarity with server gradient."""
        print("[DEBUG][pipeline_simple.py] SERVER: Computing trust scores...")
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
            
            print(f"[DEBUG][pipeline_simple.py] SERVER: Client {client_id} trust_score={self.trust_scores[client_id]:.4f}")
        
        return self.trust_scores
    
    def aggregate(self) -> np.ndarray:
        """Perform weighted aggregation based on trust scores."""
        print("[DEBUG][pipeline_simple.py] SERVER: Aggregating gradients...")
        if not self.trust_scores:
            self.compute_trust_scores()
        
        total_trust = sum(self.trust_scores.values())
        print(f"[DEBUG][pipeline_simple.py] SERVER: total_trust={total_trust:.4f}")
        
        if total_trust == 0:
            weights = {cid: 1.0 / len(self.client_updates) for cid in self.client_updates}
            print("[DEBUG][pipeline_simple.py] SERVER: Using uniform weights (total_trust=0)")
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
        print(f"[DEBUG][pipeline_simple.py] SERVER: Aggregation complete, result_norm={np.linalg.norm(result):.4f}")
        return result
    
    def update_global_model(self, gradient: np.ndarray, learning_rate: float = 1.0):
        """Apply aggregated gradient to global model.
        
        Note: 'gradient' is actually the model update (new_params - old_params),
        so we ADD it to the global model (not subtract).
        """
        print(f"[DEBUG][pipeline_simple.py] SERVER: Updating global model, lr={learning_rate}")
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
        print("[DEBUG][pipeline_simple.py] SERVER: Global model updated")
    
    def reset(self):
        """Reset server state for new round."""
        print("[DEBUG][pipeline_simple.py] SERVER: Resetting state for new round")
        self.client_updates = {}
        self.trust_scores = {}


class RFLPAPipelineSimple:
    """
    Simplified RFLPA pipeline without full cryptographic operations.
    Useful for quick testing and attack simulation experiments.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_data: Any,
        test_data: Any,
        root_data: Any,
        config: RFLPAConfig,
    ):
        print("[DEBUG][pipeline_simple.py] RFLPAPipelineSimple.__init__")
        self.config = config
        self.model = model.to(config.device)
        self.train_data = train_data
        self.test_data = test_data
        self.root_data = root_data
        
        # Partition data
        print("[DEBUG][pipeline_simple.py] Partitioning data...")
        if config.data_distribution == "iid":
            self.client_data = iid_partition(train_data, config.num_clients)
        else:
            self.client_data = non_iid_partition(train_data, config.num_clients, config.alpha)
        
        # Initialize server
        self.server = RFLPAServerSimple(
            args=config,
            global_model=copy.deepcopy(model),
            num_clients=config.num_clients,
            min_clients=config.min_clients,
        )
        
        # Track malicious clients
        num_malicious = int(config.num_clients * config.attack_prop)
        self.malicious_clients = set(range(num_malicious))
        print(f"[DEBUG][pipeline_simple.py] Malicious clients: {self.malicious_clients}")
        
        # Initialize attack handler from attack_utils
        self.attack_config = AttackConfig(
            attack_type=config.attack_type,
            attack_prop=config.attack_prop,
            num_classes=config.num_classes,
            flip_mode=config.flip_mode,
            target_label=config.target_label,
            scale_factor=config.scale_factor,
            scale_mode=config.scale_mode,
            partial_ratio=config.partial_ratio,
            trigger_pattern=config.trigger_pattern,
            trigger_size=config.trigger_size,
            trigger_value=config.trigger_value,
            trigger_label=config.trigger_label,
            poison_ratio=config.poison_ratio,
        )
        self.attack = create_attack(self.attack_config)
        print(f"[DEBUG][pipeline_simple.py] Attack handler: {type(self.attack).__name__}")
        
        self.history = {'test_accuracy': [], 'test_loss': []}
    
    def train(self, num_iterations: int) -> Dict[str, List[float]]:
        """Run simplified training loop."""
        print(f"\n[DEBUG][pipeline_simple.py] ========== TRAINING START ==========")
        print(f"[DEBUG][pipeline_simple.py] Starting simplified RFLPA training for {num_iterations} iterations...")
        print(f"[DEBUG][pipeline_simple.py]   - Clients: {self.config.num_clients}")
        print(f"[DEBUG][pipeline_simple.py]   - Malicious: {len(self.malicious_clients)}")
        print()
        
        for t in range(1, num_iterations + 1):
            print(f"\n[DEBUG][pipeline_simple.py] ========== ITERATION {t}/{num_iterations} ==========")
            self.server.reset()
            
            # Server update
            print("[DEBUG][pipeline_simple.py] SERVER: Computing server gradient from root data...")
            server_grad = self._compute_gradient(self.root_data, client_id=-1)
            self.server.set_server_gradient(server_grad)
            
            # Client updates
            print("[DEBUG][pipeline_simple.py] Collecting client updates...")
            for client_id in range(self.config.num_clients):
                indices = self.client_data[client_id]
                print(f"[DEBUG][pipeline_simple.py] CLIENT {client_id}: Computing local gradient...")
                grad = self._compute_gradient(indices, client_id)
                print(f"[DEBUG][pipeline_simple.py] CLIENT {client_id} -> SERVER: Sending gradient")
                self.server.receive_client_update(client_id, grad)
            
            # Aggregate and update
            print("[DEBUG][pipeline_simple.py] SERVER: Processing updates...")
            self.server.compute_trust_scores()
            agg_grad = self.server.aggregate()
            self.server.update_global_model(agg_grad, self.config.learning_rate)
            
            # Evaluate
            if t % 10 == 0 or t == 1:
                print("[DEBUG][pipeline_simple.py] Evaluating model...")
                metrics = self._evaluate()
                self.history['test_accuracy'].append(metrics['accuracy'])
                self.history['test_loss'].append(metrics['loss'])
                print(f"[DEBUG][pipeline_simple.py] Iter {t}: Acc={metrics['accuracy']:.2f}%, Loss={metrics['loss']:.4f}")
        
        print("\n[DEBUG][pipeline_simple.py] ========== TRAINING COMPLETE ==========")
        return self.history
    
    def _compute_gradient(self, data_indices, client_id: int) -> np.ndarray:
        """Compute gradient for given data."""
        is_malicious = client_id in self.malicious_clients
        if is_malicious:
            print(f"[DEBUG][pipeline_simple.py] CLIENT {client_id}: MALICIOUS - applying attack")
        
        model = copy.deepcopy(self.server.global_model)
        model.train()
        
        if isinstance(data_indices, (set, np.ndarray, list)):
            subset = Subset(self.train_data, list(data_indices))
        else:
            subset = data_indices
        
        # Apply data-level attack (e.g., label flip, backdoor) using attack_utils
        if client_id >= 0:  # Not server
            subset = self.attack.apply_data_attack(
                subset, client_id, self.config.num_clients
            )
        
        loader = DataLoader(subset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        old_params = self._get_params(model)
        
        for _ in range(self.config.local_epochs):
            for data, labels in loader:
                if data.size(0) < 2:
                    continue
                data = data.to(self.config.device)
                labels = labels.to(self.config.device)
                
                optimizer.zero_grad()
                loss = criterion(model(data), labels)
                loss.backward()
                optimizer.step()
        
        new_params = self._get_params(model)
        gradient = new_params - old_params
        
        # Apply model-level attack (e.g., scaling) using attack_utils
        if client_id >= 0:  # Not server
            gradient_dict = {'gradient': torch.tensor(gradient)}
            global_dict = {'gradient': torch.tensor(old_params)}
            attacked_dict = self.attack.apply_model_attack(
                gradient_dict, global_dict, client_id, self.config.num_clients
            )
            gradient = attacked_dict['gradient'].numpy()
        
        print(f"[DEBUG][pipeline_simple.py] CLIENT {client_id}: Gradient computed, norm={np.linalg.norm(gradient):.4f}")
        return gradient
    
    def _get_params(self, model: nn.Module) -> np.ndarray:
        params = []
        for p in model.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def _evaluate(self) -> Dict[str, float]:
        model = self.server.global_model
        model.eval()
        
        correct = 0
        total = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        loader = DataLoader(self.test_data, batch_size=self.config.batch_size)
        
        with torch.no_grad():
            for data, labels in loader:
                data = data.to(self.config.device)
                labels = labels.to(self.config.device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return {
            'accuracy': 100 * correct / total if total > 0 else 0,
            'loss': total_loss / len(loader) if len(loader) > 0 else 0,
        }
    
    def get_trust_scores(self) -> Dict[int, float]:
        """Get current trust scores."""
        return self.server.trust_scores.copy()
    
    def get_model(self) -> nn.Module:
        """Get current global model."""
        return self.server.global_model
