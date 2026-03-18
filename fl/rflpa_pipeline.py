"""
RFLPA Full Pipeline Implementation - Coordinator

This module coordinates the full RFLPA protocol by importing and orchestrating:
- config.py: Configuration (RFLPAConfig)
- key_setup.py: Algorithm 2 (SetupKeys)
- robust_sec_agg.py: 4-round RobustSecAgg protocol
- pipeline_simple.py: Simplified pipeline for testing

References:
- Algorithm 1: Main RFLPA training loop (implemented here)
- Algorithm 2: Key setup protocol (key_setup.py)
- RobustSecAgg: 4-round secure aggregation (robust_sec_agg.py)
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
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from torch.utils.data import DataLoader, Subset

# Import from fl folder (matches nodes.py pattern)
from fl.config import RFLPAConfig, DEFAULT_PRIME
from fl.key_setup import KeySetup, ClientKeys
from fl.robust_sec_agg import RobustSecAgg
from fl.pipeline_simple import RFLPAPipelineSimple, RFLPAServerSimple
from fl.server import RFLPAServer
from fl.nodes import Client, ClientUpdate
from fl.attack_utils import AttackConfig, create_attack, BaseAttack

# Fallback utilities
data2label = {"Mnist": 10, "Cifar10": 10, "Cifar100": 100}
bd_attacks = ["badnet", "sig", "wanet"]


def iid_partition(data, n):
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    return {i: set(indices[i::n]) for i in range(n)}


def non_iid_partition(data, n, alpha):
    return iid_partition(data, n)


def testing(model, data, batch_size, criterion, num_classes, args):
    """Simple testing function."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    loader = DataLoader(data, batch_size=batch_size)
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    return accuracy, avg_loss


# =============================================================================
# MAIN RFLPA PIPELINE (Algorithm 1)
# =============================================================================

class RFLPAPipeline:
    """
    Main RFLPA Pipeline implementing Algorithm 1.
    
    Orchestrates the full federated learning process with:
    - Key setup phase (Algorithm 2)
    - Iterative training with secure aggregation (RobustSecAgg)
    - Support for attack simulations
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_data: Any,
        test_data: Any,
        root_data: Any,
        config: RFLPAConfig,
    ):
        """
        Initialize RFLPA pipeline.
        
        Args:
            model: Neural network model to train
            train_data: Training dataset
            test_data: Test dataset  
            root_data: Server's root dataset D0
            config: Pipeline configuration
        """
        self.config = config
        self.model = model.to(config.device)
        self.train_data = train_data
        self.test_data = test_data
        self.root_data = root_data
        
        # Partition data among clients
        self.client_data_indices = self._partition_data()
        
        # Initialize components (set in setup())
        self.server: Optional[RFLPAServer] = None
        self.clients: Dict[int, Client] = {}
        self.client_keys: Dict[int, ClientKeys] = {}
        
        # Training state
        self.current_iteration = 0
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'test_accuracy': [],
            'test_loss': [],
            'round_times': [],
        }
        
        # Attack tracking
        self.malicious_clients: Set[int] = set()
        
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
    
    def _partition_data(self) -> Dict[int, np.ndarray]:
        """Partition training data among clients."""
        if self.config.data_distribution == "iid":
            return iid_partition(self.train_data, self.config.num_clients)
        else:
            return non_iid_partition(
                self.train_data, 
                self.config.num_clients, 
                self.config.alpha
            )
    
    def setup(self):
        """
        Execute setup phase (Algorithm 2: SetupKeys).
        
        Initializes server, clients, and cryptographic keys.
        """
        print("Setting up RFLPA pipeline...")
        
        # Step 1: Setup cryptographic keys using KeySetup from key_setup.py
        key_setup = KeySetup(self.config.num_clients, self.config.security_param)
        self.client_keys = key_setup.setup_all_keys()
        print(f"  - Generated keys for {self.config.num_clients} clients")
        
        # Step 2: Initialize server with global model
        self.server = RFLPAServer(
            args=self.config,
            global_model=copy.deepcopy(self.model),
            num_clients=self.config.num_clients,
            min_clients=self.config.min_clients,
            prime=self.config.prime,
        )
        print("  - Initialized server")
        
        # Step 3: Register client public keys with server
        for client_id, keys in self.client_keys.items():
            self.server.register_client_public_key(
                client_id, 
                keys.signing_key.public_key
            )
        
        # Step 4: Get crypto params from server to initialize clients
        crypto_params = self.server.get_crypto_params()
        n = crypto_params['n']
        d = crypto_params['d']
        l = crypto_params['l']
        p = crypto_params['p']
        alpha_points = crypto_params['alpha_points']
        e_points = crypto_params['e_points']
        e_points_p = crypto_params['e_points_p']
        prime = crypto_params['prime']
        
        # Step 5: Initialize client protocols with new Client dataclass
        recipients_ids = list(range(self.config.num_clients))
        
        for client_id in range(self.config.num_clients):
            client_dataset = self.client_data_indices[client_id]
            local_train_fn = self._create_local_train_fn(client_id)
            
            # Create Client with new constructor parameters
            client = Client(
                client_id=client_id,
                dataset=client_dataset,
                local_train_fn=local_train_fn,
                l=l,
                p=p,
                d=d,
                e_points=list(e_points),
                e_points_p=list(e_points_p),
                alpha_points=list(alpha_points),
                alpha=alpha_points[client_id] if client_id < len(alpha_points) else client_id + 1,
                recipients_ids=recipients_ids,
                v0_shares=[],
                inbox_shares=[],
                partial_cs_nr={},
                inbox_cs=[],
                inbox_nr=[],
            )
            self.clients[client_id] = client
        
        print(f"  - Initialized {len(self.clients)} client protocols")
        
        # Step 6: Setup malicious clients for attack simulation
        if self.config.attack_type != "none" and self.config.attack_prop > 0:
            num_malicious = int(self.config.num_clients * self.config.attack_prop)
            self.malicious_clients = set(range(num_malicious))
            print(f"  - Configured {len(self.malicious_clients)} malicious clients")
            print(f"  - Attack handler: {type(self.attack).__name__}")
        
        print("Setup complete!")
    
    def _create_local_train_fn(self, client_id: int) -> Callable:
        """Create local training function for a client."""
        def local_train(global_model: np.ndarray, dataset_indices) -> np.ndarray:
            model = copy.deepcopy(self.model)
            self._load_flat_params(model, global_model)
            model.to(self.config.device)
            
            subset = Subset(self.train_data, list(dataset_indices))
            
            # Apply data-level attack (e.g., label flip, backdoor) using attack_utils
            subset = self.attack.apply_data_attack(
                subset, client_id, self.config.num_clients
            )
            
            loader = DataLoader(subset, batch_size=self.config.batch_size, shuffle=True)
            
            model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            
            old_params = self._get_flat_params(model)
            
            for epoch in range(self.config.local_epochs):
                for data, labels in loader:
                    if data.size(0) < 2:
                        continue
                    
                    data = data.to(self.config.device)
                    labels = labels.to(self.config.device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
            
            new_params = self._get_flat_params(model)
            gradient = new_params - old_params
            gradient = np.clip(gradient, -self.config.clip, self.config.clip)
            
            # Apply model-level attack (e.g., scaling) using attack_utils
            gradient_dict = {'gradient': torch.tensor(gradient)}
            global_dict = {'gradient': torch.tensor(old_params)}
            attacked_dict = self.attack.apply_model_attack(
                gradient_dict, global_dict, client_id, self.config.num_clients
            )
            gradient = attacked_dict['gradient'].numpy()
            
            return gradient
        
        return local_train
    
    def _apply_attack(self, labels: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Deprecated: Attack application is now handled by attack_utils."""
        return labels
    
    def _get_flat_params(self, model: nn.Module) -> np.ndarray:
        """Get flattened model parameters."""
        params = []
        for param in model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def _load_flat_params(self, model: nn.Module, flat_params: np.ndarray):
        """Load flattened parameters into model."""
        idx = 0
        for param in model.parameters():
            num_params = param.numel()
            param_data = flat_params[idx:idx + num_params].reshape(param.shape)
            param.data = torch.tensor(param_data, dtype=param.dtype, device=param.device)
            idx += num_params
    
    def _server_local_update(self) -> Tuple[np.ndarray, float]:
        """Server computes reference gradient using root dataset."""
        def server_train_fn(global_model: np.ndarray, root_data) -> np.ndarray:
            model = copy.deepcopy(self.model)
            self._load_flat_params(model, global_model)
            model.to(self.config.device)
            
            loader = DataLoader(root_data, batch_size=self.config.batch_size, shuffle=True)
            
            model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            
            old_params = self._get_flat_params(model)
            
            for epoch in range(self.config.local_epochs):
                for data, labels in loader:
                    if data.size(0) < 2:
                        continue
                    data = data.to(self.config.device)
                    labels = labels.to(self.config.device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
            
            new_params = self._get_flat_params(model)
            return new_params - old_params
        
        return self.server.compute_server_update(self.root_data, server_train_fn)
    
    def train_iteration(self) -> Dict[str, float]:
        """Execute one training iteration (one round of Algorithm 1)."""
        self.current_iteration += 1
        start_time = time.time()
        
        # Step 1: Server computes reference update
        server_gradient, g0_norm = self._server_local_update()
        
        # Step 2: Select active clients
        active_clients = self._select_active_clients()
        
        # Step 3: Get global model and v0 shares
        global_model = self.server.get_global_parameters_flat()
        v0_shares = self.server.state.server_gradient_shares
        
        # Step 4: Execute RobustSecAgg (from robust_sec_agg.py)
        robust_sec_agg = RobustSecAgg(
            server=self.server,
            clients=self.clients,
            client_keys=self.client_keys,
            config=self.config,
        )
        
        aggregated_gradient = robust_sec_agg.execute(
            active_clients=active_clients,
            global_model=global_model,
            v0_shares=v0_shares,
            g0_norm=g0_norm,
        )
        
        # Step 5: Update global model
        if aggregated_gradient is not None:
            self.server.update_global_model(
                gradient=aggregated_gradient,
                learning_rate=self.config.learning_rate,
            )
        
        # Step 6: Evaluate
        iteration_time = time.time() - start_time
        metrics = self._evaluate()
        metrics['iteration_time'] = iteration_time
        metrics['trust_scores'] = self.server.trust_scores.copy()
        
        # Update history
        self.history['test_accuracy'].append(metrics['test_accuracy'])
        self.history['test_loss'].append(metrics['test_loss'])
        self.history['round_times'].append(iteration_time)
        
        # Clear client state for next round
        for client in self.clients.values():
            client.clear_round_state()
        
        return metrics
    
    def _select_active_clients(self) -> Set[int]:
        """Select active clients for current iteration."""
        return set(range(self.config.num_clients))
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate current global model."""
        model = self.server.global_model
        model.eval()
        
        criterion = nn.CrossEntropyLoss()
        num_classes = data2label.get(self.config.dataset, 10)
        
        class Args:
            device = self.config.device
        
        test_acc, test_loss = testing(
            model, self.test_data, self.config.batch_size,
            criterion, num_classes, Args()
        )
        
        return {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
        }
    
    def train(self, num_iterations: Optional[int] = None) -> Dict[str, List[float]]:
        """Execute full training loop (Algorithm 1 main loop)."""
        if num_iterations is None:
            num_iterations = self.config.num_iterations
        
        print(f"\nStarting RFLPA training for {num_iterations} iterations...")
        print(f"  - Clients: {self.config.num_clients}")
        print(f"  - Min clients: {self.config.min_clients}")
        print(f"  - Attack: {self.config.attack_type} ({self.config.attack_prop*100:.1f}%)")
        print()
        
        for t in range(1, num_iterations + 1):
            metrics = self.train_iteration()
            
            if t % 10 == 0 or t == 1:
                print(f"Iteration {t}/{num_iterations}: "
                      f"Acc={metrics['test_accuracy']:.2f}%, "
                      f"Loss={metrics['test_loss']:.4f}, "
                      f"Time={metrics['iteration_time']:.2f}s")
        
        print("\nTraining complete!")
        return self.history
    
    def get_trust_scores(self) -> Dict[int, float]:
        """Get current trust scores for all clients."""
        return self.server.trust_scores.copy()
    
    def get_model(self) -> nn.Module:
        """Get current global model."""
        return self.server.global_model


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_rflpa_pipeline(
    model: nn.Module,
    train_data: Any,
    test_data: Any,
    root_data: Any,
    num_clients: int = 10,
    attack_type: str = "none",
    attack_prop: float = 0.0,
    simple: bool = False,
    **kwargs,
) -> RFLPAPipeline:
    """
    Create and setup an RFLPA pipeline.
    
    Args:
        model: Neural network model
        train_data: Training dataset
        test_data: Test dataset
        root_data: Server's root dataset
        num_clients: Number of clients
        attack_type: Type of attack simulation
        attack_prop: Proportion of malicious clients
        simple: Use simplified pipeline (no crypto)
        **kwargs: Additional config parameters
        
    Returns:
        Configured RFLPAPipeline or RFLPAPipelineSimple instance
    """
    config = RFLPAConfig(
        num_clients=num_clients,
        attack_type=attack_type,
        attack_prop=attack_prop,
        **kwargs,
    )
    
    if simple:
        return RFLPAPipelineSimple(
            model=model,
            train_data=train_data,
            test_data=test_data,
            root_data=root_data,
            config=config,
        )
    
    pipeline = RFLPAPipeline(
        model=model,
        train_data=train_data,
        test_data=test_data,
        root_data=root_data,
        config=config,
    )
    
    pipeline.setup()
    return pipeline


def run_attack_simulation(
    model: nn.Module,
    train_data: Any,
    test_data: Any,
    root_data: Any,
    attack_types: List[str] = None,
    attack_props: List[float] = None,
    num_iterations: int = 100,
    num_clients: int = 10,
    simple: bool = True,
) -> Dict[str, Dict]:
    """
    Run attack simulations with different attack configurations.
    
    Args:
        model: Neural network model
        train_data: Training dataset
        test_data: Test dataset
        root_data: Server's root dataset
        attack_types: List of attack types to simulate
        attack_props: List of attack proportions to test
        num_iterations: Number of training iterations
        num_clients: Number of clients
        simple: Use simplified pipeline
        
    Returns:
        Results dictionary with metrics for each configuration
    """
    if attack_types is None:
        attack_types = ["none", "label_flip", "scaling"]
    
    if attack_props is None:
        attack_props = [0.0, 0.1, 0.2, 0.3]
    
    results = {}
    
    for attack_type in attack_types:
        for attack_prop in attack_props:
            if attack_type == "none" and attack_prop > 0:
                continue
            
            print(f"\n{'='*60}")
            print(f"Attack: {attack_type}, Proportion: {attack_prop*100:.0f}%")
            print(f"{'='*60}")
            
            config_key = f"{attack_type}_{int(attack_prop*100)}pct"
            
            try:
                pipeline = create_rflpa_pipeline(
                    model=copy.deepcopy(model),
                    train_data=train_data,
                    test_data=test_data,
                    root_data=root_data,
                    num_clients=num_clients,
                    attack_type=attack_type,
                    attack_prop=attack_prop,
                    num_iterations=num_iterations,
                    simple=simple,
                )
                
                history = pipeline.train(num_iterations)
                
                results[config_key] = {
                    'history': history,
                    'final_accuracy': history['test_accuracy'][-1] if history['test_accuracy'] else 0,
                    'trust_scores': pipeline.get_trust_scores(),
                }
                
            except Exception as e:
                print(f"Error in simulation: {e}")
                import traceback
                traceback.print_exc()
                results[config_key] = {'error': str(e)}
    
    return results


# Export all public classes and functions
__all__ = [
    'RFLPAConfig',
    'RFLPAPipeline',
    'RFLPAPipelineSimple',
    'RFLPAServerSimple',
    'KeySetup',
    'ClientKeys',
    'RobustSecAgg',
    'create_rflpa_pipeline',
    'run_attack_simulation',
    'DEFAULT_PRIME',
]
