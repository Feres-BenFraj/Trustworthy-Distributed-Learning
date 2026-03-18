"""
RFLPA Runner Script

Example usage:
    python run_rflpa.py --dataset Mnist --num_clients 10 --iterations 100
    python run_rflpa.py --dataset Cifar10 --attack label_flip --attack_prop 0.2
    python run_rflpa.py --run_attack_simulation
"""

import sys
import os

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import Subset, random_split

# Import from fl folder (matches nodes.py pattern)
from fl.rflpa_pipeline import (
    RFLPAPipeline,
    RFLPAPipelineSimple,
    RFLPAConfig,
    create_rflpa_pipeline,
    run_attack_simulation,
)
from fl.attack_utils import AttackConfig, create_attack, get_available_attacks


# =============================================================================
# SIMPLE MODELS
# =============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for MNIST."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNNCifar(nn.Module):
    """Simple CNN for CIFAR-10."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNetMNIST(nn.Module):
    """ResNet18 adapted for MNIST (1-channel, 28x28)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        # Modify first conv layer for 1-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove maxpool for small images
        self.resnet.maxpool = nn.Identity()
        # Modify final FC layer
        self.resnet.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


class ResNetCIFAR(nn.Module):
    """ResNet18 adapted for CIFAR-10 (3-channel, 32x32)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        # Modify first conv layer for small images
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove maxpool for small images
        self.resnet.maxpool = nn.Identity()
        # Modify final FC layer
        self.resnet.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


# ResNet9 building blocks
def conv_block(in_channels, out_channels, pool=False):
    """Conv block with optional pooling."""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    """ResNet9 - compact 9-layer ResNet for small images."""
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


class ResNet9MNIST(ResNet9):
    """ResNet9 for MNIST (1-channel, 28x28)."""
    def __init__(self, num_classes=10):
        super().__init__(in_channels=1, num_classes=num_classes)


class ResNet9CIFAR(ResNet9):
    """ResNet9 for CIFAR-10 (3-channel, 32x32)."""
    def __init__(self, num_classes=10):
        super().__init__(in_channels=3, num_classes=num_classes)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(dataset_name: str, model_name: str = "simple", root_size: int = 100):
    """Load dataset and create train/test/root splits."""
    
    if dataset_name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
        if model_name.lower() == "resnet":
            model = ResNetMNIST(num_classes=10)
        elif model_name.lower() == "resnet9":
            model = ResNet9MNIST(num_classes=10)
        else:
            model = SimpleCNN(num_classes=10)
        
    elif dataset_name.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        if model_name.lower() == "resnet":
            model = ResNetCIFAR(num_classes=10)
        elif model_name.lower() == "resnet9":
            model = ResNet9CIFAR(num_classes=10)
        else:
            model = SimpleCNNCifar(num_classes=10)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create root dataset for server (small subset)
    root_indices = list(range(root_size))
    root_data = Subset(train_data, root_indices)
    
    # Remaining data for clients
    client_train_data = Subset(train_data, list(range(root_size, len(train_data))))
    
    return model, client_train_data, test_data, root_data


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def run_single_experiment(args):
    """Run a single RFLPA training experiment."""
    print(f"\n{'='*60}")
    print(f"RFLPA Training")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Clients: {args.num_clients}")
    print(f"Iterations: {args.iterations}")
    print(f"Attack: {args.attack} ({args.attack_prop*100:.0f}%)")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Load data
    model, train_data, test_data, root_data = load_data(args.dataset, args.model)
    
    # Create config
    config = RFLPAConfig(
        num_clients=args.num_clients,
        min_clients=max(3, args.num_clients * 2 // 3),
        num_iterations=args.iterations,
        learning_rate=args.lr,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        attack_type=args.attack,
        attack_prop=args.attack_prop,
        num_classes=10,  # MNIST/CIFAR10
        # Label-flip attack options
        flip_mode=args.flip_mode,
        target_label=args.target_label,
        # Scaling attack options
        scale_factor=args.scale_factor,
        scale_mode=args.scale_mode,
        partial_ratio=args.partial_ratio,
        # Backdoor attack options
        trigger_pattern=args.trigger_pattern,
        trigger_size=args.trigger_size,
        trigger_value=args.trigger_value,
        trigger_label=args.trigger_label,
        poison_ratio=args.poison_ratio,
        device=args.device,
        dataset=args.dataset,
        data_distribution=args.data_dist,
    )
    
    # Use simple pipeline for quick testing, full pipeline otherwise
    if args.simple:
        pipeline = RFLPAPipelineSimple(
            model=model,
            train_data=train_data,
            test_data=test_data,
            root_data=root_data,
            config=config,
        )
        history = pipeline.train(args.iterations)
    else:
        pipeline = RFLPAPipeline(
            model=model,
            train_data=train_data,
            test_data=test_data,
            root_data=root_data,
            config=config,
        )
        pipeline.setup()
        history = pipeline.train(args.iterations)
    
    # Print final results
    print(f"\n{'='*60}")
    print("Final Results:")
    if history['test_accuracy']:
        print(f"  Final Accuracy: {history['test_accuracy'][-1]:.2f}%")
    print(f"{'='*60}")
    
    return history


def run_attack_experiments(args):
    """Run attack simulation experiments."""
    print(f"\n{'='*60}")
    print(f"RFLPA Attack Simulation")
    print(f"{'='*60}\n")
    
    # Load data
    model, train_data, test_data, root_data = load_data(args.dataset, args.model)
    
    # Run simulations
    results = run_attack_simulation(
        model=model,
        train_data=train_data,
        test_data=test_data,
        root_data=root_data,
        attack_types=["none", "label_flip"],
        attack_props=[0.0, 0.1, 0.2, 0.3],
        num_iterations=args.iterations,
        num_clients=args.num_clients,
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("Attack Simulation Summary:")
    print(f"{'='*60}")
    for config_key, result in results.items():
        if 'error' in result:
            print(f"  {config_key}: ERROR - {result['error']}")
        else:
            print(f"  {config_key}: {result['final_accuracy']:.2f}%")
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run RFLPA Federated Learning')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='Mnist',
                        choices=['Mnist', 'Cifar10'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, default='simple',
                        choices=['simple', 'resnet', 'resnet9'],
                        help='Model architecture: simple (CNN), resnet (ResNet18), or resnet9')
    
    # Training parameters
    parser.add_argument('--num_clients', type=int, default=10,
                        help='Number of clients')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='Local epochs per client')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    # Attack parameters
    parser.add_argument('--attack', type=str, default='none',
                        choices=['none', 'label_flip', 'scaling', 'backdoor', 'badnet'],
                        help='Attack type')
    parser.add_argument('--attack_prop', type=float, default=0.0,
                        help='Proportion of malicious clients (0.0-1.0)')
    
    # Label-flip attack options
    parser.add_argument('--flip_mode', type=str, default='random',
                        choices=['random', 'targeted'],
                        help='Label flip mode: random or targeted')
    parser.add_argument('--target_label', type=int, default=0,
                        help='Target label for targeted label-flip attack')
    
    # Scaling attack options
    parser.add_argument('--scale_factor', type=float, default=10.0,
                        help='Multiplier for scaling attack')
    parser.add_argument('--scale_mode', type=str, default='full',
                        choices=['full', 'partial'],
                        help='Scale all params (full) or subset (partial)')
    parser.add_argument('--partial_ratio', type=float, default=0.5,
                        help='Fraction of params to scale when scale_mode=partial')
    
    # Backdoor attack options
    parser.add_argument('--trigger_pattern', type=str, default='pixel',
                        choices=['pixel', 'patch', 'pattern'],
                        help='Backdoor trigger pattern type')
    parser.add_argument('--trigger_size', type=int, default=3,
                        help='Size of backdoor trigger in pixels')
    parser.add_argument('--trigger_value', type=float, default=1.0,
                        help='Intensity of backdoor trigger')
    parser.add_argument('--trigger_label', type=int, default=0,
                        help='Target label for backdoor samples')
    parser.add_argument('--poison_ratio', type=float, default=0.5,
                        help='Fraction of local data to poison for backdoor attack')
    
    # Data distribution
    parser.add_argument('--data_dist', type=str, default='iid',
                        choices=['iid', 'non_iid'],
                        help='Data distribution')
    
    # Mode
    parser.add_argument('--simple', action='store_true',
                        help='Use simplified pipeline (faster, no crypto)')
    parser.add_argument('--run_attack_simulation', action='store_true',
                        help='Run attack simulation experiments')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Run
    if args.run_attack_simulation:
        run_attack_experiments(args)
    else:
        run_single_experiment(args)


if __name__ == '__main__':
    main()
