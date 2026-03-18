# RFLPA: Robust Federated Learning against Poisoning Attacks

A federated learning framework with **secure aggregation** and **Byzantine-robust aggregation** using trust scores. Includes support for simulating various poisoning attacks.

## Features

- **Secure Aggregation**: Packed Shamir secret sharing for privacy-preserving gradient aggregation
- **Trust-based Aggregation**: Cosine similarity-based trust scores to detect and mitigate malicious clients
- **Attack Simulation**: Label-flip, scaling, and backdoor attacks for robustness evaluation
- **Two Pipelines**:
  - `RFLPAPipeline`: Full cryptographic secure aggregation
  - `RFLPAPipelineSimple`: Simplified pipeline for fast testing (no crypto overhead)

## Installation

```bash
pip install -r requirements.txt
```

Requires **Python 3.11+**.

## Quick Start

### Basic Training (No Attacks)
```bash
python fl/run_rflpa.py --simple --num_clients 5 --iterations 10
```

### Training with Attacks
```bash
# Label-flip attack (30% malicious clients)
python fl/run_rflpa.py --simple --attack label_flip --attack_prop 0.3 --iterations 10

# Scaling attack
python fl/run_rflpa.py --simple --attack scaling --attack_prop 0.3 --scale_factor 10

# Backdoor attack
python fl/run_rflpa.py --simple --attack backdoor --attack_prop 0.5 --poison_ratio 0.5
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | Mnist | Dataset: Mnist, Cifar10 |
| `--num_clients` | 10 | Number of federated clients |
| `--iterations` | 50 | Training rounds |
| `--lr` | 0.01 | Learning rate |
| `--local_epochs` | 5 | Local training epochs per round |
| `--batch_size` | 32 | Batch size |
| `--simple` | False | Use simplified pipeline (no crypto) |
| `--device` | cpu | Device: cpu, cuda |

### Attack Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--attack` | none | Attack: none, label_flip, scaling, backdoor |
| `--attack_prop` | 0.0 | Fraction of malicious clients |
| `--flip_mode` | random | Label-flip mode: random, targeted |
| `--scale_factor` | 10.0 | Gradient scaling multiplier |
| `--poison_ratio` | 0.5 | Fraction of data poisoned (backdoor) |

## Project Structure

```
rflpa/
├── fl/                      # Federated learning core
│   ├── run_rflpa.py         # Main entry point
│   ├── rflpa_pipeline.py    # Full RFLPA pipeline
│   ├── pipeline_simple.py   # Simplified pipeline
│   ├── attack_utils.py      # Attack implementations
│   ├── config.py            # Configuration dataclass
│   └── nodes.py             # Client node implementation
├── crypto/                  # Cryptographic primitives
│   ├── packed_shamir.py     # Packed secret sharing
│   ├── shamir.py            # Shamir secret sharing
│   └── finite_field.py      # Finite field operations
├── agg/                     # Aggregation strategies
│   └── fl_trust.py          # Trust-based aggregation
└── data/                    # Dataset storage
```

## Attack Types

1. **Label-Flip**: Malicious clients flip training labels to degrade model accuracy
2. **Scaling**: Malicious clients scale gradients to dominate aggregation
3. **Backdoor (BadNet)**: Inject trigger patterns to create backdoor behavior

