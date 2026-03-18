# RFLPA: Robust Federated Learning against Poisoning Attacks

A federated learning framework with **secure aggregation** and **Byzantine-robust aggregation** using trust scores.

## Installation

```bash
pip install -r requirements.txt
```

Requires **Python 3.11+**.

## Quick Start

```bash
# Basic training (no attacks)
python fl/run_rflpa.py --simple  --model renet9 --num_clients 5 --iterations 10

# With label-flip attack
python fl/run_rflpa.py --simple --model renet9 --attack label_flip --attack_prop 0.3

# With scaling attack
python fl/run_rflpa.py --simple --model renet9 --attack scaling --scale_factor 10

# With backdoor attack
python fl/run_rflpa.py --simple --model renet9 --attack backdoor --poison_ratio 0.5
```

## Features

- **Secure Aggregation**: Packed Shamir secret sharing for privacy-preserving gradient aggregation
- **Trust-based Aggregation**: Cosine similarity-based trust scores to detect malicious clients  
- **Attack Simulation**: Label-flip, scaling, and backdoor attacks
- **Two Pipelines**: Full crypto pipeline or simplified (fast) pipeline

## Project Structure

```
rflpa/
├── fl/              # Federated learning core
├── crypto/          # Cryptographic primitives (Shamir secret sharing)
├── agg/             # Aggregation strategies (FLTrust)
└── data/            # Dataset storage
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | Mnist | Mnist or Cifar10 |
| `--num_clients` | 10 | Number of clients |
| `--iterations` | 50 | Training rounds |
| `--attack` | none | none, label_flip, scaling, backdoor |
| `--attack_prop` | 0.0 | Fraction of malicious clients |
| `--simple` | - | Use fast pipeline (no crypto) |
| `--model` | resnet9 | simple |
See [fl/README.md](fl/README.md) for full documentation.
