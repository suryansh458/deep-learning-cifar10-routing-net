# CIFAR-10 RoutingNet (PyTorch)

End-to-end PyTorch project for CIFAR-10 with a custom **routed CNN** (multi-expert attention), mixup regularisation, cosine LR, and a reproducible training pipeline. Designed with production conventions: modular code, configs, and automation.

## Features
- Routing backbone with K conv experts and learned attention gate
- SGD + momentum, cosine schedule, optional label smoothing
- Clean configuration (YAML), deterministic seeding
- Notebook for exploration and full script training

## Repository Layout
deep-learning-cifar10-routing-net/
├── src/                     model, training, utils
├── notebooks/               cifar10_routing_net.ipynb
├── configs/                 training_config.yaml
├── docs/                    report.pdf
├── requirements.txt         dependencies
└── README.md

## Quickstart
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.training.train --config configs/training_config.yaml

## Evaluate
python -m src.training.evaluate

## Skills
Deep Learning (CNNs, attention/routing, augmentation, schedulers), Software Engineering (modular PyTorch, config-driven training, repo hygiene), Experimentation (metrics, reporting).

License: MIT
