# CIFAR-10 RoutingNet (PyTorch)

Production-quality deep-learning repo in **PyTorch** for **CIFAR-10**. Uses a **routed CNN** (multi-expert attention), strong augmentation with **mixup**, and a reproducible training pipeline. Built for engineering review and hiring managers.

## Highlights
- Routing backbone: multiple conv experts gated by GAP→MLP→softmax
- Training: SGD + momentum, cosine LR schedule, optional label smoothing
- Data: CIFAR-10 transforms; early-phase mixup
- Reproducibility: deterministic seeds, clean I/O, single-command setup

## Structure
deep-learning-cifar10-routing-net/
├── src/
│   ├── models/                # routing_net.py
│   ├── training/              # add train.py for CLI pipeline
│   └── utils/                 # my_utils.py
├── notebooks/                 # cifar10_routing_net.ipynb
├── docs/                      # Report.pdf
├── report/                    # figures (git-ignored)
├── data/                      # datasets (git-ignored)
├── requirements.txt
└── README.md

## Quickstart
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/cifar10_routing_net.ipynb

## Skills
Deep Learning (CNNs, routing/attention, augmentation, schedulers),
Software Engineering (modular PyTorch, reproducible setup, repo hygiene),
Experimentation (metrics, technical reporting)

License: MIT
