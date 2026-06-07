# CIFAR-10 Image Classification with CNN

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange?style=flat-square&logo=pytorch)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-74.93%25-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A Convolutional Neural Network (CNN) built and trained from scratch using PyTorch on the CIFAR-10 benchmark dataset, achieving **74.93% test accuracy** across 10 object categories.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Getting Started](#getting-started)
- [Training](#training)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

This project implements a CNN classifier on CIFAR-10 using pure PyTorch — no pretrained weights or transfer learning. The focus is on understanding end-to-end deep learning: data loading, architecture design, training loop, and evaluation.

---

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60,000 color images across 10 classes:

| Property | Value |
|---|---|
| Training set | 50,000 images |
| Test set | 10,000 images |
| Image size | 32 × 32 × 3 (RGB) |
| Classes | 10 |
| Batch size | 64 |

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

The dataset is downloaded automatically via `torchvision.datasets.CIFAR10` on first run.

**Preprocessing:**
```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

---

## Model Architecture

```
Input (3 × 32 × 32)
    ↓
Conv2d(3 → 32, kernel=3, padding=1) → ReLU → MaxPool2d(2×2)    # → 32 × 16 × 16
    ↓
Conv2d(32 → 64, kernel=3, padding=1) → ReLU → MaxPool2d(2×2)   # → 64 × 8 × 8
    ↓
Conv2d(64 → 128, kernel=3, padding=1) → ReLU → MaxPool2d(2×2)  # → 128 × 4 × 4
    ↓
Flatten → 2048
    ↓
Linear(2048 → 256) → ReLU
    ↓
Linear(256 → 10)
```

Defined in PyTorch:

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
```

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **74.93%** |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam (default lr = 0.001) |
| Epochs | 10 |

**Training loss per epoch:**

| Epoch | Loss |
|---|---|
| 1 | 0.1320 |
| 2 | 0.1143 |
| 3 | 0.0975 |
| 4 | 0.0870 |
| 5 | 0.0829 |
| 6 | 0.0796 |
| 7 | 0.0712 |
| 8 | 0.0715 |
| 9 | 0.0634 |
| 10 | 0.0756 |

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/cifar10-cnn.git
cd cifar10-cnn

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training

Open and run the notebook:

```bash
jupyter notebook CIFAR10.ipynb
```

Or run as a script:

```bash
python train.py
```

The CIFAR-10 dataset (~170 MB) will be downloaded automatically into `./data/` on the first run.

---

## Project Structure

```
cifar10-cnn/
├── data/               # CIFAR-10 dataset (auto-downloaded)
├── CIFAR10.ipynb       # Main notebook (data loading, model, training, evaluation)
├── requirements.txt
└── README.md
```

---

## Dependencies

```
torch
torchvision
```

Install with:

```bash
pip install torch torchvision
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
