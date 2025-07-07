# OpenVLA: Open-Source Vision-Language-Action Model

A PyTorch implementation of the OpenVLA: An [Open-Source Vision-Language-Action Model](https://arxiv.org/pdf/2406.09246) paper model that combines visual perception, natural language understanding, and robotic action prediction for embodied AI tasks.

## Overview

OpenVLA is a multimodal transformer that:

- **Sees**: Processes RGB images using a vision encoder (DINOv2 + Siglip)
- **Understands**: Interprets natural language instructions using Qwen2-0.5B
- **Acts**: Predicts discretized robotic actions (7-DOF: position, rotation, gripper)

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Image     │    │ Instruction  │    │   Action        │
│ (224x224x3) │    │   (Text)     │    │ (7D vector)     │
└─────┬───────┘    └──────┬───────┘    └─────────────────┘
      │                   │                     ▲
      ▼                   ▼                     │
┌─────────────┐    ┌──────────────┐             │
│   Vision    │    │  Language    │             │
│  Encoder    │    │  Tokenizer   │             │
│  (DINOv2)   │    │   (Qwen2)    │             │
└─────┬───────┘    └──────┬───────┘             │
      │                   │                     │
      └───────┬───────────┘                     │
              ▼                                 │
      ┌──────────────┐                          │
      │   Qwen2-0.5B │                          │
      │ Language LM  │                          │
      └──────┬───────┘                          │
             │                                  │
             ▼                                  │
      ┌──────────────┐    ┌─────────────────┐   │
      │    Logits    │    │     Action      │   │
      │ (Vocab Size) │───▶│  Discretizer    │───┘
      └──────────────┘    └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/pie33000/OpenVLA
cd OpenVLA

# Create conda environment
conda create -n openvla python=3.12
conda activate openvla

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### Supported Datasets

- **VIOLA**: Vision-based manipulation tasks
- ALL Open X Embodiement Datasets

**Install Google Cloud CLI**

    https://cloud.google.com/storage/docs/gsutil_install#linux

**Download Viola Dataset**

    gsutil -m cp -r gs://gresearch/robotics/viola/0.1.0 .

### Convert TensorFlow Datasets to HDF5

```bash
python convert_tf_to_numpy.py
```

This will:

1. Download TensorFlow datasets
2. Convert to HDF5 format in `data/` directory
3. Extract: images, instructions, and 7-DOF actions

### Data Structure

```
data/
├── viola_0.h5
├── viola_1.h5
└── viola_2.h5
```

Each HDF5 file contains:

- `images`: RGB images (H, W, 3)
- `instrs`: Natural language instructions
- `actions`: 7-DOF action vectors [x, y, z, rx, ry, rz, gripper]

## Quick Start

### Training

```python
# Train the model
python train.py
```

Training features:

- **Scalable Data Loading**: Handles large datasets efficiently
- **Action Discretization**: Converts continuous actions to discrete tokens
- **Cross-entropy Loss**: Standard language modeling objective
- **Memory Optimization**: File handle caching and lazy loading

## Key Components

### 1. Vision Encoder (`vision_encoder.py`)

- **Base**: DINOv2 vision transformer + Siglip
- **Output**: 1536-dimensional visual embeddings
- **Input**: 224×224 RGB images

### 2. Action Discretizer (`action.py`)

```python
discretizer = ActionDiscretizer(
    tokenizer=tokenizer,      # Qwen2 tokenizer
    action_dim=7,             # [x,y,z,rx,ry,rz,gripper]
    num_bins=256,             # Discretization bins
    min_action=-1.0,          # Action range minimum
    max_action=1.0            # Action range maximum
)

# Convert continuous action to tokens
action = np.array([-0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0])
tokens = discretizer(action)  # Returns token IDs

# Convert tokens back to actions
decoded_action = discretizer.decode_token_ids_to_actions(tokens)
```

### 3. OpenVLA Model (`openvla.py`)

```python
class OpenVLA(nn.Module):
    def __init__(self, dim=896, device="cuda", action_dim=7, num_bins=256):
        # Vision encoder (DINOv2)
        self.vision_encoder = VisionEncoder(dim=dim)

        # Language model (Qwen2-0.5B)
        self.language_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

        # Action discretizer
        self.action_discretizer = ActionDiscretizer(...)
```

## Advanced Features

### Custom Action Spaces

```python
# 6-DOF manipulation (no gripper)
discretizer = ActionDiscretizer(
    tokenizer=tokenizer,
    action_dim=6,
    num_bins=512,  # Higher resolution
    min_action=-2.0,
    max_action=2.0
)

# Mobile manipulation (base + arm)
discretizer = ActionDiscretizer(
    tokenizer=tokenizer,
    action_dim=10,  # [base_x, base_y, base_theta, arm_joints...]
    num_bins=256,
    min_action=-1.0,
    max_action=1.0
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - [see the LICENSE file for details](LICENSE).

## References

- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)
- [RT-1: Robotics Transformer](https://arxiv.org/abs/2212.06817)

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---
