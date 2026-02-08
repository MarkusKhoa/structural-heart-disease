# Structural Heart Disease - Joint Embedding Model

A PyTorch implementation of a **joint embedding model** for multimodal ECG data, combining waveform signals and tabular features using cross-modal attention.

## Project Structure

```
structural_heart_disease/
├── echonext_dataset/              # Data folder (data files only)
│   ├── EchoNext_train_waveforms.npy
│   ├── EchoNext_train_tabular_features.npy
│   ├── EchoNext_val_waveforms.npy
│   ├── EchoNext_val_tabular_features.npy
│   ├── EchoNext_test_waveforms.npy
│   ├── EchoNext_test_tabular_features.npy
│   ├── EchoNext_metadata_100k.csv
│   ├── README.md                  # Dataset documentation
│   └── LICENSE.txt
│
├── joint_embedding_model.py       # Core model implementation
├── dataset.py                     # PyTorch dataset classes
├── train.py                       # Training script
├── visualize_embeddings.py        # Visualization utilities
├── example_usage.py               # Usage examples
├── requirements.txt               # Python dependencies
├── JOINT_EMBEDDING_README.md      # Detailed documentation
├── IMPLEMENTATION_SUMMARY.md      # Implementation overview
└── README.md                      # This file
```

## Quick Start

### Automated Setup (Recommended)

**Using venv:**
```bash
./setup_env.sh
```

**Using Conda/Anaconda:**
```bash
./setup_conda.sh
```

**Using Makefile:**
```bash
make install-dev
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

### Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .  # Install in editable mode

# 3. Run examples
python example_usage.py

# 4. Train a model
python train.py --data_dir ./echonext_dataset --epochs 50
```

### Using Makefile

```bash
make help          # Show all commands
make example       # Run examples
make train         # Train model
make test          # Run tests
make format        # Format code
```

## Key Features

✅ **Joint Embeddings**: Cross-modal attention fuses waveforms and tabular features into a unified embedding space  
✅ **Flexible Architecture**: Supports both embedding extraction and classification tasks  
✅ **Production-Ready**: Complete with training, evaluation, and visualization tools  
✅ **Well-Documented**: Comprehensive documentation and examples  

## Core Components

### 1. **joint_embedding_model.py**
- `JointEmbeddingModel`: Main model for creating joint embeddings
- `JointEmbeddingWithClassifier`: Model with classification head
- `CrossModalAttention`: Bidirectional cross-modal attention mechanism
- `WaveformEncoder`: 1D CNN encoder for ECG waveforms (1×2500×12)
- `TabularEncoder`: MLP encoder for tabular features (7 features)

### 2. **dataset.py**
- `EchoNextDataset`: PyTorch dataset for single-label tasks
- `EchoNextMultiLabelDataset`: Dataset for multi-label classification
- Memory-efficient loading with memory-mapped arrays
- Support for 12 binary diagnostic labels

### 3. **train.py**
- Complete training pipeline with validation and checkpointing
- Command-line interface for easy experimentation
- Automatic learning rate scheduling and early stopping
- Function to extract embeddings for downstream tasks

### 4. **visualize_embeddings.py**
- t-SNE/PCA visualization of embeddings
- Embedding distribution analysis
- Cosine similarity heatmaps

## Architecture Overview

```
ECG Waveform (1×2500×12)          Tabular Features (7)
         ↓                                  ↓
   WaveformEncoder                   TabularEncoder
    (1D CNN layers)                   (MLP layers)
         ↓                                  ↓
   (batch, 32, 256)                   (batch, 1, 256)
         ↓                                  ↓
         └──────────→ Cross-Modal Attention ←──────────┘
                              ↓
                    Concatenate & Pool
                              ↓
                      Fusion Layers
                              ↓
                   Joint Embedding (512)
```

## Usage Example

```python
from joint_embedding_model import create_joint_embedding_model
from dataset import EchoNextDataset
import torch

# Load dataset
dataset = EchoNextDataset(
    data_dir='./echonext_dataset',
    split='train',
    label_column='shd_moderate_or_greater_flag'
)

# Create model
model = create_joint_embedding_model(
    embedding_dim=512,
    hidden_dim=256,
    use_cross_attention=True
)

# Generate embedding
sample = dataset[0]
waveform = sample['waveform'].unsqueeze(0)
tabular = sample['tabular'].unsqueeze(0)

with torch.no_grad():
    embedding, _ = model(waveform, tabular)
    
print(f"Joint embedding shape: {embedding.shape}")  # (1, 512)
```

## Dataset

The EchoNext dataset contains 100,000 ECGs from Columbia and Allen hospitals:
- **Training**: ~87,000 samples
- **Validation**: ~3,000 samples
- **Test**: ~4,000 samples

Each ECG includes:
- **Waveforms**: (1, 2500, 12) - 10 seconds, 12 leads, 250 Hz
- **Tabular features**: 7 features (sex, rates, intervals, age)
- **Labels**: 12 binary diagnostic labels

---

**Version**: 1.0.0  
**Last Updated**: 2026-01-13
