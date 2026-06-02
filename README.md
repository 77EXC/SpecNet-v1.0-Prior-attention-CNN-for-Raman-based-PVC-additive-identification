# SpecNet: Rapid Identification of PVC Plastics and Their Additives via Raman Spectroscopy

<div align="center">

[![Python](https://img.shields.io/pypi/pyversions/torch)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/github/license/chenxq1993/SpecNet)](LICENSE)

</div>

## Project Overview

**SpecNet** is a deep learning framework designed for the rapid identification of polyvinyl chloride (PVC) plastics and their potential toxic chemical additives using Raman spectroscopy. The system implements a multi-branch convolutional neural network (CNN) architecture capable of simultaneous classification and semi-quantitative analysis of multiple additives.

### Key Features

- **Multi-Branch CNN Architecture**: Specialized branches for fingerprint region, mid-frequency, and high-frequency spectral bands
- **Multi-Label Classification**: Simultaneously identify multiple additives in a single sample
- **Semi-Quantitative Analysis**: Predict additive content percentages
- **Data Augmentation**: Built-in augmentation for improved generalization
- **Early Stopping**: Prevent overfitting with configurable patience

### Supported Additives

The model can identify the following chemical components:

- **Polymer**: PVC (Polyvinyl Chloride)
- **Plasticizers**: DEHP, DINP, DIDP, ATBC, TOCP, ESBO
- **Stabilizers**: ZnO, CaZn, BaCd, Pb Stevens, Pb Silicate
- **Fillers**: TiO2, CaCO3
- **Flame Retardants**: HBCD, TBPH, TBBPA
- And more...

## Installation

```bash
# Clone the repository
git clone https://github.com/chenxq1993/SpecNet.git
cd SpecNet

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
torch>=2.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.4.0
pybaselines>=2.0.0
```

## Quick Start

### 1. Data Preparation

Prepare your Raman spectroscopy data in JSON format:

```json
{
  "sample_id": "sample_001",
  "raman_spectrum": {
    "wavenumber": [100, 150, 200, ...],
    "intensity": [0.1, 0.5, 0.8, ...]
  },
  "components": {
    "PVC": 100,
    "DEHP": 25.5,
    "ZnO": 3.2
  }
}
```

### 2. Training

```python
python train.py
```

Key training parameters can be configured in `train.py`:

```python
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 0.001
PATIENCE = 50  # Early stopping patience
```

### 3. Evaluation

```python
python eval.py
```

### 4. Inference

Predict components for a single sample:

```python
python inference.py sample_001 --data_path validation_dataset.json
```

## Project Structure

```
SpecNet/
├── train.py           # Training script
├── eval.py            # Evaluation script
├── inference.py       # Inference script
├── data_processing.py # Data preprocessing
├── data_convert.py   # Data format conversion
├── data_detection.py # Data validation
├── split.py          # Dataset splitting
├── dft.py            # DFT analysis utilities
├── requirements.txt # Dependencies
├── README.md        # This file
└── LICENSE          # MIT License
```

## Model Architecture

The SpecNet model adopts a multi-branch architecture:

1. **Fingerprint Branch** (0-1750 cm⁻¹): Captures polymer characteristic peaks
2. **Mid-Frequency Branch** (1750-2750 cm⁻¹): C-H stretching region
3. **High-Frequency Branch** (2750-3500 cm⁻¹): O-H and N-H stretching
4. **Global Branch**: Full spectrum for context

All branches are fused through a shared fully connected layer for joint classification and regression.

## Data Preprocessing

The pipeline includes:

1. **Baseline Correction**: Asymmetric Least Squares (ARPLS)
2. **Smoothing**: Savitzky-Golay filter
3. **Interpolation**: Uniform wavenumber grid
4. **Normalization**: Min-Max scaling

## Performance

Based on our trained model on the validation dataset:

| Metric | Score |
|--------|-------|
| Exact Match Ratio | 91.36% |
| Average Precision | 97.18% |
| Average Recall | 98.29% |
| Average F1 Score | 97.22% |
| Content MAE | ~11.15% |

> **Note**: The MAE for content prediction varies significantly across different additive types. Major components (high concentration) show better accuracy than trace additives.

## Citation

If you use SpecNet in your research, please cite:

```bibtex
@software{specnet2026,
  author = {Feng Huy},
  title = {SpecNet: Rapid Identification of PVC Plastics and Their Additives via Raman Spectroscopy},
  year = {2026},
  url = {https://github.com/chenxq1993/SpecNet}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [1] Machine Learning of Polymer Types From the Spectral Signature Of Raman Spectroscopy microplastics data
- [2] ConInceDeep: A novel deep learning method for component identification of mixture based on Raman spectroscopy
- [3] RamanFormer: A Transformer-Based Quantification Approach for Raman Mixture components

## Acknowledgments

This research was conducted as part of the NJU2027 P2 project focusing on microplastic analysis.

---

<div align="center">

Built with PyTorch ❤️

</div>