# MoG-RPCA: Mixture of Gaussians Robust PCA

Python implementation of the MoG-RPCA algorithm migrated from the LRSLibrary MATLAB codebase.

## Overview

MoG-RPCA (Mixture of Gaussians - Robust PCA) is a robust principal component analysis algorithm that uses Bayesian inference with a Mixture of Gaussians model to handle complex noise patterns in data decomposition.

**Key Features:**
- Bayesian factorization with uncertainty quantification
- Mixture of Gaussians noise modeling
- Automatic rank determination
- SVD-based initialization for fast convergence
- Suitable for video background/foreground separation

## Algorithm Details

- **Category:** RPCA (Robust Principal Component Analysis)
- **Speed Class:** 2 (2-5 seconds execution time)
- **Complexity:** Medium-High (⭐⭐⭐⭐)
- **Dependencies:** NumPy, SciPy (no external RPCA libraries required)

**Reference:**
> Qian Zhao, Deyu Meng, Zongben Xu, Wangmeng Zuo, Lei Zhang. 
> "Robust Principal Component Analysis with Complex Noise." ICML, 2014.

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- numpy>=1.20.0
- scipy>=1.7.0
- scikit-learn>=1.0.0
- opencv-python>=4.5.0 (for video processing)
- streamlit>=1.25.0 (for GUI)

## Usage

### Python API

```python
from algorithms.rpca.mog_rpca import run_alg

# Prepare your data matrix M (each column is a vectorized frame)
M = ...  # shape: (pixels, frames)

# Set parameters
params = {
    'mog_k': 3,           # Number of Gaussian components
    'lr_init': 'SVD',     # Initialization method: 'SVD' or 'rand'
    'maxiter': 100,       # Maximum iterations
    'r': 1,               # Base rank (initial_rank = 2*r)
    'tol': 1e-3          # Convergence tolerance
}

# Run algorithm
results = run_alg(M, params)

# Extract results
L = results['L']  # Low-rank component (background)
S = results['S']  # Sparse component (foreground)
O = results['O']  # Additional outputs (models, estimated rank)
```

### Video Processing

```python
from utils.video_utils import process_video_with_algorithm
from algorithms.rpca.mog_rpca import run_alg

# Process a video file
results = process_video_with_algorithm(
    'input_video.mp4',
    run_alg,
    params={'mog_k': 3, 'r': 1},
    output_dir='output'
)

# Results contain paths to processed videos
print(results['L'])  # Low-rank component video
print(results['S'])  # Sparse component video
```

### GUI Interface

Launch the Streamlit web interface:

```bash
cd gui
streamlit run app.py
```

The GUI provides:
- Video file upload (AVI, MP4)
- Interactive parameter adjustment
- Real-time processing
- Side-by-side visualization of I, O, L, S components
- Download links for results

## File Structure

```
lrslibrary_python/
├── algorithms/
│   └── rpca/
│       └── mog_rpca/
│           ├── __init__.py
│           ├── mog_rpca.py      # Core algorithm implementation
│           └── run_alg.py       # Standard interface
├── utils/
│   ├── __init__.py
│   └── video_utils.py           # Video processing utilities
├── gui/
│   └── app.py                   # Streamlit GUI
├── test_mog_rpca.py            # Test script
├── requirements.txt
└── README.md
```

## Algorithm Parameters

### Core Parameters
- `mog_k` (int, default=3): Number of Gaussian components in the mixture model
- `lr_init` (str, default='SVD'): Low-rank initialization method ('SVD' or 'rand')
- `maxiter` (int, default=100): Maximum number of iterations
- `initial_rank` (int, default=2*r): Initial rank of low-rank component
- `tol` (float, default=1e-3): Convergence tolerance

### Prior Hyperparameters

**Low-Rank Priors:**
- `lr_a0` (float, default=1e-6): Gamma prior shape parameter
- `lr_b0` (float, default=1e-6): Gamma prior rate parameter

**MoG Priors:**
- `mog_mu0` (float, default=0): Prior mean
- `mog_c0` (float, default=1e-3): Gamma prior shape for precision
- `mog_d0` (float, default=1e-3): Gamma prior rate for precision
- `mog_alpha0` (float, default=1e-3): Dirichlet prior for mixture weights
- `mog_beta0` (float, default=1e-3): Prior precision for mean

## Testing

Run the test script with synthetic data:

```bash
python test_mog_rpca.py
```

This generates synthetic low-rank data with mixture of Gaussian noise and verifies the algorithm can correctly decompose it.

## Implementation Notes

### Key Differences from MATLAB
- Uses NumPy/SciPy instead of MATLAB built-ins
- `scipy.special.digamma` replaces MATLAB's `psi`
- `scipy.linalg.svd` with `full_matrices=False` for efficient SVD
- Custom `logsumexp` implementation for numerical stability
- OpenCV for video I/O instead of MATLAB's video tools

### Numerical Considerations
- Log-space computations prevent underflow in probability calculations
- Automatic rank pruning removes redundant dimensions (threshold=100)
- Covariance matrices stored as 3D arrays for efficiency

## Performance

Tested on synthetic data (100×100 matrix, rank-5):
- Convergence in ~20-50 iterations
- Relative reconstruction error < 0.05
- Execution time: 2-5 seconds on modern hardware

## License

Code is migrated from LRSLibrary MATLAB implementation. Please cite the original paper if you use this implementation in your research.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer.
