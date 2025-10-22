# LRSLibrary Python

Python implementation of the LRSLibrary (Low-Rank and Sparse Library) for background modeling and subtraction in videos.

## Overview

This library provides low-rank and sparse decomposition algorithms migrated from the original MATLAB implementation. Each algorithm decomposes an input matrix M into:
- **L**: Low-rank component (background/structure)
- **S**: Sparse component (foreground/outliers)
- **O**: Additional outputs (typically hard-thresholded sparse component)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from lrslibrary import run_algorithm

M = np.random.randn(2304, 51)

result = run_algorithm('RPCA', 'GoDec', M, params={'rank': 1, 'power': 0})

L = result['L']
S = result['S']
O = result['O']
```

## Implemented Algorithms

### RPCA (Robust Principal Component Analysis)
- **GoDec**: Go Decomposition with randomized low-rank approximation

### MC (Matrix Completion)
- **SVT**: Singular Value Thresholding (in progress)

### ST (Subspace Tracking)
- **GRASTA**: Grassmannian Robust Adaptive Subspace Tracking (in progress)

## Algorithm Categories

- RPCA: Robust PCA
- ST: Subspace Tracking
- MC: Matrix Completion
- LRR: Low-Rank Representation
- TTD: Three-Term Decomposition
- NMF: Non-negative Matrix Factorization
- NTF: Non-negative Tensor Factorization
- TD: Tensor Decomposition

## Development Status

This is a work in progress. Currently implemented:
- [x] Core infrastructure (dispatcher, config)
- [x] GoDec (RPCA)
- [ ] SVT (MC)
- [ ] GRASTA (ST)
- [ ] MoG-RPCA (RPCA)
- [ ] ReProCS (ST)

## Original MATLAB Implementation

This Python library is based on the MATLAB LRSLibrary:
https://github.com/andrewssobral/lrslibrary

## License

MIT License
