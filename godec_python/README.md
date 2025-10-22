# GoDec Algorithm - Python Implementation

This directory contains a Python implementation of the **GoDec (Go Decomposition)** algorithm, migrated from the MATLAB version in the LRSLibrary.

## Overview

GoDec is a Robust Principal Component Analysis (RPCA) algorithm that decomposes a matrix X into:
- **L**: Low-rank component (background)
- **S**: Sparse component (foreground/outliers)

The algorithm uses randomized low-rank approximation and hard thresholding for efficient matrix decomposition, primarily used for background modeling and subtraction in videos.

## Algorithm Details

**Reference**: Tianyi Zhou and Dacheng Tao, "GoDec: Randomized Low-rank & Sparse Matrix Decomposition in Noisy Case", ICML 2011

**Key Features**:
- Randomized power iteration for low-rank approximation
- Hard thresholding for sparse component extraction
- Handles matrix transposition when m < n (more features than samples)
- Fast execution (Class 1 speed in LRSLibrary)
- Minimal dependencies (NumPy and SciPy only)

## Files

- `godec.py` - Core GoDec algorithm implementation
- `run_alg.py` - Standard interface wrapper compatible with LRSLibrary structure
- `test_godec.py` - Comprehensive unit tests
- `README.md` - This file

## Installation

```bash
pip install numpy scipy
```

## Usage

### Basic Usage

```python
import numpy as np
from godec import godec

X = np.random.randn(100, 50)
rank = 5
card = 500
power = 0

L, S, RMSE, error = godec(X, rank, card, power)

print(f"Reconstruction error: {error}")
print(f"Low-rank shape: {L.shape}")
print(f"Sparse shape: {S.shape}")
print(f"Iterations: {len(RMSE)}")
```

### Using Standard Interface

```python
from run_alg import run_alg

M = np.random.randn(100, 50)

result = run_alg(M, params={'rank': 5, 'card': 500, 'power': 1})
L = result['L']
S = result['S']
rmse_list = result['O']['RMSE']
final_error = result['O']['error']
```

### Video Processing Example

```python
import numpy as np
from godec import godec

frame_height, frame_width, num_frames = 48, 48, 51
video_3d = np.random.randn(frame_height, frame_width, num_frames)

M = video_3d.reshape(-1, num_frames)

rank = 3
card = int(M.size * 0.05)
power = 1

L, S, RMSE, error = godec(M, rank, card, power)

L_video = L.reshape(frame_height, frame_width, num_frames)
S_video = S.reshape(frame_height, frame_width, num_frames)
```

## Parameters

### Input Parameters

- **X** (np.ndarray): n×p data matrix with n samples and p features
- **rank** (int): Constraint on rank(L) ≤ rank
- **card** (int): Cardinality constraint on S (number of non-zero elements)
- **power** (int): Power scheme modification (≥0). Higher values lead to better accuracy but more computation time.

### Output

- **L** (np.ndarray): Low-rank component matrix
- **S** (np.ndarray): Sparse component matrix
- **RMSE** (List[float]): List of error values per iteration
- **error** (float): Final reconstruction error ||X-L-S||/||X||

## Algorithm Details

1. **Initialization**:
   - If m < n, transpose X
   - L = X, S = sparse zeros

2. **Main Loop** (max 100 iterations):
   - **Update L** (randomized power iteration):
     - Generate random matrix Y2 of size (n, rank)
     - Iterate (power+1) times: Y1 = L×Y2, Y2 = L'×Y1
     - QR decomposition: Q, R = qr(Y2)
     - L_new = (L×Q)×Q'
   
   - **Update S** (hard thresholding):
     - T = L - L_new + S
     - L = L_new
     - Select top 'card' elements by magnitude from T
     - S[top_indices] = T[top_indices]
   
   - **Check Convergence**:
     - Zero out selected elements in T
     - RMSE = ||T||
     - Stop if RMSE < 1e-3 or iterations > 100
     - Otherwise: L = L + T

3. **Finalization**:
   - Compute error = ||L+S-X||/||X||
   - If transposed, transpose L and S back

## Testing

Run the comprehensive test suite:

```bash
python test_godec.py
```

The test suite includes:
1. **Basic Decomposition**: Tests with known low-rank + sparse structure
2. **Convergence**: Verifies RMSE decreases or stabilizes
3. **Transpose Handling**: Tests wide matrices (m < n)
4. **Power Parameter**: Tests different power values
5. **Rank Parameter**: Tests robustness with different ranks
6. **Video Frame Format**: Simulates video processing workflow

All tests use synthetic data with known ground truth components.

## Performance

- **Speed Class**: Class 1 (fastest category in LRSLibrary)
- **Typical execution time**: <2 seconds for 2304×51 matrices
- **Memory**: Efficient with sparse matrix representation

## Differences from MATLAB Version

- Uses NumPy's `qr(..., mode='reduced')` instead of MATLAB's `qr(..., 0)`
- Uses Python's flattening and indexing conventions
- Returns regular NumPy arrays instead of MATLAB sparse matrices (more flexible)
- Otherwise maintains identical algorithm logic and numerical behavior

## License

This implementation maintains compatibility with the original MATLAB LRSLibrary license.

## Original MATLAB Implementation

The original MATLAB implementation can be found at:
- Repository: `SachetCognition/lrslibrary_matlab`
- File: `algorithms/rpca/GoDec/GoDec.m`
