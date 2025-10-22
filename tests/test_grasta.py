import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/lrslibrary_python')

from lrslibrary import run_algorithm

np.random.seed(42)

m, n, p = 48, 48, 100

background = np.random.randn(m * n)
L_true = np.outer(background, np.ones(p))
L_true = L_true / np.linalg.norm(L_true) * 10

S_true = np.zeros((m * n, p))
sparse_ratio = 0.01  # 1% sparse (typical for foreground objects)
num_sparse = int(sparse_ratio * m * n * p)
sparse_idx = np.random.choice(m * n * p, num_sparse, replace=False)
sparse_rows = sparse_idx // p
sparse_cols = sparse_idx % p
S_true[sparse_rows, sparse_cols] = np.random.randn(num_sparse) * 2

M = L_true + S_true

print("Testing GRASTA (Subspace Tracking) with synthetic low-rank + sparse data...")
print(f"Input shape: {M.shape}")
print(f"True L rank: {np.linalg.matrix_rank(L_true)}")
print(f"True S sparsity: {np.sum(np.abs(S_true) > 1e-6) / S_true.size:.2%}")

result = run_algorithm('ST', 'GRASTA', M, params={
    'rank': 1, 
    'subsampling': 1.0,
    'use_svd_init': True,
    'svd_init_frames': 10,
    'debug_admm': True
})

L = result['L']
S = result['S']

print(f"\nRecovered L shape: {L.shape}")
print(f"Recovered S shape: {S.shape}")
print(f"Reconstruction error: {np.linalg.norm(M - L - S) / np.linalg.norm(M):.6f}")
print(f"Recovered L rank: {np.linalg.matrix_rank(L)}")
print(f"Recovered S sparsity: {np.sum(np.abs(S) > 1e-6) / S.size:.2%}")
print(f"L recovery error: {np.linalg.norm(L - L_true) / np.linalg.norm(L_true):.6f}")
print(f"S recovery error: {np.linalg.norm(S - S_true) / np.linalg.norm(S_true):.6f}")
print(f"CPU time: {result['cputime']:.4f} seconds")

if np.linalg.matrix_rank(L) <= 5 and np.sum(np.abs(S) > 1e-6) / S.size < 0.5:
    print("GRASTA test PASSED - Successfully separated low-rank and sparse components!\n")
else:
    print("GRASTA test WARNING - Decomposition quality could be better\n")
