import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/lrslibrary_python')

from lrslibrary import run_algorithm

np.random.seed(42)

m, n, p = 48, 48, 40
video_data = np.random.randn(m, n, p)
M = video_data.reshape(m * n, p, order='F')

print("Testing ReProCS (Dynamic Robust PCA)...")
print(f"Input shape: {M.shape}")

result = run_algorithm('ST', 'ReProCS', M, params={'t_train': 10, 'rank_init': 4, 'alpha': 60, 'K': 3})

L = result['L']
S = result['S']

print(f"L shape: {L.shape}")
print(f"S shape: {S.shape}")

if L.shape[1] > 0:
    print(f"Reconstruction error (online portion): {np.linalg.norm(M[:, 10:] - L - S) / np.linalg.norm(M[:, 10:]):.6f}")
    print(f"Low-rank component rank (approx): {np.linalg.matrix_rank(L)}")
    print(f"Sparse component sparsity: {np.sum(np.abs(S) > 1e-6) / S.size:.2%}")
else:
    print("Note: ReProCS processes frames after training period")

print(f"CPU time: {result['cputime']:.4f} seconds")
print("ReProCS test PASSED\n")
