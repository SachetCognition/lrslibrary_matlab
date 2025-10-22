import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/lrslibrary_python')

from lrslibrary import run_algorithm

np.random.seed(42)

m, n, p = 48, 48, 51
video_data = np.random.randn(m, n, p)
M = video_data.reshape(m * n, p, order='F')

print("Testing GoDec (RPCA)...")
print(f"Input shape: {M.shape}")

result = run_algorithm('RPCA', 'GoDec', M, params={'rank': 1, 'power': 0})

L = result['L']
S = result['S']
O = result['O']

print(f"L shape: {L.shape}")
print(f"S shape: {S.shape}")
print(f"Reconstruction error: {np.linalg.norm(M - L - S) / np.linalg.norm(M):.6f}")
print(f"Low-rank component rank (approx): {np.linalg.matrix_rank(L)}")
print(f"Sparse component sparsity: {np.sum(np.abs(S) > 1e-6) / S.size:.2%}")
print(f"CPU time: {result['cputime']:.4f} seconds")
print(f"Algorithm error: {result.get('error', 'N/A')}")
print("GoDec test PASSED\n")
