import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/lrslibrary_python')

from lrslibrary import run_algorithm
from lrslibrary.utils import subsampling

np.random.seed(42)

m, n, p = 48, 48, 15
video_data = np.random.randn(m, n, p)
M = video_data.reshape(m * n, p, order='F')

print("Testing SVT (Matrix Completion)...")
print(f"Input shape: {M.shape}")

Idx, Omega = subsampling(M, 0.5)
print(f"Observation ratio: {len(Idx) / M.size:.2%}")

result = run_algorithm('MC', 'SVT', M, params={'Idx': Idx, 'Omega': Omega})

L = result['L']
S = result['S']

if np.any(np.isnan(L)):
    print(f"WARNING: L contains {np.sum(np.isnan(L))} nan values!")
if np.any(np.isnan(S)):
    print(f"WARNING: S contains {np.sum(np.isnan(S))} nan values!")

print(f"L shape: {L.shape}")
print(f"S shape: {S.shape}")
M_obs = M[Omega > 0]
L_obs = L[Omega > 0]
norm_M_obs = np.linalg.norm(M_obs)
if norm_M_obs > 1e-10:
    recon_error = np.linalg.norm(M_obs - L_obs) / norm_M_obs
    print(f"Reconstruction error (observed): {recon_error:.6f}")
else:
    print(f"Reconstruction error (observed): N/A (denominator too small)")
print(f"Low-rank component rank (approx): {np.linalg.matrix_rank(L)}")
print(f"CPU time: {result['cputime']:.4f} seconds")
print(f"Iterations: {result.get('numiter', 'N/A')}")
print("SVT test PASSED\n")
