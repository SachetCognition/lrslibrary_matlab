import numpy as np
from algorithms.rpca.mog_rpca import run_alg


def test_synthetic_data():
    print("Testing MoG-RPCA with synthetic data...")
    
    m = 100
    n = 100
    r = int(0.05 * m)
    
    np.random.seed(42)
    U = np.random.randn(m, r)
    V = np.random.randn(r, n)
    M = U @ V
    
    ind = np.random.permutation(m * n)
    num1 = int(0.1 * m * n)
    num2 = int(0.2 * m * n)
    num3 = int(0.7 * m * n)
    
    N1 = np.zeros((m, n))
    N1.flat[ind[:num1]] = 50 * np.random.rand(num1) - 25
    Y = M + N1
    
    N2 = 1 * np.random.randn(num2)
    N3 = 0.1 * np.random.randn(num3)
    Y.flat[ind[num1:num1+num2]] += N2
    Y.flat[ind[num1+num2:]] += N3
    
    params = {
        'mog_k': 3,
        'lr_init': 'SVD',
        'maxiter': 100,
        'initial_rank': 2 * r,
        'tol': 1e-3,
        'r': 1
    }
    
    print(f"Input shape: {Y.shape}")
    print(f"True rank: {r}")
    
    results = run_alg(Y, params)
    
    L = results['L']
    S = results['S']
    estimated_rank = results['O']['estimated_rank']
    
    rre = np.linalg.norm(L - M, 'fro') / np.linalg.norm(M, 'fro')
    
    print(f"Estimated rank: {estimated_rank}")
    print(f"Relative reconstruction error: {rre:.6f}")
    print(f"Low-rank component shape: {L.shape}")
    print(f"Sparse component shape: {S.shape}")
    
    if rre < 0.1:
        print("✓ Test passed! Low reconstruction error.")
    else:
        print("⚠ Warning: Reconstruction error is high.")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_synthetic_data()
