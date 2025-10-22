import unittest
import numpy as np
from godec import godec


class TestGoDec(unittest.TestCase):
    """Comprehensive test suite for GoDec algorithm"""
    
    def setUp(self):
        np.random.seed(42)
    
    def test_basic_decomposition(self):
        """Test 1: Basic Decomposition with known low-rank + sparse structure"""
        m, n = 100, 50
        true_rank = 5
        sparsity = 0.05
        
        U = np.random.randn(m, true_rank)
        V = np.random.randn(true_rank, n)
        L_true = U @ V
        
        num_sparse = int(m * n * sparsity)
        S_true = np.zeros((m, n))
        sparse_idx = np.random.choice(m * n, num_sparse, replace=False)
        S_true.flat[sparse_idx] = np.random.randn(num_sparse) * 5
        
        X = L_true + S_true
        
        L, S, RMSE, error = godec(X, rank=true_rank, card=num_sparse, power=1)
        
        self.assertEqual(L.shape, X.shape, "L shape should match input")
        self.assertEqual(S.shape, X.shape, "S shape should match input")
        
        self.assertLess(error, 0.1, "Reconstruction error should be small")
        
        reconstruction = L + S
        reconstruction_error = np.linalg.norm(reconstruction - X) / np.linalg.norm(X)
        self.assertLess(reconstruction_error, 0.1, "Reconstruction should be accurate")
        
        sparse_count = np.sum(np.abs(S) > 1e-10)
        self.assertLessEqual(sparse_count, num_sparse * 1.01, 
                            "Sparsity constraint should be approximately satisfied")
    
    def test_convergence(self):
        """Test 2: Convergence - RMSE should decrease or stabilize"""
        m, n = 80, 60
        true_rank = 3
        
        U = np.random.randn(m, true_rank)
        V = np.random.randn(true_rank, n)
        L_true = U @ V
        
        num_sparse = int(m * n * 0.03)
        S_true = np.zeros((m, n))
        sparse_idx = np.random.choice(m * n, num_sparse, replace=False)
        S_true.flat[sparse_idx] = np.random.randn(num_sparse) * 3
        
        X = L_true + S_true
        
        L, S, RMSE, error = godec(X, rank=true_rank, card=num_sparse, power=1)
        
        self.assertGreater(len(RMSE), 0, "Should have at least one RMSE value")
        
        for i in range(1, len(RMSE)):
            ratio = RMSE[i] / RMSE[i-1]
            self.assertLessEqual(ratio, 1.1, 
                                f"RMSE should not increase significantly (iteration {i})")
    
    def test_transpose_handling(self):
        """Test 3: Transpose Handling - Test with wide matrices (m < n)"""
        m, n = 30, 100
        true_rank = 3
        
        U = np.random.randn(m, true_rank)
        V = np.random.randn(true_rank, n)
        L_true = U @ V
        
        num_sparse = int(m * n * 0.02)
        S_true = np.zeros((m, n))
        sparse_idx = np.random.choice(m * n, num_sparse, replace=False)
        S_true.flat[sparse_idx] = np.random.randn(num_sparse) * 4
        
        X = L_true + S_true
        
        L, S, RMSE, error = godec(X, rank=true_rank, card=num_sparse, power=1)
        
        self.assertEqual(L.shape, (m, n), "L should maintain original shape (m < n)")
        self.assertEqual(S.shape, (m, n), "S should maintain original shape (m < n)")
        
        reconstruction_error = np.linalg.norm(L + S - X) / np.linalg.norm(X)
        self.assertLess(reconstruction_error, 0.1, 
                       "Reconstruction should be accurate for wide matrices")
    
    def test_power_parameter(self):
        """Test 4: Power Parameter - Higher power should improve or maintain accuracy"""
        m, n = 60, 40
        true_rank = 4
        
        U = np.random.randn(m, true_rank)
        V = np.random.randn(true_rank, n)
        L_true = U @ V
        
        num_sparse = int(m * n * 0.04)
        S_true = np.zeros((m, n))
        sparse_idx = np.random.choice(m * n, num_sparse, replace=False)
        S_true.flat[sparse_idx] = np.random.randn(num_sparse) * 3.5
        
        X = L_true + S_true
        
        _, _, _, error_power0 = godec(X, rank=true_rank, card=num_sparse, power=0)
        _, _, _, error_power1 = godec(X, rank=true_rank, card=num_sparse, power=1)
        _, _, _, error_power2 = godec(X, rank=true_rank, card=num_sparse, power=2)
        
        self.assertLess(error_power0, 0.15, "Power=0 should give reasonable accuracy")
        self.assertLess(error_power1, 0.15, "Power=1 should give reasonable accuracy")
        self.assertLess(error_power2, 0.15, "Power=2 should give reasonable accuracy")
        
        self.assertLessEqual(error_power2, error_power0 * 1.2, 
                            "Higher power should not significantly worsen accuracy")
    
    def test_rank_parameter(self):
        """Test 5: Rank Parameter - Test with different rank values"""
        m, n = 70, 50
        true_rank = 5
        
        U = np.random.randn(m, true_rank)
        V = np.random.randn(true_rank, n)
        L_true = U @ V
        
        num_sparse = int(m * n * 0.03)
        S_true = np.zeros((m, n))
        sparse_idx = np.random.choice(m * n, num_sparse, replace=False)
        S_true.flat[sparse_idx] = np.random.randn(num_sparse) * 3
        
        X = L_true + S_true
        
        for rank in [3, 5, 10]:
            L, S, RMSE, error = godec(X, rank=rank, card=num_sparse, power=1)
            
            self.assertEqual(L.shape, X.shape, f"L shape correct for rank={rank}")
            self.assertEqual(S.shape, X.shape, f"S shape correct for rank={rank}")
            self.assertLess(error, 0.5, f"Error reasonable for rank={rank}")
    
    def test_video_frame_format(self):
        """Test 6: Video Frame Format - Simulate video data"""
        frame_height = 48
        frame_width = 48
        num_frames = 51
        
        m = frame_height * frame_width
        n = num_frames
        
        true_rank = 3
        U = np.random.randn(m, true_rank)
        V = np.random.randn(true_rank, n)
        L_true = U @ V
        
        sparsity = 0.02
        num_sparse = int(m * n * sparsity)
        S_true = np.zeros((m, n))
        sparse_idx = np.random.choice(m * n, num_sparse, replace=False)
        S_true.flat[sparse_idx] = np.random.randn(num_sparse) * 4
        
        X = L_true + S_true
        
        L, S, RMSE, error = godec(X, rank=true_rank, card=num_sparse, power=1)
        
        self.assertEqual(L.shape, (m, n), "L should be (pixels x frames)")
        self.assertEqual(S.shape, (m, n), "S should be (pixels x frames)")
        
        L_frames = L.reshape(frame_height, frame_width, num_frames)
        S_frames = S.reshape(frame_height, frame_width, num_frames)
        
        self.assertEqual(L_frames.shape, (frame_height, frame_width, num_frames),
                        "L can be reshaped back to video format")
        self.assertEqual(S_frames.shape, (frame_height, frame_width, num_frames),
                        "S can be reshaped back to video format")
        
        self.assertLess(error, 0.1, "Video decomposition quality is good")
        
        reconstruction = L + S
        frame_errors = []
        for i in range(num_frames):
            X_frame = X[:, i].reshape(frame_height, frame_width)
            recon_frame = reconstruction[:, i].reshape(frame_height, frame_width)
            frame_error = np.linalg.norm(recon_frame - X_frame) / np.linalg.norm(X_frame)
            frame_errors.append(frame_error)
        
        avg_frame_error = np.mean(frame_errors)
        self.assertLess(avg_frame_error, 0.15, 
                       "Average per-frame reconstruction error should be small")


if __name__ == '__main__':
    unittest.main()
