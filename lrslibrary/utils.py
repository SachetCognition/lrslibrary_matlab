import numpy as np
from typing import Tuple

def subsampling(M: np.ndarray, observation_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    total_entries = M.size
    num_observed = int(total_entries * observation_ratio)
    
    Idx = np.random.choice(total_entries, size=num_observed, replace=False)
    Idx = np.sort(Idx)
    
    Omega = np.zeros_like(M)
    Omega.ravel()[Idx] = 1
    
    return Idx, Omega

def convert_video3d_to_2d(V: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
    if V.ndim != 3:
        raise ValueError("Input must be a 3D array (height x width x frames)")
    
    m, n, p = V.shape
    M = V.reshape(m * n, p, order='F')
    
    return M, m, n, p

def validate_input_matrix(M: np.ndarray, name: str = "M") -> None:
    """
    Validate input matrix for common issues.
    
    Args:
        M: Input matrix to validate
        name: Name of the matrix for error messages
        
    Raises:
        ValueError: If matrix has issues (NaN, Inf, wrong dimensions)
    """
    if not isinstance(M, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(M)}")
    
    if M.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got shape {M.shape}")
    
    if M.size == 0:
        raise ValueError(f"{name} cannot be empty")
    
    if np.any(np.isnan(M)):
        raise ValueError(f"{name} contains NaN values")
    
    if np.any(np.isinf(M)):
        raise ValueError(f"{name} contains Inf values")


def validate_rank(rank: int, M: np.ndarray, name: str = "rank") -> None:
    """
    Validate rank parameter.
    
    Args:
        rank: Desired rank
        M: Input matrix
        name: Name of the parameter for error messages
        
    Raises:
        ValueError: If rank is invalid
    """
    if not isinstance(rank, int):
        raise TypeError(f"{name} must be an integer, got {type(rank)}")
    
    if rank < 1:
        raise ValueError(f"{name} must be positive, got {rank}")
    
    max_rank = min(M.shape)
    if rank > max_rank:
        raise ValueError(f"{name}={rank} exceeds maximum possible rank {max_rank} for matrix shape {M.shape}")


def validate_positive_param(value: float, name: str, allow_zero: bool = False) -> None:
    """
    Validate that a parameter is positive.
    
    Args:
        value: Parameter value
        name: Parameter name for error messages
        allow_zero: Whether to allow zero values
        
    Raises:
        ValueError: If value is invalid
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value)}")
    
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
