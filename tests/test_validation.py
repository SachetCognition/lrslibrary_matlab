import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/lrslibrary_python')

from lrslibrary import run_algorithm

def test_godec_validation():
    """Test GoDec input validation."""
    M = np.random.randn(100, 50)
    
    print("Testing GoDec validation...")
    
    try:
        run_algorithm('RPCA', 'GoDec', M, params={'rank': 0})
        print("  ERROR: Should have raised ValueError for rank=0")
        return False
    except ValueError as e:
        if "rank" in str(e).lower() and "positive" in str(e).lower():
            print("  ✓ Correctly rejected rank=0")
        else:
            print(f"  ERROR: Wrong error message: {e}")
            return False
    
    try:
        run_algorithm('RPCA', 'GoDec', M, params={'rank': 200})
        print("  ERROR: Should have raised ValueError for rank=200")
        return False
    except ValueError as e:
        if "rank" in str(e).lower() and "exceeds" in str(e).lower():
            print("  ✓ Correctly rejected rank=200 (exceeds max)")
        else:
            print(f"  ERROR: Wrong error message: {e}")
            return False
    
    M_nan = M.copy()
    M_nan[0, 0] = np.nan
    try:
        run_algorithm('RPCA', 'GoDec', M_nan, params={'rank': 1})
        print("  ERROR: Should have raised ValueError for NaN input")
        return False
    except ValueError as e:
        if "nan" in str(e).lower():
            print("  ✓ Correctly rejected NaN input")
        else:
            print(f"  ERROR: Wrong error message: {e}")
            return False
    
    M_inf = M.copy()
    M_inf[0, 0] = np.inf
    try:
        run_algorithm('RPCA', 'GoDec', M_inf, params={'rank': 1})
        print("  ERROR: Should have raised ValueError for Inf input")
        return False
    except ValueError as e:
        if "inf" in str(e).lower():
            print("  ✓ Correctly rejected Inf input")
        else:
            print(f"  ERROR: Wrong error message: {e}")
            return False
    
    print("GoDec validation tests PASSED\n")
    return True

def test_svt_validation():
    """Test SVT input validation."""
    print("Testing SVT validation...")
    
    M = np.random.randn(100, 50)
    n = M.shape
    
    try:
        from lrslibrary.algorithms.mc.svt.svt import svt
        svt(n, np.array([]), np.array([]), tau=1.0, delta=1.2)
        print("  ERROR: Should have raised ValueError for empty Omega")
        return False
    except ValueError as e:
        if "cannot be empty" in str(e).lower():
            print("  ✓ Correctly rejected empty Omega")
        else:
            print(f"  ERROR: Wrong error message: {e}")
            return False
    
    try:
        from lrslibrary.algorithms.mc.svt.svt import svt
        Omega = np.array([0, 1, 2])
        b = np.array([1.0, 2.0])
        svt(n, Omega, b, tau=1.0, delta=1.2)
        print("  ERROR: Should have raised ValueError for mismatched lengths")
        return False
    except ValueError as e:
        if "must match" in str(e).lower():
            print("  ✓ Correctly rejected mismatched Omega and b lengths")
        else:
            print(f"  ERROR: Wrong error message: {e}")
            return False
    
    print("SVT validation tests PASSED\n")
    return True

def test_reprocs_validation():
    """Test ReProCS input validation."""
    print("Testing ReProCS validation...")
    
    M = np.random.randn(100, 40)
    L_init_wrong = np.random.randn(50, 5)
    mu = np.mean(M, axis=1)
    
    try:
        from lrslibrary.algorithms.st.reprocs.reprocs import reprocs
        reprocs(M, L_init_wrong, mu, ev_thresh=0.1, alpha=1.0, K=10)
        print("  ERROR: Should have raised ValueError for mismatched L_init shape")
        return False
    except ValueError as e:
        if "must match" in str(e).lower():
            print("  ✓ Correctly rejected mismatched L_init shape")
        else:
            print(f"  ERROR: Wrong error message: {e}")
            return False
    
    print("ReProCS validation tests PASSED\n")
    return True

if __name__ == '__main__':
    all_passed = True
    
    try:
        all_passed = test_godec_validation() and all_passed
    except Exception as e:
        print(f"GoDec validation test FAILED with exception: {e}\n")
        all_passed = False
    
    try:
        all_passed = test_svt_validation() and all_passed
    except Exception as e:
        print(f"SVT validation test FAILED with exception: {e}\n")
        all_passed = False
    
    try:
        all_passed = test_reprocs_validation() and all_passed
    except Exception as e:
        print(f"ReProCS validation test FAILED with exception: {e}\n")
        all_passed = False
    
    if all_passed:
        print("="*50)
        print("ALL VALIDATION TESTS PASSED!")
        print("="*50)
    else:
        print("="*50)
        print("SOME VALIDATION TESTS FAILED")
        print("="*50)
        sys.exit(1)
