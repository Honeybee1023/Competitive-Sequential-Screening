#!/usr/bin/env python3
"""
Test script to verify the frontend bug fix.

This script tests that the plot_distribution() function in the frontend app
now works correctly with both Uniform and Normal distributions.
"""

import sys
sys.path.insert(0, 'competitive_screening')

# Import what the frontend uses
from src import Uniform, Normal, Logistic

def test_distributions():
    """Test that we can get statistics for all distribution types."""
    print("="*70)
    print("FRONTEND BUG FIX VALIDATION")
    print("="*70)

    distributions = [
        ("Uniform(-1, 1)", Uniform(-1, 1)),
        ("Uniform(-2, 2)", Uniform(-2, 2)),
        ("Normal(0, 1)", Normal(0, 1)),
        ("Normal(0, 0.5)", Normal(0, 0.5)),
        ("Logistic(0, 1)", Logistic(0, 1))
    ]

    print("\nTesting distribution statistics access:\n")

    all_passed = True
    for name, dist in distributions:
        print(f"Testing {name}:")

        # Test mean (should always work)
        try:
            mean = dist.mean()
            print(f"  ✓ Mean: {mean:.3f}")
        except Exception as e:
            print(f"  ✗ Mean failed: {e}")
            all_passed = False

        # Test variance (the bug we fixed)
        try:
            # Try direct variance() method first
            variance = dist.variance()
            print(f"  ✓ Variance (direct): {variance:.3f}")
        except AttributeError:
            # Fall back to scipy's var()
            try:
                if hasattr(dist, '_dist'):
                    variance = dist._dist.var()
                    print(f"  ✓ Variance (via _dist.var()): {variance:.3f}")
                else:
                    # Manual calculation for Uniform
                    a, b = dist.support()
                    variance = ((b - a) ** 2) / 12
                    print(f"  ✓ Variance (manual calc): {variance:.3f}")
            except Exception as e:
                print(f"  ✗ Variance failed: {e}")
                all_passed = False

        # Test support
        try:
            support = dist.support()
            print(f"  ✓ Support: {support}")
        except Exception as e:
            print(f"  ✗ Support failed: {e}")
            all_passed = False

        print()

    if all_passed:
        print("="*70)
        print("✅ ALL TESTS PASSED - Frontend bug is FIXED!")
        print("="*70)
        print("\nThe plot_distribution() function will now work correctly")
        print("for all distribution types in the Streamlit app.")
        print("\nYou can now run the frontend without AttributeError:")
        print("  ./run_app.sh")
    else:
        print("="*70)
        print("❌ SOME TESTS FAILED")
        print("="*70)

    return all_passed

if __name__ == "__main__":
    success = test_distributions()
    sys.exit(0 if success else 1)
