"""Quick test of Total Surplus parameter sweep functionality."""

import sys
sys.path.insert(0, 'competitive_screening')

from src import Uniform, Normal
from src.analysis import sweep_v0, sweep_information_precision
from src.analysis.visualization import plot_ts_comparison
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("=" * 70)
print("QUICK TOTAL SURPLUS SWEEP TEST")
print("=" * 70)

# Test 1: Sweep over v_0
print("\nTest 1: Sweeping v_0 from 4 to 8...")
G = Uniform(-1, 1)
F = Normal(0, 1)
v0_values = [4, 5, 6, 7, 8]

result1 = sweep_v0(v0_values, F, G, verbose=False)

print("\nResults:")
print(f"{'v_0':>6} {'NE':>8} {'SP':>8} {'E':>8} {'MM':>8} {'Winner':>8}")
print("-" * 55)
for i, v0 in enumerate(v0_values):
    ranking = result1.get_ranking(i)
    winner = ranking[0][0]
    print(f"{v0:6.1f} {result1.TS_NE[i]:8.3f} {result1.TS_SP[i]:8.3f} "
          f"{result1.TS_E[i]:8.3f} {result1.TS_MM[i]:8.3f} {winner:>8}")

# Create plot
print("\nCreating visualization...")
fig = plot_ts_comparison(result1, title="Total Surplus vs. v_0")
fig.savefig('test_v0_sweep.png', dpi=300, bbox_inches='tight')
print("✓ Saved: test_v0_sweep.png")

# Test 2: Sweep over information precision
print("\n" + "=" * 70)
print("Test 2: Sweeping information precision (σ)...")
sigma_values = [0.5, 1.0, 2.0, 3.0]
v0 = 6.0

result2 = sweep_information_precision(sigma_values, v0, G, verbose=False)

print("\nResults:")
print(f"{'σ':>6} {'NE':>8} {'SP':>8} {'E':>8} {'MM':>8} {'Winner':>8}")
print("-" * 55)
for i, sigma in enumerate(sigma_values):
    ranking = result2.get_ranking(i)
    winner = ranking[0][0]
    print(f"{sigma:6.1f} {result2.TS_NE[i]:8.3f} {result2.TS_SP[i]:8.3f} "
          f"{result2.TS_E[i]:8.3f} {result2.TS_MM[i]:8.3f} {winner:>8}")

# Create plot
print("\nCreating visualization...")
fig = plot_ts_comparison(result2, title="Total Surplus vs. σ",
                         xlabel="σ (taste shock std dev)")
fig.savefig('test_sigma_sweep.png', dpi=300, bbox_inches='tight')
print("✓ Saved: test_sigma_sweep.png")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)
print("\nGenerated files:")
print("  - test_v0_sweep.png")
print("  - test_sigma_sweep.png")
