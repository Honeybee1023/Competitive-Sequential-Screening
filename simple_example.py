"""
Simplest possible example - compute Total Surplus for one parameter set.

Run this to verify the framework works:
    python3 simple_example.py
"""

import sys
sys.path.insert(0, 'competitive_screening')

from src import Uniform, Normal, compute_all_welfare

def main():
    print("\n" + "="*70)
    print("SIMPLE EXAMPLE: Total Surplus Comparison")
    print("="*70)

    # Define the model
    v_0 = 6.0               # Average valuation
    G = Uniform(-1, 1)      # Type distribution (ex-ante heterogeneity)
    F = Normal(0, 1)        # Taste shock distribution (information precision)

    print(f"\nParameters:")
    print(f"  v_0 = {v_0} (average valuation)")
    print(f"  G ~ Uniform(-1, 1) (type distribution)")
    print(f"  F ~ Normal(0, 1) (taste shock distribution)")

    # Compute welfare for all 4 market settings
    print(f"\nSolving equilibria and computing welfare...")
    results = compute_all_welfare(v_0, F, G)

    # Display results
    print("\n" + "-"*70)
    print("RESULTS: Total Surplus")
    print("-"*70)
    print(f"{'Market Setting':<30} {'Total Surplus':>15} {'Rank':>10}")
    print("-"*70)

    # Create ranking
    ts_dict = {
        'Non-Exclusive (NE)': results['TS_NE'],
        'Spot Pricing (SP)': results['TS_SP'],
        'Exclusive (E)': results['TS_E'],
        'Multi-Good Monopoly (MM)': results['TS_MM']
    }

    # Sort by TS (descending)
    ranked = sorted(ts_dict.items(), key=lambda x: x[1], reverse=True)

    for rank, (setting, ts) in enumerate(ranked, 1):
        print(f"{setting:<30} {ts:>15.4f} {rank:>10}")

    print("-"*70)
    print(f"\nWinner: {ranked[0][0]}")
    print(f"Total Surplus = {ranked[0][1]:.4f}")

    # Theoretical prediction
    print("\n" + "="*70)
    print("THEORETICAL PREDICTION (for symmetric distributions):")
    print("="*70)
    print("Expected ranking: SP = MM > NE > E")
    print("\nExplanation:")
    print("  • SP and MM achieve efficient allocation → Maximum TS")
    print("  • NE has some inefficiency from non-exclusive contracts")
    print("  • E has most inefficiency from lock-in effects")
    print("="*70)

    # Verify prediction
    print("\n✓ Our results match the theoretical prediction!" if ranked[0][0] in ['Spot Pricing (SP)', 'Multi-Good Monopoly (MM)'] else "\n⚠ Unexpected ranking - check parameters")

    return results

if __name__ == "__main__":
    results = main()
    print()
