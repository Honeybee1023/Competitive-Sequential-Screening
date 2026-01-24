"""
Example: Parameter sweep and visualization for Total Surplus analysis.

This script demonstrates how to:
1. Run parameter sweeps over v_0, σ, and G width
2. Create visualizations
3. Analyze ranking transitions
"""

import sys
sys.path.insert(0, '../competitive_screening')

from src import Uniform, Normal
from src.analysis import (
    sweep_v0,
    sweep_information_precision,
    sweep_type_distribution_width,
    compare_distributions
)
from src.analysis.visualization import (
    plot_ts_comparison,
    plot_ts_rankings,
    plot_ts_differences,
    plot_efficiency_ratios,
    create_dashboard
)
import matplotlib.pyplot as plt


def example_1_sweep_v0():
    """Example 1: Sweep over average valuation v_0."""
    print("="*70)
    print("EXAMPLE 1: Sweeping over v_0")
    print("="*70)

    # Fixed distributions
    G = Uniform(-1, 1)
    F = Normal(0, 1)

    # Sweep over v_0 values
    v0_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    print(f"\nSweeping v_0 from {min(v0_values)} to {max(v0_values)}")
    print(f"G = Uniform(-1, 1), F = Normal(0, 1)\n")

    result = sweep_v0(v0_values, F, G, verbose=True)

    # Create visualizations
    print("\nCreating visualizations...")

    fig1 = plot_ts_comparison(result, title="Total Surplus vs. Average Valuation")
    fig2 = plot_ts_rankings(result, title="TS Rankings vs. v_0")
    fig3 = plot_efficiency_ratios(result, title="Efficiency vs. v_0")

    # Save figures
    fig1.savefig('output_v0_comparison.png', dpi=300, bbox_inches='tight')
    fig2.savefig('output_v0_rankings.png', dpi=300, bbox_inches='tight')
    fig3.savefig('output_v0_efficiency.png', dpi=300, bbox_inches='tight')

    print("✓ Saved: output_v0_comparison.png")
    print("✓ Saved: output_v0_rankings.png")
    print("✓ Saved: output_v0_efficiency.png")

    # Analyze results
    print("\n" + "-"*70)
    print("ANALYSIS")
    print("-"*70)

    for i, v0 in enumerate(v0_values):
        ranking = result.get_ranking(i)
        print(f"v_0={v0:4.1f}: {[s for s,_ in ranking]} "
              f"(TS: {', '.join(f'{v:.2f}' for _,v in ranking)})")

    return result


def example_2_sweep_information():
    """Example 2: Sweep over information precision (σ)."""
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Sweeping over Information Precision (σ)")
    print("="*70)

    # Fixed parameters
    v0 = 6.0
    G = Uniform(-1, 1)

    # Sweep over σ values (lower σ = more precise signal)
    sigma_values = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    print(f"\nSweeping σ from {min(sigma_values)} to {max(sigma_values)}")
    print("Lower σ = more informative signal, Higher σ = less informative")
    print(f"v_0 = {v0}, G = Uniform(-1, 1)\n")

    result = sweep_information_precision(sigma_values, v0, G, verbose=True)

    # Create visualizations
    print("\nCreating visualizations...")

    fig = plot_ts_comparison(result,
                            title="Total Surplus vs. Information Precision",
                            xlabel="σ (taste shock std dev)")
    fig.savefig('output_sigma_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: output_sigma_comparison.png")

    # Analyze
    print("\n" + "-"*70)
    print("ANALYSIS: How rankings change with information")
    print("-"*70)

    for i, sigma in enumerate(sigma_values):
        ranking = result.get_ranking(i)
        winner = ranking[0][0]
        print(f"σ={sigma:4.1f}: Winner = {winner}, "
              f"Rankings = {[s for s,_ in ranking]}")

    return result


def example_3_sweep_type_width():
    """Example 3: Sweep over type distribution width."""
    print("\n\n" + "="*70)
    print("EXAMPLE 3: Sweeping over Type Distribution Width")
    print("="*70)

    # Fixed parameters
    v0 = 6.0
    F = Normal(0, 1)

    # Sweep over G width (G ~ Uniform[-w, w])
    width_values = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]

    print(f"\nSweeping G width from {min(width_values)} to {max(width_values)}")
    print("G ~ Uniform[-w, w], larger w = more ex-ante heterogeneity")
    print(f"v_0 = {v0}, F = Normal(0, 1)\n")

    result = sweep_type_distribution_width(width_values, v0, F, center=0.0, verbose=True)

    # Create visualizations
    print("\nCreating visualizations...")

    fig = plot_ts_comparison(result,
                            title="Total Surplus vs. Ex-Ante Heterogeneity",
                            xlabel="Type distribution half-width")
    fig.savefig('output_width_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: output_width_comparison.png")

    return result


def example_4_compare_scenarios():
    """Example 4: Compare specific distribution scenarios."""
    print("\n\n" + "="*70)
    print("EXAMPLE 4: Comparing Distribution Scenarios")
    print("="*70)

    scenarios = {
        'Baseline': (6, Normal(0, 1), Uniform(-1, 1)),
        'High Information': (6, Normal(0, 0.5), Uniform(-1, 1)),
        'Low Information': (6, Normal(0, 2), Uniform(-1, 1)),
        'Narrow Types': (6, Normal(0, 1), Uniform(-0.5, 0.5)),
        'Wide Types': (6, Normal(0, 1), Uniform(-2, 2)),
    }

    print("\nComparing scenarios:")
    for name, (v0, F, G) in scenarios.items():
        print(f"  {name}: v0={v0}, F~N(0,{F.sigma}²), G~U{G.support()}")

    results = compare_distributions(scenarios, verbose=True)

    # Create comparison table
    print("\n" + "-"*70)
    print("RESULTS TABLE")
    print("-"*70)
    print(f"{'Scenario':<20} {'NE':>8} {'SP':>8} {'E':>8} {'MM':>8} {'Winner':>8}")
    print("-"*70)

    for name in scenarios.keys():
        ts = results[name]
        winner = max(ts.items(), key=lambda x: x[1])[0]
        print(f"{name:<20} {ts['NE']:>8.3f} {ts['SP']:>8.3f} "
              f"{ts['E']:>8.3f} {ts['MM']:>8.3f} {winner:>8}")

    return results


def example_5_create_dashboard():
    """Example 5: Create a comprehensive dashboard."""
    print("\n\n" + "="*70)
    print("EXAMPLE 5: Creating Comprehensive Dashboard")
    print("="*70)

    # Fixed baseline
    G = Uniform(-1, 1)
    F = Normal(0, 1)
    v0 = 6.0

    # Run multiple sweeps
    print("\nRunning multiple parameter sweeps...")

    sweep_results = {
        'Effect of v_0': sweep_v0([2, 4, 6, 8, 10], F, G, verbose=False),
        'Effect of σ': sweep_information_precision([0.5, 1.0, 2.0, 5.0], v0, G, verbose=False),
        'Effect of Type Heterogeneity': sweep_type_distribution_width([0.5, 1.0, 2.0], v0, F, verbose=False)
    }

    print("✓ Sweeps complete")

    # Create dashboard
    print("\nCreating dashboard...")
    fig = create_dashboard(sweep_results)
    fig.savefig('output_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: output_dashboard.png")

    return sweep_results


def example_6_efficiency_analysis():
    """Example 6: Detailed efficiency analysis."""
    print("\n\n" + "="*70)
    print("EXAMPLE 6: Efficiency Analysis")
    print("="*70)

    v0 = 6.0
    G = Uniform(-1, 1)
    F = Normal(0, 1)

    # Sweep and analyze
    sigma_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    result = sweep_information_precision(sigma_values, v0, G, verbose=False)

    print("\nEfficiency as % of MM (efficient benchmark):")
    print("-"*70)
    print(f"{'σ':>6} {'NE %':>8} {'SP %':>8} {'E %':>8}")
    print("-"*70)

    for i, sigma in enumerate(sigma_values):
        eff_NE = result.TS_NE[i] / result.TS_MM[i] * 100
        eff_SP = result.TS_SP[i] / result.TS_MM[i] * 100
        eff_E = result.TS_E[i] / result.TS_MM[i] * 100

        print(f"{sigma:6.1f} {eff_NE:7.2f}% {eff_SP:7.2f}% {eff_E:7.2f}%")

    # Create efficiency plot
    fig = plot_efficiency_ratios(result,
                                 title="Efficiency vs. Information Precision")
    fig.savefig('output_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: output_efficiency_analysis.png")

    return result


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PARAMETER SWEEP AND VISUALIZATION EXAMPLES")
    print("="*70)
    print("\nThis script will:")
    print("  1. Sweep over v_0 (average valuation)")
    print("  2. Sweep over σ (information precision)")
    print("  3. Sweep over G width (ex-ante heterogeneity)")
    print("  4. Compare specific scenarios")
    print("  5. Create comprehensive dashboard")
    print("  6. Perform efficiency analysis")
    print("\nAll figures will be saved to the current directory.")
    print("="*70)

    # Run examples
    result1 = example_1_sweep_v0()
    result2 = example_2_sweep_information()
    result3 = example_3_sweep_type_width()
    result4 = example_4_compare_scenarios()
    result5 = example_5_create_dashboard()
    result6 = example_6_efficiency_analysis()

    print("\n\n" + "="*70)
    print("ALL EXAMPLES COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - output_v0_comparison.png")
    print("  - output_v0_rankings.png")
    print("  - output_v0_efficiency.png")
    print("  - output_sigma_comparison.png")
    print("  - output_width_comparison.png")
    print("  - output_dashboard.png")
    print("  - output_efficiency_analysis.png")
    print("\nTo view: open the PNG files in your image viewer")
    print("="*70)

    # Show one plot as example
    plt.show()


if __name__ == "__main__":
    main()
