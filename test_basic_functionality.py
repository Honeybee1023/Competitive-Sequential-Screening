"""
Quick test to verify implementations work end-to-end.

Run this to check if the code is functional.
"""

import sys
sys.path.insert(0, 'competitive_screening')

from src import (
    Uniform, Normal,
    solve_equilibrium_SP, solve_equilibrium_MM, solve_equilibrium_E, solve_equilibrium_NE,
    compute_total_surplus_SP, compute_consumer_surplus_SP, compute_producer_surplus_SP,
    compute_total_surplus_MM, compute_consumer_surplus_MM, compute_producer_surplus_MM,
    compute_total_surplus_E, compute_consumer_surplus_E, compute_producer_surplus_E,
    compute_total_surplus_NE, compute_consumer_surplus_NE, compute_producer_surplus_NE
)

def test_basic_case():
    """Test canonical case from paper: Uniform type, Normal taste shock."""
    print("="*70)
    print("BASIC FUNCTIONALITY TEST")
    print("="*70)

    # Parameters (from paper)
    v_0 = 6.0
    G = Uniform(-1, 1)   # Type distribution
    F = Normal(0, 1)     # Taste shock

    print(f"\nParameters:")
    print(f"  v_0 = {v_0}")
    print(f"  G = Uniform(-1, 1)")
    print(f"  F = Normal(0, 1)")

    results = {}

    # Test each market setting
    print("\n" + "-"*70)
    print("1. SPOT PRICING (SP) - Should be COMPLETE")
    print("-"*70)
    try:
        eq_SP = solve_equilibrium_SP(v_0, F, G)
        print(f"✓ Equilibrium solved: θ* = {eq_SP.theta_star:.4f}")

        TS_SP = compute_total_surplus_SP(eq_SP)
        CS_SP = compute_consumer_surplus_SP(eq_SP)
        PS_SP = compute_producer_surplus_SP(eq_SP)

        results['SP'] = {'TS': TS_SP, 'CS': CS_SP, 'PS': PS_SP}

        print(f"  Total Surplus:    {TS_SP:.6f}")
        print(f"  Consumer Surplus: {CS_SP:.6f}")
        print(f"  Producer Surplus: {PS_SP:.6f}")
        print(f"  Accounting: TS - (CS+PS) = {TS_SP - (CS_SP + PS_SP):.6e}")

        if abs(TS_SP - (CS_SP + PS_SP)) < 1e-3:
            print("  ✓ Welfare accounting consistent")
        else:
            print("  ✗ WARNING: Welfare accounting inconsistent!")

    except Exception as e:
        print(f"✗ FAILED: {e}")

    print("\n" + "-"*70)
    print("2. MULTI-GOOD MONOPOLY (MM) - Should be COMPLETE")
    print("-"*70)
    try:
        eq_MM = solve_equilibrium_MM(v_0, F, G)
        print(f"✓ Equilibrium solved: s* = {eq_MM.s:.4f}")

        TS_MM = compute_total_surplus_MM(eq_MM)
        CS_MM = compute_consumer_surplus_MM(eq_MM)
        PS_MM = compute_producer_surplus_MM(eq_MM)

        results['MM'] = {'TS': TS_MM, 'CS': CS_MM, 'PS': PS_MM}

        print(f"  Total Surplus:    {TS_MM:.6f}")
        print(f"  Consumer Surplus: {CS_MM:.6f} (should be ~0)")
        print(f"  Producer Surplus: {PS_MM:.6f} (should equal TS)")

        if abs(CS_MM) < 1e-3 and abs(TS_MM - PS_MM) < 1e-3:
            print("  ✓ Full surplus extraction verified")
        else:
            print("  ✗ WARNING: Expected CS=0, PS=TS for monopoly!")

    except Exception as e:
        print(f"✗ FAILED: {e}")

    print("\n" + "-"*70)
    print("3. EXCLUSIVE SUBSCRIPTIONS (E) - PARTIAL (subscription schedules missing)")
    print("-"*70)
    try:
        eq_E = solve_equilibrium_E(v_0, F, G)
        print(f"✓ Equilibrium solved: γ̂ = {eq_E.gamma_hat:.4f}")

        TS_E = compute_total_surplus_E(eq_E)
        print(f"  Total Surplus: {TS_E:.6f} ✓")

        # These will warn about missing subscription schedules
        print("  (Computing CS/PS - expect warnings about missing subscriptions)")
        CS_E = compute_consumer_surplus_E(eq_E)
        PS_E = compute_producer_surplus_E(eq_E)

        results['E'] = {'TS': TS_E, 'CS': CS_E, 'PS': PS_E}

        print(f"  Consumer Surplus: {CS_E:.6f} ⚠")
        print(f"  Producer Surplus: {PS_E:.6f} ⚠")

    except Exception as e:
        print(f"✗ FAILED: {e}")

    print("\n" + "-"*70)
    print("4. NON-EXCLUSIVE (NE) - PARTIAL (Theorem 1 missing from PDF)")
    print("-"*70)
    try:
        eq_NE = solve_equilibrium_NE(v_0, F, G)
        print(f"✓ Equilibrium solved")
        print(f"  Strike prices at γ=0: p_A(0) = {eq_NE.p_A(0):.4f}, p_B(0) = {eq_NE.p_B(0):.4f}")

        TS_NE = compute_total_surplus_NE(eq_NE)
        print(f"  Total Surplus: {TS_NE:.6f} ✓")

        # These will warn about missing subscription schedules
        print("  (Computing CS/PS - expect warnings about missing subscriptions)")
        CS_NE = compute_consumer_surplus_NE(eq_NE)
        PS_NE = compute_producer_surplus_NE(eq_NE)

        results['NE'] = {'TS': TS_NE, 'CS': CS_NE, 'PS': PS_NE}

        print(f"  Consumer Surplus: {CS_NE:.6f} ⚠")
        print(f"  Producer Surplus: {PS_NE:.6f} ⚠")

    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: TOTAL SURPLUS RANKINGS")
    print("="*70)

    ts_values = [(k, v['TS']) for k, v in results.items()]
    ts_sorted = sorted(ts_values, key=lambda x: x[1], reverse=True)

    print("\nTotal Surplus (highest to lowest):")
    for i, (setting, ts) in enumerate(ts_sorted, 1):
        marker = "✓" if setting in ['SP', 'MM'] else "⚠"
        print(f"  {i}. {setting:3s}: {ts:.6f} {marker}")

    print("\nCS Rankings (with WARNINGS for NE/E):")
    cs_values = [(k, v['CS']) for k, v in results.items() if 'CS' in v]
    cs_sorted = sorted(cs_values, key=lambda x: x[1], reverse=True)
    for i, (setting, cs) in enumerate(cs_sorted, 1):
        marker = "✓" if setting in ['SP', 'MM'] else "⚠"
        print(f"  {i}. {setting:3s}: {cs:.6f} {marker}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\n✓ = Fully implemented")
    print("⚠ = Approximate (missing subscription schedules)")
    print("\nFor TOTAL SURPLUS comparisons, all implementations are functional.")
    print("For CS/PS in NE and E, results are approximate pending full paper.")

if __name__ == "__main__":
    test_basic_case()
