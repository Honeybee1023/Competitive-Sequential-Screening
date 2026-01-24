"""
Basic usage example for the competitive screening framework.

NOTE: This will not run correctly until placeholders are filled.
"""

import sys
sys.path.append('..')

from src import (
    Uniform, Normal,
    solve_equilibrium_NE, solve_equilibrium_SP, solve_equilibrium_E, solve_equilibrium_MM,
    compute_total_surplus_NE, compute_consumer_surplus_NE, compute_producer_surplus_NE,
    compute_total_surplus_SP, compute_consumer_surplus_SP, compute_producer_surplus_SP,
    compute_total_surplus_E, compute_consumer_surplus_E, compute_producer_surplus_E,
    compute_total_surplus_MM, compute_consumer_surplus_MM, compute_producer_surplus_MM,
    validate_parameters
)


def main():
    """
    Canonical example: Uniform type distribution, Normal taste shock.

    This should replicate the baseline case from the paper.
    """

    # Parameters (from paper)
    v_0 = 6.0
    G = Uniform(-1, 1)   # Type distribution: γ ~ U[-1, 1]
    F = Normal(0, 1)     # Taste shock: ε ~ N(0, 1)

    print("=" * 70)
    print("Competitive Sequential Screening - Welfare Analysis")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  v_0 = {v_0}")
    print(f"  G = Uniform({G.a}, {G.b})")
    print(f"  F = Normal({F.mu}, {F.sigma})")
    print()

    # Validate assumptions
    print("Validating assumptions...")
    try:
        results = validate_parameters(v_0, F, G, strict=True)
        print("✓ All assumptions satisfied")
    except Exception as e:
        print(f"✗ Assumption violation: {e}")
        return

    print("\n" + "=" * 70)

    # =========================================================================
    # EFFICIENT API: Solve once, compute multiple metrics
    # =========================================================================

    print("\n1. NON-EXCLUSIVE SUBSCRIPTIONS (NE)")
    print("-" * 70)
    try:
        eq_NE = solve_equilibrium_NE(v_0, F, G)
        print(f"✓ Equilibrium solved (coverage: {eq_NE.is_covered})")

        TS_NE = compute_total_surplus_NE(eq_NE)
        CS_NE = compute_consumer_surplus_NE(eq_NE)
        PS_NE = compute_producer_surplus_NE(eq_NE)

        print(f"  Total Surplus:    {TS_NE:.6f}")
        print(f"  Consumer Surplus: {CS_NE:.6f}")
        print(f"  Producer Surplus: {PS_NE:.6f}")
        print(f"  Accounting check: TS - (CS + PS) = {TS_NE - (CS_NE + PS_NE):.6e}")

    except Exception as e:
        print(f"✗ Failed: {e}")

    print("\n2. SPOT PRICING (SP)")
    print("-" * 70)
    try:
        eq_SP = solve_equilibrium_SP(v_0, F, G)
        print(f"✓ Equilibrium solved (θ* = {eq_SP.theta_star:.4f})")

        TS_SP = compute_total_surplus_SP(eq_SP)
        CS_SP = compute_consumer_surplus_SP(eq_SP)
        PS_SP = compute_producer_surplus_SP(eq_SP)

        print(f"  Total Surplus:    {TS_SP:.6f}")
        print(f"  Consumer Surplus: {CS_SP:.6f}")
        print(f"  Producer Surplus: {PS_SP:.6f}")
        print(f"  Accounting check: TS - (CS + PS) = {TS_SP - (CS_SP + PS_SP):.6e}")

    except Exception as e:
        print(f"✗ Failed: {e}")

    print("\n3. EXCLUSIVE SUBSCRIPTIONS (E)")
    print("-" * 70)
    try:
        eq_E = solve_equilibrium_E(v_0, F, G)
        print(f"✓ Equilibrium solved (γ̂ = {eq_E.gamma_hat:.4f})")

        TS_E = compute_total_surplus_E(eq_E)
        CS_E = compute_consumer_surplus_E(eq_E)
        PS_E = compute_producer_surplus_E(eq_E)

        print(f"  Total Surplus:    {TS_E:.6f}")
        print(f"  Consumer Surplus: {CS_E:.6f}")
        print(f"  Producer Surplus: {PS_E:.6f}")
        print(f"  Accounting check: TS - (CS + PS) = {TS_E - (CS_E + PS_E):.6e}")

    except Exception as e:
        print(f"✗ Failed: {e}")

    print("\n4. MULTI-GOOD MONOPOLY (MM)")
    print("-" * 70)
    try:
        eq_MM = solve_equilibrium_MM(v_0, F, G)
        print(f"✓ Equilibrium solved (s = {eq_MM.s:.4f})")

        TS_MM = compute_total_surplus_MM(eq_MM)
        CS_MM = compute_consumer_surplus_MM(eq_MM)
        PS_MM = compute_producer_surplus_MM(eq_MM)

        print(f"  Total Surplus:    {TS_MM:.6f}")
        print(f"  Consumer Surplus: {CS_MM:.6f} (should be 0)")
        print(f"  Producer Surplus: {PS_MM:.6f}")
        print(f"  Accounting check: TS - (CS + PS) = {TS_MM - (CS_MM + PS_MM):.6e}")

    except Exception as e:
        print(f"✗ Failed: {e}")

    # =========================================================================
    # WELFARE COMPARISONS
    # =========================================================================

    print("\n" + "=" * 70)
    print("WELFARE COMPARISONS")
    print("=" * 70)

    # Total surplus ranking (should be: MM > NE > E > SP for some parameters)
    print("\nTotal Surplus Ranking:")
    ts_values = [
        ('NE', TS_NE),
        ('SP', TS_SP),
        ('E', TS_E),
        ('MM', TS_MM)
    ]
    ts_sorted = sorted(ts_values, key=lambda x: x[1], reverse=True)
    for i, (setting, ts) in enumerate(ts_sorted, 1):
        print(f"  {i}. {setting:3s}: {ts:.6f}")

    # Consumer surplus ranking (varies with parameters)
    print("\nConsumer Surplus Ranking:")
    cs_values = [
        ('NE', CS_NE),
        ('SP', CS_SP),
        ('E', CS_E),
        ('MM', CS_MM)
    ]
    cs_sorted = sorted(cs_values, key=lambda x: x[1], reverse=True)
    for i, (setting, cs) in enumerate(cs_sorted, 1):
        print(f"  {i}. {setting:3s}: {cs:.6f}")

    # Proposition 4: NE should be more efficient than E (for symmetric G)
    if G.is_symmetric():
        print("\nProposition 4 (NE more efficient than E for symmetric G):")
        if TS_NE > TS_E:
            print(f"  ✓ Verified: TS_NE ({TS_NE:.6f}) > TS_E ({TS_E:.6f})")
        else:
            print(f"  ✗ VIOLATION: TS_NE ({TS_NE:.6f}) ≤ TS_E ({TS_E:.6f})")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
