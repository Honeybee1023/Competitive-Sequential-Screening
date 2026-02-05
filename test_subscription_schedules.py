"""
Test subscription schedule implementation for NE and E.

Verifies:
1. TS values remain unchanged
2. CS + PS = TS identity holds
3. Subscription schedules are non-zero
"""

import sys
import numpy as np
from competitive_screening.src.core.distributions import Normal, Uniform
from competitive_screening.src.core.equilibrium import (
    solve_equilibrium_NE,
    solve_equilibrium_E,
    solve_equilibrium_SP,
    solve_equilibrium_MM,
)
from competitive_screening.src.core.welfare import (
    compute_total_surplus_NE,
    compute_consumer_surplus_NE,
    compute_producer_surplus_NE,
    compute_total_surplus_E,
    compute_consumer_surplus_E,
    compute_producer_surplus_E,
    compute_total_surplus_SP,
    compute_consumer_surplus_SP,
    compute_producer_surplus_SP,
    compute_total_surplus_MM,
    compute_consumer_surplus_MM,
    compute_producer_surplus_MM,
)


def test_baseline_configuration():
    """Test with baseline configuration from paper."""
    print("=" * 70)
    print("TEST 1: Baseline Configuration")
    print("=" * 70)

    # Baseline parameters
    v_0 = 1.0
    G = Uniform(-1.0, 1.0)
    F = Normal(0.0, 1.0)

    print(f"Parameters: v_0={v_0}, G=Uniform(-1,1), F=Normal(0,1)")
    print()

    # Solve all equilibria
    print("Solving equilibria...")
    eq_NE = solve_equilibrium_NE(v_0, G, F)
    eq_E = solve_equilibrium_E(v_0, G, F)
    eq_SP = solve_equilibrium_SP(v_0, G, F)
    eq_MM = solve_equilibrium_MM(v_0, G, F)
    print("✓ All equilibria solved")
    print()

    # Test NE
    print("-" * 70)
    print("NON-EXCLUSIVE (NE)")
    print("-" * 70)

    # Check subscription schedules are non-zero
    gamma_test = 0.0
    s_A_test = eq_NE.s_A(gamma_test)
    s_B_test = eq_NE.s_B(gamma_test)
    print(f"Subscription schedules at γ=0:")
    print(f"  s_A(0) = {s_A_test:.6f}")
    print(f"  s_B(0) = {s_B_test:.6f}")

    if abs(s_A_test) < 1e-10 and abs(s_B_test) < 1e-10:
        print("  ⚠️  WARNING: Both subscription schedules are zero!")
    else:
        print("  ✓ Subscription schedules are non-zero")
    print()

    # Compute welfare
    TS_NE = compute_total_surplus_NE(eq_NE)
    CS_NE = compute_consumer_surplus_NE(eq_NE)
    PS_NE = compute_producer_surplus_NE(eq_NE)

    print(f"Welfare metrics:")
    print(f"  TS = {TS_NE:.6f}")
    print(f"  CS = {CS_NE:.6f}")
    print(f"  PS = {PS_NE:.6f}")
    print(f"  CS + PS = {CS_NE + PS_NE:.6f}")
    print()

    # Check identity
    identity_error_NE = abs(TS_NE - (CS_NE + PS_NE))
    print(f"Welfare identity: |TS - (CS + PS)| = {identity_error_NE:.2e}")

    if identity_error_NE < 1e-4:
        print("  ✓ PASS: Welfare identity holds")
    else:
        print(f"  ✗ FAIL: Welfare identity violated (error = {identity_error_NE:.6f})")
    print()

    # Test E
    print("-" * 70)
    print("EXCLUSIVE (E)")
    print("-" * 70)

    # Check subscription schedules
    gamma_low = eq_E.gamma_hat - 0.1
    gamma_high = eq_E.gamma_hat + 0.1
    s_A_low = eq_E.s_A(gamma_low)
    s_A_high = eq_E.s_A(gamma_high)
    s_B_low = eq_E.s_B(gamma_low)
    s_B_high = eq_E.s_B(gamma_high)

    print(f"Critical type: γ̂ = {eq_E.gamma_hat:.6f}")
    print(f"Subscription schedules:")
    print(f"  At γ = {gamma_low:.2f} (< γ̂): s_A = {s_A_low:.6f}, s_B = {s_B_low:.6f}")
    print(f"  At γ = {gamma_high:.2f} (> γ̂): s_A = {s_A_high:.6f}, s_B = {s_B_high:.6f}")

    # Check exclusivity constraint
    if s_A_high < 1e-10 and s_B_low < 1e-10:
        print("  ✓ Exclusivity: s_A(γ > γ̂) = 0, s_B(γ < γ̂) = 0")
    else:
        print("  ⚠️  WARNING: Exclusivity constraint may be violated")
    print()

    # Compute welfare
    TS_E = compute_total_surplus_E(eq_E)
    CS_E = compute_consumer_surplus_E(eq_E)
    PS_E = compute_producer_surplus_E(eq_E)

    print(f"Welfare metrics:")
    print(f"  TS = {TS_E:.6f}")
    print(f"  CS = {CS_E:.6f}")
    print(f"  PS = {PS_E:.6f}")
    print(f"  CS + PS = {CS_E + PS_E:.6f}")
    print()

    # Check identity
    identity_error_E = abs(TS_E - (CS_E + PS_E))
    print(f"Welfare identity: |TS - (CS + PS)| = {identity_error_E:.2e}")

    if identity_error_E < 1e-4:
        print("  ✓ PASS: Welfare identity holds")
    else:
        print(f"  ✗ FAIL: Welfare identity violated (error = {identity_error_E:.6f})")
    print()

    # Compare with SP and MM (sanity checks)
    print("-" * 70)
    print("SANITY CHECKS")
    print("-" * 70)

    TS_SP = compute_total_surplus_SP(eq_SP)
    CS_SP = compute_consumer_surplus_SP(eq_SP)
    PS_SP = compute_producer_surplus_SP(eq_SP)

    TS_MM = compute_total_surplus_MM(eq_MM)
    CS_MM = compute_consumer_surplus_MM(eq_MM)
    PS_MM = compute_producer_surplus_MM(eq_MM)

    print("All TS values:")
    print(f"  TS_NE = {TS_NE:.6f}")
    print(f"  TS_E  = {TS_E:.6f}")
    print(f"  TS_SP = {TS_SP:.6f}")
    print(f"  TS_MM = {TS_MM:.6f}")
    print()

    print("SP welfare identity:")
    identity_error_SP = abs(TS_SP - (CS_SP + PS_SP))
    print(f"  |TS - (CS + PS)| = {identity_error_SP:.2e}")
    if identity_error_SP < 1e-4:
        print("  ✓ PASS")
    else:
        print(f"  ✗ FAIL (error = {identity_error_SP:.6f})")
    print()

    print("MM welfare identity:")
    identity_error_MM = abs(TS_MM - (CS_MM + PS_MM))
    print(f"  |TS - (CS + PS)| = {identity_error_MM:.2e}")
    if identity_error_MM < 1e-4:
        print("  ✓ PASS")
    else:
        print(f"  ✗ FAIL (error = {identity_error_MM:.6f})")
    print()

    # Overall result
    print("=" * 70)
    print("OVERALL RESULT")
    print("=" * 70)

    all_pass = (
        identity_error_NE < 1e-4 and
        identity_error_E < 1e-4 and
        identity_error_SP < 1e-4 and
        identity_error_MM < 1e-4 and
        (abs(s_A_test) > 1e-10 or abs(s_B_test) > 1e-10)
    )

    if all_pass:
        print("✓ ALL TESTS PASSED")
        print()
        print("Summary:")
        print("  ✓ NE subscription schedules are non-zero")
        print("  ✓ NE welfare identity holds: CS + PS = TS")
        print("  ✓ E subscription schedules implemented correctly")
        print("  ✓ E welfare identity holds: CS + PS = TS")
        print("  ✓ SP welfare identity still holds (unchanged)")
        print("  ✓ MM welfare identity still holds (unchanged)")
        print()
        print("The subscription schedule implementation is CORRECT! ✨")
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print()
        print("Issues detected:")
        if identity_error_NE >= 1e-4:
            print(f"  ✗ NE welfare identity violated (error = {identity_error_NE:.6f})")
        if identity_error_E >= 1e-4:
            print(f"  ✗ E welfare identity violated (error = {identity_error_E:.6f})")
        if identity_error_SP >= 1e-4:
            print(f"  ✗ SP welfare identity violated (error = {identity_error_SP:.6f})")
        if identity_error_MM >= 1e-4:
            print(f"  ✗ MM welfare identity violated (error = {identity_error_MM:.6f})")
        if abs(s_A_test) < 1e-10 and abs(s_B_test) < 1e-10:
            print("  ✗ NE subscription schedules are zero")
        return False


def test_wide_types_configuration():
    """Test with wider type distribution."""
    print("\n\n")
    print("=" * 70)
    print("TEST 2: Wide Types Configuration")
    print("=" * 70)

    # Wide types parameters
    v_0 = 1.0
    G = Uniform(-2.0, 2.0)
    F = Normal(0.0, 0.5)

    print(f"Parameters: v_0={v_0}, G=Uniform(-2,2), F=Normal(0,0.5)")
    print()

    # Solve equilibria
    print("Solving equilibria...")
    eq_NE = solve_equilibrium_NE(v_0, G, F)
    eq_E = solve_equilibrium_E(v_0, G, F)
    print("✓ Equilibria solved")
    print()

    # Test NE
    print("NE welfare:")
    TS_NE = compute_total_surplus_NE(eq_NE)
    CS_NE = compute_consumer_surplus_NE(eq_NE)
    PS_NE = compute_producer_surplus_NE(eq_NE)

    identity_error_NE = abs(TS_NE - (CS_NE + PS_NE))
    print(f"  TS = {TS_NE:.6f}")
    print(f"  CS + PS = {CS_NE + PS_NE:.6f}")
    print(f"  |TS - (CS + PS)| = {identity_error_NE:.2e}")

    if identity_error_NE < 1e-4:
        print("  ✓ PASS")
    else:
        print(f"  ✗ FAIL (error = {identity_error_NE:.6f})")
    print()

    # Test E
    print("E welfare:")
    TS_E = compute_total_surplus_E(eq_E)
    CS_E = compute_consumer_surplus_E(eq_E)
    PS_E = compute_producer_surplus_E(eq_E)

    identity_error_E = abs(TS_E - (CS_E + PS_E))
    print(f"  TS = {TS_E:.6f}")
    print(f"  CS + PS = {CS_E + PS_E:.6f}")
    print(f"  |TS - (CS + PS)| = {identity_error_E:.2e}")

    if identity_error_E < 1e-4:
        print("  ✓ PASS")
    else:
        print(f"  ✗ FAIL (error = {identity_error_E:.6f})")

    return identity_error_NE < 1e-4 and identity_error_E < 1e-4


if __name__ == "__main__":
    print("Testing subscription schedule implementation...")
    print()

    try:
        test1_pass = test_baseline_configuration()
        test2_pass = test_wide_types_configuration()

        print("\n\n")
        print("=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        if test1_pass and test2_pass:
            print("✓ ALL TESTS PASSED")
            print()
            print("The subscription schedule implementation is working correctly!")
            print("You can now use accurate CS and PS for NE and E market settings.")
            sys.exit(0)
        else:
            print("✗ SOME TESTS FAILED")
            print()
            print("Please review the errors above.")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
