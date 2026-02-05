"""Test subscription schedules with COVERED market."""

import warnings
warnings.filterwarnings('ignore')

from competitive_screening.src.core.distributions import Normal, Uniform
from competitive_screening.src.core.equilibrium import solve_equilibrium_NE
from competitive_screening.src.core.welfare import (
    compute_total_surplus_NE,
    compute_consumer_surplus_NE,
    compute_producer_surplus_NE,
)

print("Testing with COVERED market...")
print()

# Covered market: v_0 = 3 > max(1/g(γ)) = 2
v_0 = 3.0
G = Uniform(-1.0, 1.0)
F = Normal(0.0, 1.0)

print(f"Parameters: v_0={v_0}, G=Uniform(-1,1), F=Normal(0,1)")
print(f"Coverage check: max(1/g(γ)) = 1/0.5 = 2.0")
print(f"v_0 = {v_0} > 2.0 → Market IS COVERED ✓")
print()

# Solve equilibrium
print("Solving NE equilibrium...")
eq_NE = solve_equilibrium_NE(v_0, F, G)
print("Done.")
print()

# Check coverage flag
print(f"eq_NE.is_covered = {eq_NE.is_covered}")
print()

# Check subscription schedules
gamma_test = 0.0
s_A_test = eq_NE.s_A(gamma_test)
s_B_test = eq_NE.s_B(gamma_test)

print(f"Subscription schedules at γ=0:")
print(f"  s_A(0) = {s_A_test:.6f}")
print(f"  s_B(0) = {s_B_test:.6f}")
print()

# Compute welfare
print("Computing welfare...")
TS = compute_total_surplus_NE(eq_NE)
CS = compute_consumer_surplus_NE(eq_NE)
PS = compute_producer_surplus_NE(eq_NE)

print(f"  TS = {TS:.6f}")
print(f"  CS = {CS:.6f}")
print(f"  PS = {PS:.6f}")
print(f"  CS + PS = {CS + PS:.6f}")
print()

# Check identity
error = abs(TS - (CS + PS))
print(f"Welfare identity: |TS - (CS + PS)| = {error:.2e}")

if error < 1e-3:
    print("✓ PASS: Welfare identity holds!")
    print()
    print("CONCLUSION: NE works correctly when market is COVERED.")
    print("The issue with the baseline test was due to uncovered market (v_0=1 < 2).")
else:
    print(f"✗ FAIL: Welfare identity still violated (error = {error:.6f})")
    print()
    print("CONCLUSION: Issue is NOT related to market coverage.")
    print("Need to investigate NE formula implementation further.")
