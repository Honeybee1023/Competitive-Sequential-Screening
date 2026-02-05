"""Test s_A function directly to debug."""

import warnings
warnings.filterwarnings('ignore')

from competitive_screening.src.core.distributions import Normal, Uniform
from competitive_screening.src.core.equilibrium import solve_equilibrium_NE

# Baseline parameters
v_0 = 1.0
G = Uniform(-1.0, 1.0)
F = Normal(0.0, 1.0)

print("Solving NE equilibrium...")
# NOTE: Function signature is solve_equilibrium_NE(v_0, F, G)
# where F = taste shock distribution, G = type distribution
eq_NE = solve_equilibrium_NE(v_0, F, G)
print("Done.")
print()

# Test s_A
gamma_test = 0.0
print(f"Calling eq_NE.s_A({gamma_test})...")

try:
    result = eq_NE.s_A(gamma_test)
    print(f"Result: {result}")
    print(f"Type: {type(result)}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print()

# Also test that the functions exist
print(f"eq_NE.s_A: {eq_NE.s_A}")
print(f"eq_NE.s_B: {eq_NE.s_B}")
print()

# Test at a few other gamma values
for gamma in [-0.5, 0.0, 0.5]:
    try:
        s_A_val = eq_NE.s_A(gamma)
        s_B_val = eq_NE.s_B(gamma)
        print(f"γ={gamma:5.2f}: s_A={s_A_val:.6f}, s_B={s_B_val:.6f}")
    except Exception as e:
        print(f"γ={gamma:5.2f}: ERROR - {e}")
