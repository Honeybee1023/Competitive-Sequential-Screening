"""Test that compute_all_welfare now includes CS and PS for NE and E."""

import warnings
warnings.filterwarnings('ignore')

from competitive_screening.src.core.distributions import Normal, Uniform
from competitive_screening.src.core.welfare import compute_all_welfare

# Test with baseline parameters (covered market)
v_0 = 6.0
G = Uniform(-1.0, 1.0)
F = Normal(0.0, 1.0)

print("Testing compute_all_welfare()...")
print(f"Parameters: v_0={v_0}, G=Uniform(-1,1), F=Normal(0,1)")
print()

# Compute all welfare
results = compute_all_welfare(v_0, F, G)

print("Results dictionary keys:")
for key in sorted(results.keys()):
    print(f"  {key}")
print()

# Check that all expected keys are present
expected_keys = [
    'TS_NE', 'TS_SP', 'TS_E', 'TS_MM',
    'CS_NE', 'CS_SP', 'CS_E', 'CS_MM',
    'PS_NE', 'PS_SP', 'PS_E', 'PS_MM'
]

missing_keys = [k for k in expected_keys if k not in results]
if missing_keys:
    print(f"❌ MISSING KEYS: {missing_keys}")
else:
    print("✓ All expected keys present")
print()

# Display all values
print("Total Surplus (TS):")
for setting in ['NE', 'SP', 'E', 'MM']:
    print(f"  TS_{setting} = {results[f'TS_{setting}']:.6f}")
print()

print("Consumer Surplus (CS):")
for setting in ['NE', 'SP', 'E', 'MM']:
    val = results.get(f'CS_{setting}')
    if val is not None:
        print(f"  CS_{setting} = {val:.6f}")
    else:
        print(f"  CS_{setting} = None (missing!)")
print()

print("Producer Surplus (PS):")
for setting in ['NE', 'SP', 'E', 'MM']:
    val = results.get(f'PS_{setting}')
    if val is not None:
        print(f"  PS_{setting} = {val:.6f}")
    else:
        print(f"  PS_{setting} = None (missing!)")
print()

# Verify welfare identities
print("Welfare Identity Checks (CS + PS = TS):")
for setting in ['NE', 'SP', 'E', 'MM']:
    TS = results[f'TS_{setting}']
    CS = results.get(f'CS_{setting}')
    PS = results.get(f'PS_{setting}')

    if CS is not None and PS is not None:
        error = abs(TS - (CS + PS))
        status = "✓" if error < 1e-3 else "✗"
        print(f"  {setting}: {status} |TS - (CS + PS)| = {error:.2e}")
    else:
        print(f"  {setting}: ✗ Missing CS or PS")

print()
print("=" * 60)
if all(results.get(f'CS_{s}') is not None and results.get(f'PS_{s}') is not None
       for s in ['NE', 'SP', 'E', 'MM']):
    print("✓ SUCCESS: CS and PS available for all market settings!")
else:
    print("✗ FAILURE: CS or PS missing for some settings")
