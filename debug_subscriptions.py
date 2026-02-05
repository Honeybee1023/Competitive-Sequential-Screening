"""Debug subscription schedule implementation."""

import warnings
warnings.filterwarnings('ignore')

from competitive_screening.src.core.distributions import Normal, Uniform
from competitive_screening.src.core.equilibrium import solve_equilibrium_NE
from scipy.integrate import quad

# Baseline parameters
v_0 = 1.0
G = Uniform(-1.0, 1.0)
F = Normal(0.0, 1.0)

print("Solving NE equilibrium...")
eq_NE = solve_equilibrium_NE(v_0, G, F)
gamma_min, gamma_max = G.support()
print(f"Done. gamma_min={gamma_min}, gamma_max={gamma_max}")
print()

# Check strike prices
gamma_test = 0.0
p_A_test = eq_NE.p_A(gamma_test)
p_B_test = eq_NE.p_B(gamma_test)

print(f"Strike prices at γ=0:")
print(f"  p_A(0) = {p_A_test}")
print(f"  p_B(0) = {p_B_test}")
print()

# Compute upper bounds manually
g_min = G.pdf(gamma_min)
g_max = G.pdf(gamma_max)

p_bar_A = 2.0 / g_min
p_bar_B = 2.0 / g_max

print(f"Upper bounds:")
print(f"  g(γ_min) = {g_min}, p̄_A = {p_bar_A}")
print(f"  g(γ_max) = {g_max}, p̄_B = {p_bar_B}")
print()

# Check boundary utility term for s_A
print("Checking boundary utility term for s_A:")
print(f"  Computing E[θ|γ̄][(v_A(θ) - p̄_A - (v_B(θ))⁺)⁺] with γ̄ = {gamma_max}")

def boundary_integrand_A(epsilon):
    theta = gamma_max + epsilon
    v_A_theta = v_0 - theta
    v_B_theta = v_0 + theta
    surplus = max(v_A_theta - p_bar_A - max(v_B_theta, 0), 0)
    return surplus * F.pdf(epsilon)

eps_min, eps_max = F.support()
print(f"  Epsilon support: [{eps_min}, {eps_max}]")

# Sample a few epsilon values
test_epsilons = [-2, -1, 0, 1, 2]
print(f"  Sample evaluations:")
for eps in test_epsilons:
    theta = gamma_max + eps
    v_A = v_0 - theta
    v_B = v_0 + theta
    surplus = max(v_A - p_bar_A - max(v_B, 0), 0)
    print(f"    ε={eps:2d}: θ={theta:5.2f}, v_A={v_A:6.2f}, v_B={v_B:5.2f}, surplus={surplus:6.2f}")

boundary_util_A, _ = quad(boundary_integrand_A, eps_min, eps_max, limit=50)
print(f"  Boundary utility = {boundary_util_A}")
print()

# Check if Q_A is non-zero
print(f"Checking interim demand Q_A:")
for gamma in [-0.5, 0.0, 0.5]:
    Q_val = eq_NE.Q_A(gamma)
    p_val = eq_NE.p_A(gamma)
    print(f"  Q_A({gamma:4.1f}) = {Q_val:.6f}, p_A({gamma:4.1f}) = {p_val:.4f}")
print()

# Try to compute the integral term manually
print("Checking integral term for s_A at γ=0:")
print(f"  Need to compute ∫[p_A(0) to p̄_A] Q*_A(p') dp'")
print(f"  Integration limits: [{p_A_test:.4f}, {p_bar_A:.4f}]")

# Try a simple integral check
def test_integrand(p_prime):
    # Just return a constant to see if integration works
    return 0.1

test_integral, _ = quad(test_integrand, p_A_test, p_bar_A, limit=50)
print(f"  Test integral (constant 0.1): {test_integral:.6f}")
print(f"  Expected: {0.1 * (p_bar_A - p_A_test):.6f}")
print()

# Now try the actual integrand
from scipy.optimize import brentq

def actual_integrand(p_prime):
    def equation(g):
        return eq_NE.p_A(g) - p_prime
    try:
        gamma_prime = brentq(equation, gamma_min + 1e-9, gamma_max - 1e-9, xtol=1e-9)
        Q_val = eq_NE.Q_A(gamma_prime)
        return Q_val
    except:
        return 0.0

# Sample a few points
print("Sampling actual integrand:")
test_prices = [p_A_test, (p_A_test + p_bar_A) / 2, p_bar_A - 0.1]
for p in test_prices:
    val = actual_integrand(p)
    print(f"  p'={p:.4f}: integrand={val:.6f}")
print()

try:
    actual_integral, _ = quad(actual_integrand, p_A_test, p_bar_A, limit=50)
    print(f"Actual integral: {actual_integral:.6f}")
except Exception as e:
    print(f"Error computing integral: {e}")
print()

# Check subscription schedule
print(f"Checking subscription schedule function:")
s_A_0 = eq_NE.s_A(0.0)
print(f"  s_A(0) = {s_A_0}")
