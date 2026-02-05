"""
Equilibrium objects and solvers for the four market settings.

Design Philosophy:
- Equilibrium solving is EXPENSIVE (root-finding, optimization)
- Equilibrium objects should be IMMUTABLE and CACHEABLE
- Welfare computations should take equilibrium objects, not re-solve

PLACEHOLDER NOTES:
- Exact functional forms need to be extracted from paper
- Numerical tolerances should be tuned based on observed behavior
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
from scipy.optimize import root_scalar, minimize_scalar, brentq
from scipy.integrate import quad

from .distributions import Distribution, ConvolutionDistribution
from ..config import ROOT_FINDING_TOL, QUAD_LIMIT


@dataclass(frozen=True)
class EquilibriumNE:
    """
    Non-Exclusive Subscriptions Equilibrium (Theorem 1).

    Components:
    - Strike price functions: p_A(γ), p_B(γ)
    - Subscription fee schedules: s_A(γ), s_B(γ)
    - Interim demand functions: Q_A(γ), Q_B(γ)
    """
    v_0: float
    F: Distribution
    G: Distribution

    # Strike prices as callables
    p_A: Callable[[float], float] = field(repr=False)
    p_B: Callable[[float], float] = field(repr=False)

    # Subscription schedules
    s_A: Callable[[float], float] = field(repr=False)
    s_B: Callable[[float], float] = field(repr=False)

    # Interim demand
    Q_A: Callable[[float], float] = field(repr=False)
    Q_B: Callable[[float], float] = field(repr=False)

    # Coverage condition
    is_covered: bool = True

    def __post_init__(self):
        """Verify equilibrium conditions (optional, for debugging)."""
        # PLACEHOLDER: Add equilibrium consistency checks
        pass


@dataclass(frozen=True)
class EquilibriumSP:
    """
    Spot Pricing Equilibrium (Proposition 2).

    Key object: critical position θ* solving:
        θ* = (1 - 2H(θ*)) / h(θ*)

    where H is the convolution of G and F.
    """
    v_0: float
    F: Distribution
    G: Distribution
    H: ConvolutionDistribution

    theta_star: float  # Critical position
    p_A: float         # Spot price for firm A
    p_B: float         # Spot price for firm B

    is_covered: bool = True


@dataclass(frozen=True)
class EquilibriumE:
    """
    Exclusive Subscriptions Equilibrium (Proposition 3).

    Key object: critical type γ̂ where consumers are indifferent
    between subscribing to A vs B.

    Equilibrium involves solving indifference condition (equation 9 in paper).
    """
    v_0: float
    F: Distribution
    G: Distribution

    gamma_hat: float  # Critical type (market split point)

    # Monopoly strike prices
    p_M_A: Callable[[float], float] = field(repr=False)
    p_M_B: Callable[[float], float] = field(repr=False)

    # Subscription schedules
    s_A: Callable[[float], float] = field(repr=False)
    s_B: Callable[[float], float] = field(repr=False)

    is_covered: bool = True


@dataclass(frozen=True)
class EquilibriumMM:
    """
    Multi-good Monopoly Equilibrium (Proposition 5).

    ONLY VALID FOR SYMMETRIC G.

    Optimal mechanism: single contract (p_A*, p_B*, s*) = (0, 0, E[max{v_A(θ), v_B(θ)}])
    Achieves efficient allocation with full surplus extraction.
    """
    v_0: float
    F: Distribution
    G: Distribution

    p_A: float = 0.0  # Always 0 in optimal mechanism
    p_B: float = 0.0  # Always 0 in optimal mechanism
    s: float = 0.0    # Total subscription fee (computed)

    def __post_init__(self):
        """Verify G is symmetric."""
        if not hasattr(self.G, 'is_symmetric') or not self.G.is_symmetric():
            raise ValueError("Monopoly mechanism only defined for symmetric G")


# ==============================================================================
# EQUILIBRIUM SOLVERS
# ==============================================================================


def solve_equilibrium_NE(v_0: float, F: Distribution, G: Distribution,
                         tol: float = None) -> EquilibriumNE:
    """
    Solve for non-exclusive subscriptions equilibrium.

    From the model (pages 2-3) and comparison to monopoly (page 9):
    - Strike prices: p_A*(γ) = 2G(γ)/g(γ), p_B*(γ) = 2(1-G(γ))/g(γ)
      (These are DOUBLE the monopoly strike prices due to competition)
    - Consumer subscribes to BOTH firms in period 1, chooses which to buy from in period 2
    - Subscription schedules determined by envelope theorem

    NOTE: The full "Theorem 1" is not included in the provided PDF excerpt.
    Strike price formulas are deduced from the spec and model structure.
    Subscription schedules require envelope theorem boundary conditions not specified.

    Args:
        v_0: Average valuation
        F: Taste shock distribution (symmetric, log-concave, mean 0)
        G: Type distribution (log-concave)
        tol: Numerical tolerance (uses global config if None)

    Returns:
        EquilibriumNE object containing equilibrium functions

    Reference: Model specification pages 2-3, equation (3) page 9
    Coverage condition: v_0 ≥ max_γ (1/g(γ))
    """
    if tol is None:
        tol = ROOT_FINDING_TOL

    gamma_min, gamma_max = G.support()

    # Strike prices (double the monopoly prices - competitive effect)
    def p_A(gamma: float) -> float:
        """Strike price for firm A at type γ: p*_A(γ) = 2G(γ)/g(γ)"""
        g_val = G.pdf(gamma)
        if g_val < 1e-12:
            return np.inf
        return 2 * G.cdf(gamma) / g_val

    def p_B(gamma: float) -> float:
        """Strike price for firm B at type γ: p*_B(γ) = 2(1-G(γ))/g(γ)"""
        g_val = G.pdf(gamma)
        if g_val < 1e-12:
            return np.inf
        return 2 * (1 - G.cdf(gamma)) / g_val

    # Interim demand functions (conditional purchase probabilities)
    def Q_A(gamma: float) -> float:
        """
        Probability that type γ purchases from A in period 2.

        Consumer buys from A if: v_A(θ) - p_A(γ) ≥ v_B(θ) - p_B(γ)

        Derivation:
        (v_0 - θ) - p_A ≥ (v_0 + θ) - p_B
        => p_B - p_A ≥ 2θ
        => θ ≤ (p_B - p_A)/2
        Since θ = γ + ε: ε ≤ (p_B - p_A)/2 - γ

        With p_A = 2G/g, p_B = 2(1-G)/g:
        (p_B - p_A)/2 = (1 - 2G(γ))/g(γ)
        """
        p_a_val = p_A(gamma)
        p_b_val = p_B(gamma)
        cutoff = (p_b_val - p_a_val) / 2 - gamma
        return F.cdf(cutoff)

    def Q_B(gamma: float) -> float:
        """
        Probability that type γ purchases from B in period 2.

        By complementarity (consumer has unit demand): Q_B = 1 - Q_A
        (assuming market is covered)
        """
        return 1.0 - Q_A(gamma)

    # Subscription schedules via envelope theorem
    # Formula from Theorem 1 (pages 16-18):
    # s*_A(p_A) = E[θ|γ̄][(v_A(θ) - p̄_A - (v_B(θ))⁺)⁺] + ∫[p_A to p̄_A] Q*_A(p') dp'
    # s*_B(p_B) = E[θ|γ̲][(v_B(θ) - p̄_B - (v_A(θ))⁺)⁺] + ∫[p_B to p̄_B] Q*_B(p') dp'

    # Compute upper bounds on strike prices
    g_min_val = G.pdf(gamma_min)
    g_max_val = G.pdf(gamma_max)

    if g_min_val < 1e-12 or g_max_val < 1e-12:
        # Boundary density too small, use approximate bounds
        p_bar_A = 2.0 / (g_min_val if g_min_val > 1e-12 else 1e-6)
        p_bar_B = 2.0 / (g_max_val if g_max_val > 1e-12 else 1e-6)
    else:
        p_bar_A = 2.0 / g_min_val  # p̄_A = 2/g(γ̲)
        p_bar_B = 2.0 / g_max_val  # p̄_B = 2/g(γ̄)

    def s_A(gamma: float) -> float:
        """
        Subscription fee for firm A at type γ.

        Implementation:
        s*_A(p*_A(γ)) = E[θ|γ̄][(v_A(θ) - p̄_A - (v_B(θ))⁺)⁺] + ∫[p*_A(γ) to p̄_A] Q*_A(p') dp'

        Reference: Theorem 1, pages 16-18
        """
        p_A_val = p_A(gamma)

        # Boundary utility term: E[θ|γ̄][(v_A(θ) - p̄_A - (v_B(θ))⁺)⁺]
        def boundary_integrand(epsilon: float) -> float:
            theta = gamma_max + epsilon
            v_A_theta = v_0 - theta
            v_B_theta = v_0 + theta
            # (v_A - p̄_A - max(v_B, 0))⁺
            surplus = max(v_A_theta - p_bar_A - max(v_B_theta, 0), 0)
            return surplus * F.pdf(epsilon)

        eps_min, eps_max = F.support()
        boundary_utility, _ = quad(boundary_integrand, eps_min, eps_max,
                                   limit=QUAD_LIMIT, epsabs=1e-8, epsrel=1e-8)

        # Integral term: ∫[p_A to p̄_A] Q*_A(p') dp'
        # For each p', find γ' such that p*_A(γ') = p', then Q*_A(p') = Q_A(γ')
        if p_A_val >= p_bar_A:
            integral_term = 0.0
        else:
            def integrand_Q(p_prime: float) -> float:
                # Invert p*_A(γ) = 2G(γ)/g(γ) = p_prime to find γ
                # This is monotonic, so use root-finding
                def equation(g: float) -> float:
                    return p_A(g) - p_prime

                try:
                    # Find γ' such that p*_A(γ') = p_prime
                    result = brentq(equation, gamma_min + 1e-9, gamma_max - 1e-9,
                                   xtol=tol)
                    gamma_prime = result
                    return Q_A(gamma_prime)
                except (ValueError, RuntimeError):
                    # If inversion fails, return 0
                    return 0.0

            integral_term, _ = quad(integrand_Q, p_A_val, p_bar_A,
                                   limit=QUAD_LIMIT, epsabs=1e-8, epsrel=1e-8)

        return boundary_utility + integral_term

    def s_B(gamma: float) -> float:
        """
        Subscription fee for firm B at type γ.

        Implementation:
        s*_B(p*_B(γ)) = E[θ|γ̲][(v_B(θ) - p̄_B - (v_A(θ))⁺)⁺] + ∫[p*_B(γ) to p̄_B] Q*_B(p') dp'

        Reference: Theorem 1, pages 16-18
        """
        p_B_val = p_B(gamma)

        # Boundary utility term: E[θ|γ̲][(v_B(θ) - p̄_B - (v_A(θ))⁺)⁺]
        def boundary_integrand(epsilon: float) -> float:
            theta = gamma_min + epsilon
            v_A_theta = v_0 - theta
            v_B_theta = v_0 + theta
            # (v_B - p̄_B - max(v_A, 0))⁺
            surplus = max(v_B_theta - p_bar_B - max(v_A_theta, 0), 0)
            return surplus * F.pdf(epsilon)

        eps_min, eps_max = F.support()
        boundary_utility, _ = quad(boundary_integrand, eps_min, eps_max,
                                   limit=QUAD_LIMIT, epsabs=1e-8, epsrel=1e-8)

        # Integral term: ∫[p_B to p̄_B] Q*_B(p') dp'
        if p_B_val >= p_bar_B:
            integral_term = 0.0
        else:
            def integrand_Q(p_prime: float) -> float:
                # Invert p*_B(γ) = 2(1-G(γ))/g(γ) = p_prime to find γ
                def equation(g: float) -> float:
                    return p_B(g) - p_prime

                try:
                    result = brentq(equation, gamma_min + 1e-9, gamma_max - 1e-9,
                                   xtol=tol)
                    gamma_prime = result
                    return Q_B(gamma_prime)
                except (ValueError, RuntimeError):
                    return 0.0

            integral_term, _ = quad(integrand_Q, p_B_val, p_bar_B,
                                   limit=QUAD_LIMIT, epsabs=1e-8, epsrel=1e-8)

        return boundary_utility + integral_term

    # Coverage condition check: v_0 ≥ max_γ (1/g(γ))
    gamma_grid = np.linspace(gamma_min, gamma_max, 100)
    inverse_g = []
    for g in gamma_grid:
        g_val = G.pdf(g)
        if g_val > 1e-12:
            inverse_g.append(1.0 / g_val)
        else:
            inverse_g.append(np.inf)

    max_inverse_g = max(inverse_g)
    is_covered = v_0 >= max_inverse_g

    return EquilibriumNE(
        v_0=v_0, F=F, G=G,
        p_A=p_A, p_B=p_B,
        s_A=s_A, s_B=s_B,
        Q_A=Q_A, Q_B=Q_B,
        is_covered=is_covered
    )


def solve_equilibrium_SP(v_0: float, F: Distribution, G: Distribution,
                         tol: float = None) -> EquilibriumSP:
    """
    Solve for spot pricing equilibrium.

    From Proposition 2 (page 10, equation 4):
    - Find θ* solving: θ* = (1 - 2H(θ*)) / h(θ*)
    - p_A* = 2H(θ*) / h(θ*)
    - p_B* = 2(1 - H(θ*)) / h(θ*)

    where H = G * F (convolution), h is density of H.

    This is the standard Hotelling equilibrium with full information.
    Consumers to the left of θ* purchase from A, those to the right purchase from B.

    Args:
        v_0: Average valuation
        F: Taste shock distribution (symmetric, log-concave)
        G: Type distribution (log-concave)
        tol: Root-finding tolerance (uses global config if None)

    Returns:
        EquilibriumSP object

    Reference: Proposition 2, page 10
    """
    if tol is None:
        tol = ROOT_FINDING_TOL

    # Construct convolution H: density h(θ) = ∫ g(γ)f(θ-γ) dγ
    H = ConvolutionDistribution(G, F)

    # Find θ* via root-finding (equation 4)
    def equilibrium_condition(theta: float) -> float:
        """
        Equilibrium condition from FOC: θ* = (1 - 2H(θ*)) / h(θ*)

        Rearranged: θ* · h(θ*) - (1 - 2H(θ*)) = 0
        """
        h_val = H.pdf(theta)
        if h_val < 1e-12:
            # Avoid division by zero at boundaries
            return np.inf if theta > 0 else -np.inf
        return theta - (1 - 2*H.cdf(theta)) / h_val

    # Find root - try around median first
    theta_min, theta_max = H.support()

    # Special case: For symmetric F (mean 0) and symmetric G, θ* = 0 exactly
    # Check if equilibrium condition is close to 0 at theta=0
    eq_at_zero = equilibrium_condition(0.0)
    if abs(eq_at_zero) < tol * 10:  # Close enough to zero
        theta_star = 0.0
    else:
        # General case: find root via bracketing
        try:
            # Check that function changes sign
            val_min = equilibrium_condition(theta_min + 1e-6)
            val_max = equilibrium_condition(theta_max - 1e-6)

            if np.sign(val_min) == np.sign(val_max):
                # Try around 0 with smaller bracket
                result = root_scalar(
                    equilibrium_condition,
                    bracket=[-1e-3, 1e-3],
                    method='brentq',
                    xtol=tol
                )
                theta_star = result.root
            else:
                result = root_scalar(
                    equilibrium_condition,
                    bracket=[theta_min + 1e-6, theta_max - 1e-6],
                    method='brentq',
                    xtol=tol
                )
                theta_star = result.root
        except (ValueError, RuntimeError) as e:
            raise RuntimeError(
                f"Could not find spot pricing equilibrium: {e}\n"
                f"eq(0) = {eq_at_zero:.6e}, eq(θ_min) = {val_min:.6e}, eq(θ_max) = {val_max:.6e}\n"
                f"This may indicate violated assumptions or extreme parameters."
            )

    # Compute equilibrium prices (Proposition 2)
    h_star = H.pdf(theta_star)
    H_star = H.cdf(theta_star)

    if h_star < 1e-12:
        raise RuntimeError(f"Density h(θ*) too close to zero: {h_star}")

    p_A = 2 * H_star / h_star
    p_B = 2 * (1 - H_star) / h_star

    # Coverage condition: v_0 ≥ 1/h(θ*)
    is_covered = v_0 >= 1.0 / h_star

    return EquilibriumSP(
        v_0=v_0, F=F, G=G, H=H,
        theta_star=theta_star,
        p_A=p_A, p_B=p_B,
        is_covered=is_covered
    )


def solve_equilibrium_E(v_0: float, F: Distribution, G: Distribution,
                        tol: float = None) -> EquilibriumE:
    """
    Solve for exclusive subscriptions equilibrium.

    From Proposition 3 (pages 10-11):
    - Find γ̂ solving indifference condition (equation 5)
    - Monopoly strike prices: p^M_A(γ) = G(γ)/g(γ), p^M_B(γ) = (1-G(γ))/g(γ)
    - Market split: types γ ≤ γ̂ choose firm A, types γ > γ̂ choose firm B
    - Each firm offers monopoly pricing to its captive consumers

    The critical type γ̂ is indifferent between subscribing to A vs B.

    Args:
        v_0: Average valuation
        F: Taste shock distribution
        G: Type distribution
        tol: Root-finding tolerance (uses global config if None)

    Returns:
        EquilibriumE object

    Reference: Proposition 3, pages 10-11
    Assumption: γ_min < 0 < γ_max (ensures both firms have positive market share)
    Coverage condition: v_0 ≥ 1/g(γ̂)
    """
    if tol is None:
        tol = ROOT_FINDING_TOL

    gamma_min, gamma_max = G.support()

    # Monopoly strike prices (equation 3, page 9)
    def p_M_A(gamma: float) -> float:
        """Monopoly strike price for firm A: p^M_A(γ) = G(γ)/g(γ)"""
        g_val = G.pdf(gamma)
        if g_val < 1e-12:
            return np.inf
        return G.cdf(gamma) / g_val

    def p_M_B(gamma: float) -> float:
        """Monopoly strike price for firm B: p^M_B(γ) = (1-G(γ))/g(γ)"""
        g_val = G.pdf(gamma)
        if g_val < 1e-12:
            return np.inf
        return (1 - G.cdf(gamma)) / g_val

    # Interim monopoly demand (equation 2, page 7)
    def Q_M_A(p_A: float, gamma: float) -> float:
        """P_θ|γ(v_A(θ) ≥ p_A) where v_A(θ) = v_0 - θ"""
        # v_0 - θ ≥ p_A => θ ≤ v_0 - p_A
        # θ = γ + ε => ε ≤ v_0 - p_A - γ
        return F.cdf(v_0 - p_A - gamma)

    def Q_M_B(p_B: float, gamma: float) -> float:
        """P_θ|γ(v_B(θ) ≥ p_B) where v_B(θ) = v_0 + θ"""
        # v_0 + θ ≥ p_B => θ ≥ p_B - v_0
        # θ = γ + ε => ε ≥ p_B - v_0 - γ
        return F.sf(p_B - v_0 - gamma)

    # Indifference condition for γ̂ (equation 5, page 11)
    def indifference_condition(gamma_hat: float) -> float:
        """
        Indifference condition (equation 5):

        E_θ|γ̂[(v_A(θ) - p^M_A(γ̂))_+] - p^M_A(γ̂) Q^M_B(p^M_B(γ̂)|γ̂)
        = E_θ|γ̂[(v_B(θ) - p^M_B(γ̂))_+] - p^M_B(γ̂) Q^M_A(p^M_A(γ̂)|γ̂)

        LHS: Utility from choosing firm A as exclusive supplier
        RHS: Utility from choosing firm B as exclusive supplier

        At γ̂, consumer is indifferent.
        """
        p_A_hat = p_M_A(gamma_hat)
        p_B_hat = p_M_B(gamma_hat)

        # Compute E_θ|γ̂[(v_A(θ) - p_A)_+]
        def integrand_A(epsilon: float) -> float:
            theta = gamma_hat + epsilon
            v_A = v_0 - theta
            surplus = max(v_A - p_A_hat, 0)
            return surplus * F.pdf(epsilon)

        eps_min, eps_max = F.support()
        utility_A_product, _ = quad(integrand_A, eps_min, eps_max, limit=QUAD_LIMIT)

        # Compute E_θ|γ̂[(v_B(θ) - p_B)_+]
        def integrand_B(epsilon: float) -> float:
            theta = gamma_hat + epsilon
            v_B = v_0 + theta
            surplus = max(v_B - p_B_hat, 0)
            return surplus * F.pdf(epsilon)

        utility_B_product, _ = quad(integrand_B, eps_min, eps_max, limit=QUAD_LIMIT)

        # Opportunity costs
        Q_B_hat = Q_M_B(p_B_hat, gamma_hat)
        Q_A_hat = Q_M_A(p_A_hat, gamma_hat)

        utility_A = utility_A_product - p_A_hat * Q_B_hat
        utility_B = utility_B_product - p_B_hat * Q_A_hat

        return utility_A - utility_B

    # Find γ̂ solving indifference condition
    # Check if γ_min < 0 < γ_max (Proposition 3 assumption)
    if not (gamma_min < 0 < gamma_max):
        raise ValueError(
            f"Proposition 3 requires γ_min < 0 < γ_max, got [{gamma_min}, {gamma_max}].\n"
            "This ensures both firms have positive market share."
        )

    try:
        result = root_scalar(
            indifference_condition,
            bracket=[gamma_min + 1e-6, gamma_max - 1e-6],
            method='brentq',
            xtol=tol
        )
        gamma_hat = result.root
    except ValueError as e:
        raise RuntimeError(
            f"Could not find exclusive equilibrium critical type γ̂: {e}\n"
            f"Indifference condition may not have a solution."
        )

    # Subscription schedules (Proposition 4, pages 23-25)
    # Formula:
    # s*_A(p_A) = p̂_A · Q^M_B(p̂_B|γ̂) + ∫[p_A to p̂_A] Q*_A(p') dp'
    # s*_B(p_B) = p̂_B · Q^M_A(p̂_A|γ̂) + ∫[p_B to p̂_B] Q*_B(p') dp'

    # Equilibrium strike prices at critical type
    p_hat_A = p_M_A(gamma_hat)
    p_hat_B = p_M_B(gamma_hat)

    # Boundary terms (opportunity costs)
    Q_B_at_hat = Q_M_B(p_hat_B, gamma_hat)
    Q_A_at_hat = Q_M_A(p_hat_A, gamma_hat)

    def s_A(gamma: float) -> float:
        """
        Subscription fee for firm A at type γ.

        Implementation:
        s*_A(p^M_A(γ)) = p̂_A · Q^M_B(p̂_B|γ̂) + ∫[p^M_A(γ) to p̂_A] Q*_A(p') dp'

        Only types γ ≤ γ̂ subscribe to A (exclusive contracts).

        Reference: Proposition 4, pages 23-25
        """
        if gamma > gamma_hat:
            # Type chooses firm B, doesn't subscribe to A
            return 0.0

        p_A_val = p_M_A(gamma)

        # Boundary term
        boundary_term = p_hat_A * Q_B_at_hat

        # Integral term: ∫[p_A to p̂_A] Q*_A(p') dp'
        # For exclusive, Q*_A(p') is the monopoly demand at price p'
        if p_A_val >= p_hat_A:
            integral_term = 0.0
        else:
            def integrand_Q(p_prime: float) -> float:
                # Invert p^M_A(γ) = G(γ)/g(γ) = p_prime to find γ
                def equation(g: float) -> float:
                    return p_M_A(g) - p_prime

                try:
                    result = brentq(equation, gamma_min + 1e-9, gamma_hat - 1e-9,
                                   xtol=tol)
                    gamma_prime = result
                    # Monopoly demand for firm A at type γ'
                    return Q_M_A(p_prime, gamma_prime)
                except (ValueError, RuntimeError):
                    return 0.0

            integral_term, _ = quad(integrand_Q, p_A_val, p_hat_A,
                                   limit=QUAD_LIMIT, epsabs=1e-8, epsrel=1e-8)

        return boundary_term + integral_term

    def s_B(gamma: float) -> float:
        """
        Subscription fee for firm B at type γ.

        Implementation:
        s*_B(p^M_B(γ)) = p̂_B · Q^M_A(p̂_A|γ̂) + ∫[p^M_B(γ) to p̂_B] Q*_B(p') dp'

        Only types γ > γ̂ subscribe to B (exclusive contracts).

        Reference: Proposition 4, pages 23-25
        """
        if gamma <= gamma_hat:
            # Type chooses firm A, doesn't subscribe to B
            return 0.0

        p_B_val = p_M_B(gamma)

        # Boundary term
        boundary_term = p_hat_B * Q_A_at_hat

        # Integral term: ∫[p_B to p̂_B] Q*_B(p') dp'
        if p_B_val >= p_hat_B:
            integral_term = 0.0
        else:
            def integrand_Q(p_prime: float) -> float:
                # Invert p^M_B(γ) = (1-G(γ))/g(γ) = p_prime to find γ
                def equation(g: float) -> float:
                    return p_M_B(g) - p_prime

                try:
                    result = brentq(equation, gamma_hat + 1e-9, gamma_max - 1e-9,
                                   xtol=tol)
                    gamma_prime = result
                    # Monopoly demand for firm B at type γ'
                    return Q_M_B(p_prime, gamma_prime)
                except (ValueError, RuntimeError):
                    return 0.0

            integral_term, _ = quad(integrand_Q, p_B_val, p_hat_B,
                                   limit=QUAD_LIMIT, epsabs=1e-8, epsrel=1e-8)

        return boundary_term + integral_term

    # Coverage condition
    g_hat = G.pdf(gamma_hat)
    is_covered = v_0 >= (1.0 / g_hat if g_hat > 1e-12 else np.inf)

    return EquilibriumE(
        v_0=v_0, F=F, G=G,
        gamma_hat=gamma_hat,
        p_M_A=p_M_A, p_M_B=p_M_B,
        s_A=s_A, s_B=s_B,
        is_covered=is_covered
    )


def solve_equilibrium_MM(v_0: float, F: Distribution, G: Distribution) -> EquilibriumMM:
    """
    Solve for multi-good monopoly equilibrium.

    From Proposition 5 (page 15):
    - Optimal contract: (p_A*, p_B*, s*) = (0, 0, E_θ|0[max{v_A(θ), v_B(θ)}])
    - Efficient allocation: consumers always buy from preferred firm
    - Full surplus extraction: consumer surplus = 0

    The monopolist offers a single contract with zero strike prices, extracting
    all surplus via the subscription fee. Type γ=0 (indifferent type) determines
    the maximum extractable fee.

    ONLY VALID for symmetric G (required by Proposition 5).

    Args:
        v_0: Average valuation
        F: Taste shock distribution (symmetric around 0)
        G: Type distribution (MUST be symmetric)

    Returns:
        EquilibriumMM object

    Reference: Proposition 5, page 15
    Coverage condition: v_0 ≥ max_γ 1/(2g(γ))
    """

    # Verify G symmetry (strict requirement)
    if not (hasattr(G, 'is_symmetric') and G.is_symmetric()):
        raise ValueError(
            "Multi-good monopoly mechanism only defined for symmetric G.\n"
            "Proposition 5 requires G to be symmetric around its mean."
        )

    # Compute subscription fee: s* = E_θ|γ=0[max{v_A(θ), v_B(θ)}]
    # Since θ = γ + ε and we condition on γ=0, θ = ε ~ F
    # max{v_A(θ), v_B(θ)} = max{v_0 - θ, v_0 + θ} = v_0 + |θ|

    eps_min, eps_max = F.support()

    def integrand(epsilon: float) -> float:
        """
        E_θ|0[max{v_A(θ), v_B(θ)}] where θ = ε when γ=0.

        max{v_0 - ε, v_0 + ε} = v_0 + |ε|
        """
        theta = epsilon  # Since γ=0
        max_val = v_0 + abs(theta)
        return max_val * F.pdf(epsilon)

    expected_max_val, _ = quad(integrand, eps_min, eps_max, limit=QUAD_LIMIT)

    return EquilibriumMM(
        v_0=v_0, F=F, G=G,
        p_A=0.0, p_B=0.0,
        s=expected_max_val
    )
