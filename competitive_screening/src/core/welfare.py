"""
Welfare computation functions for all market settings.

Design Principle:
- Take SOLVED equilibrium objects as input (no re-solving)
- Return scalar welfare metrics
- Use numerical integration with adaptive quadrature

Welfare Decomposition:
- Total Surplus (TS) = Consumer Surplus (CS) + Producer Surplus (PS)
- TS = Expected social value of allocation
- CS = Consumer utility net of payments
- PS = Firm revenues (subscription fees + strike price revenues)
"""

from typing import Tuple
import numpy as np
from scipy.integrate import quad, dblquad

from .equilibrium import EquilibriumNE, EquilibriumSP, EquilibriumE, EquilibriumMM
from .distributions import Distribution, ConvolutionDistribution
from ..config import QUAD_LIMIT


# ==============================================================================
# NON-EXCLUSIVE SUBSCRIPTIONS (NE)
# ==============================================================================


def compute_total_surplus_NE(eq: EquilibriumNE) -> float:
    """
    Total surplus under non-exclusive subscriptions.

    TS = E[v_i(θ) * 1{buy from i}]

    Consumer subscribes to BOTH firms but purchases from at most one.
    Purchase decision: buy from A if v_A(θ) - p_A(γ) ≥ v_B(θ) - p_B(γ),
                       buy from B otherwise (if positive net value).

    With the equilibrium strike prices, allocation is determined by the
    interim demand functions Q_A(γ), Q_B(γ).

    Args:
        eq: Solved NE equilibrium

    Returns:
        Expected total surplus

    Reference: Model description pages 2-4
    """
    v_0 = eq.v_0
    G = eq.G
    F = eq.F

    gamma_min, gamma_max = G.support()

    def integrand_gamma(gamma: float) -> float:
        """Expected surplus from type γ."""
        eps_min, eps_max = F.support()

        p_A = eq.p_A(gamma)
        p_B = eq.p_B(gamma)

        def integrand_eps(epsilon: float) -> float:
            theta = gamma + epsilon
            v_A = v_0 - theta
            v_B = v_0 + theta

            # Consumer buys from firm with highest net value
            net_A = v_A - p_A
            net_B = v_B - p_B

            if net_A >= net_B and net_A >= 0:
                # Buy from A
                surplus = v_A
            elif net_B > net_A and net_B >= 0:
                # Buy from B
                surplus = v_B
            else:
                # Buy nothing
                surplus = 0.0

            return surplus * F.pdf(epsilon)

        result, _ = quad(integrand_eps, eps_min, eps_max, limit=QUAD_LIMIT)
        return result * G.pdf(gamma)

    ts, _ = quad(integrand_gamma, gamma_min, gamma_max, limit=QUAD_LIMIT)
    return ts


def compute_consumer_surplus_NE(eq: EquilibriumNE) -> float:
    """
    Consumer surplus under non-exclusive subscriptions.

    CS = E[U(γ)] where U(γ) = net utility for type γ

    U(γ) = -s_A(γ) - s_B(γ) + E_θ|γ[max{v_A(θ) - p_A(γ), v_B(θ) - p_B(γ), 0}]

    WARNING: PARTIAL IMPLEMENTATION
    Subscription schedules s_A(γ), s_B(γ) are not fully implemented
    (missing boundary conditions from Theorem 1 which is not in provided PDF).

    This function computes CS assuming s_A = s_B = 0, which will be INCORRECT.

    Args:
        eq: Solved NE equilibrium

    Returns:
        Expected consumer surplus (APPROXIMATE - subscription fees set to 0)

    Reference: Model pages 2-4, envelope theorem page 8
    """
    import warnings
    warnings.warn(
        "compute_consumer_surplus_NE: Subscription schedules not fully implemented. "
        "Result assumes s_A = s_B = 0, which is incorrect. "
        "Full implementation requires Theorem 1 boundary conditions."
    )

    v_0 = eq.v_0
    G = eq.G
    F = eq.F
    gamma_min, gamma_max = G.support()

    def integrand_gamma(gamma: float) -> float:
        """Consumer utility at type γ (WITHOUT subscription fees)."""
        eps_min, eps_max = F.support()

        p_A = eq.p_A(gamma)
        p_B = eq.p_B(gamma)

        def integrand_eps(epsilon: float) -> float:
            theta = gamma + epsilon
            v_A = v_0 - theta
            v_B = v_0 + theta

            # Net utility from best purchase choice
            net_A = v_A - p_A
            net_B = v_B - p_B
            purchase_utility = max(net_A, net_B, 0)

            return purchase_utility * F.pdf(epsilon)

        result, _ = quad(integrand_eps, eps_min, eps_max, limit=QUAD_LIMIT)

        # Subtract subscription fees (currently 0)
        s_A = eq.s_A(gamma)
        s_B = eq.s_B(gamma)
        net_utility = result - s_A - s_B

        return net_utility * G.pdf(gamma)

    cs, _ = quad(integrand_gamma, gamma_min, gamma_max, limit=QUAD_LIMIT)
    return cs


def compute_producer_surplus_NE(eq: EquilibriumNE) -> float:
    """
    Producer surplus (total revenue) under non-exclusive subscriptions.

    PS = E[s_A(γ) + s_B(γ) + p_i(γ) · 1{buy from i}]

    Both firms collect subscription fees. Only the firm that makes the sale
    collects the strike price.

    WARNING: PARTIAL IMPLEMENTATION
    Subscription schedules not fully implemented. Returns approximate PS.

    Args:
        eq: Solved NE equilibrium

    Returns:
        Expected producer surplus (APPROXIMATE)

    Reference: Payoffs specification page 4
    """
    import warnings
    warnings.warn(
        "compute_producer_surplus_NE: Subscription schedules not fully implemented. "
        "Result may be incorrect."
    )

    G = eq.G
    gamma_min, gamma_max = G.support()

    def integrand_gamma(gamma: float) -> float:
        """Revenue from type γ."""
        # Subscription fees (currently 0)
        s_A_val = eq.s_A(gamma)
        s_B_val = eq.s_B(gamma)

        # Expected strike price revenues
        p_A_val = eq.p_A(gamma)
        p_B_val = eq.p_B(gamma)
        Q_A_val = eq.Q_A(gamma)
        Q_B_val = eq.Q_B(gamma)

        # Total revenue: subscriptions + expected strike price revenue
        revenue = s_A_val + s_B_val + p_A_val * Q_A_val + p_B_val * Q_B_val

        return revenue * G.pdf(gamma)

    ps, _ = quad(integrand_gamma, gamma_min, gamma_max, limit=QUAD_LIMIT)
    return ps


# ==============================================================================
# SPOT PRICING (SP)
# ==============================================================================


def compute_total_surplus_SP(eq: EquilibriumSP) -> float:
    """
    Total surplus under spot pricing.

    TS = E[v_i(θ) * 1{purchase from i}]

    From Proposition 2 (page 10): Consumers with θ < θ* buy from A, θ > θ* buy from B.
    Market is covered (v_0 ≥ 1/h(θ*)), so everyone purchases.

    For θ < θ*: surplus = v_A(θ) = v_0 - θ
    For θ > θ*: surplus = v_B(θ) = v_0 + θ

    Args:
        eq: Solved SP equilibrium

    Returns:
        Expected total surplus

    Reference: Proposition 2, page 10
    """
    v_0 = eq.v_0
    H = eq.H
    theta_star = eq.theta_star

    theta_min, theta_max = H.support()

    def integrand(theta: float) -> float:
        v_A = v_0 - theta
        v_B = v_0 + theta

        # Allocation: buy from A if θ < θ*, else buy from B
        if theta < theta_star:
            surplus = v_A
        else:
            surplus = v_B

        return surplus * H.pdf(theta)

    ts, _ = quad(integrand, theta_min, theta_max, limit=QUAD_LIMIT)
    return ts


def compute_consumer_surplus_SP(eq: EquilibriumSP) -> float:
    """
    Consumer surplus under spot pricing.

    CS = E[max{v_A(θ) - p_A, v_B(θ) - p_B, 0}]

    Consumer chooses the firm offering highest net utility.
    For θ < θ*: buys from A, gets v_A(θ) - p_A
    For θ > θ*: buys from B, gets v_B(θ) - p_B

    Args:
        eq: Solved SP equilibrium

    Returns:
        Expected consumer surplus

    Reference: Standard Hotelling model
    """
    v_0 = eq.v_0
    H = eq.H
    theta_star = eq.theta_star
    p_A = eq.p_A
    p_B = eq.p_B

    theta_min, theta_max = H.support()

    def integrand(theta: float) -> float:
        v_A = v_0 - theta
        v_B = v_0 + theta

        # Consumer chooses best option
        net_A = v_A - p_A
        net_B = v_B - p_B
        utility = max(net_A, net_B, 0)

        return utility * H.pdf(theta)

    cs, _ = quad(integrand, theta_min, theta_max, limit=QUAD_LIMIT)
    return cs


def compute_producer_surplus_SP(eq: EquilibriumSP) -> float:
    """
    Producer surplus under spot pricing.

    PS = p_A * Pr(θ < θ*) + p_B * Pr(θ > θ*)

    Args:
        eq: Solved SP equilibrium

    Returns:
        Expected producer surplus
    """
    H = eq.H
    theta_star = eq.theta_star
    p_A = eq.p_A
    p_B = eq.p_B

    prob_buy_A = H.cdf(theta_star)
    prob_buy_B = 1 - prob_buy_A

    ps = p_A * prob_buy_A + p_B * prob_buy_B

    return ps


# ==============================================================================
# EXCLUSIVE SUBSCRIPTIONS (E)
# ==============================================================================


def compute_total_surplus_E(eq: EquilibriumE) -> float:
    """
    Total surplus under exclusive subscriptions.

    TS = E[v_i(θ) * 1{subscribed to i and v_i(θ) ≥ p^M_i(γ)}]

    From Proposition 3 (page 11): Market splits at γ̂.
    - Types γ ≤ γ̂: subscribe to A exclusively, buy if v_A(θ) ≥ p^M_A(γ)
    - Types γ > γ̂: subscribe to B exclusively, buy if v_B(θ) ≥ p^M_B(γ)

    EXCLUSIVE means: consumer can ONLY buy from subscribed firm,
    even if other firm would give higher utility.

    Args:
        eq: Solved E equilibrium

    Returns:
        Expected total surplus

    Reference: Proposition 3, pages 10-11, Figure 2 page 12
    """
    v_0 = eq.v_0
    G = eq.G
    F = eq.F
    gamma_hat = eq.gamma_hat

    gamma_min, gamma_max = G.support()

    def integrand_gamma(gamma: float) -> float:
        """Expected surplus from type γ."""
        eps_min, eps_max = F.support()

        if gamma <= gamma_hat:
            # Subscribed to A exclusively
            p_A = eq.p_M_A(gamma)

            def integrand_eps(epsilon: float) -> float:
                theta = gamma + epsilon
                v_A = v_0 - theta
                # Buy from A if v_A ≥ p_A, get surplus v_A
                # Cannot buy from B (exclusive contract)
                surplus = v_A if v_A >= p_A else 0.0
                return surplus * F.pdf(epsilon)

            result, _ = quad(integrand_eps, eps_min, eps_max, limit=QUAD_LIMIT)
        else:
            # Subscribed to B exclusively
            p_B = eq.p_M_B(gamma)

            def integrand_eps(epsilon: float) -> float:
                theta = gamma + epsilon
                v_B = v_0 + theta
                # Buy from B if v_B ≥ p_B, get surplus v_B
                # Cannot buy from A (exclusive contract)
                surplus = v_B if v_B >= p_B else 0.0
                return surplus * F.pdf(epsilon)

            result, _ = quad(integrand_eps, eps_min, eps_max, limit=QUAD_LIMIT)

        return result * G.pdf(gamma)

    ts, _ = quad(integrand_gamma, gamma_min, gamma_max, limit=QUAD_LIMIT)
    return ts


def compute_consumer_surplus_E(eq: EquilibriumE) -> float:
    """
    Consumer surplus under exclusive subscriptions.

    CS = E[U_i(γ) - s_i(γ)] where i is chosen firm

    U_A(γ) = E_θ|γ[(v_A(θ) - p^M_A(γ))_+] for γ ≤ γ̂
    U_B(γ) = E_θ|γ[(v_B(θ) - p^M_B(γ))_+] for γ > γ̂

    WARNING: PARTIAL IMPLEMENTATION
    Subscription schedules not fully implemented. Returns approximate CS.

    Args:
        eq: Solved E equilibrium

    Returns:
        Expected consumer surplus (APPROXIMATE)

    Reference: Proposition 3, pages 10-11
    """
    import warnings
    warnings.warn(
        "compute_consumer_surplus_E: Subscription schedules not fully implemented. "
        "Result assumes s_A = s_B = 0, which is incorrect."
    )

    v_0 = eq.v_0
    F = eq.F
    G = eq.G
    gamma_hat = eq.gamma_hat
    gamma_min, gamma_max = G.support()

    def integrand_gamma(gamma: float) -> float:
        """Consumer net utility at type γ."""
        eps_min, eps_max = F.support()

        if gamma <= gamma_hat:
            # Subscribed to A
            p_A = eq.p_M_A(gamma)

            def integrand_eps(epsilon: float) -> float:
                theta = gamma + epsilon
                v_A = v_0 - theta
                net = max(v_A - p_A, 0)
                return net * F.pdf(epsilon)

            utility, _ = quad(integrand_eps, eps_min, eps_max, limit=QUAD_LIMIT)
            s = eq.s_A(gamma)  # Currently 0
        else:
            # Subscribed to B
            p_B = eq.p_M_B(gamma)

            def integrand_eps(epsilon: float) -> float:
                theta = gamma + epsilon
                v_B = v_0 + theta
                net = max(v_B - p_B, 0)
                return net * F.pdf(epsilon)

            utility, _ = quad(integrand_eps, eps_min, eps_max, limit=QUAD_LIMIT)
            s = eq.s_B(gamma)  # Currently 0

        net_utility = utility - s
        return net_utility * G.pdf(gamma)

    cs, _ = quad(integrand_gamma, gamma_min, gamma_max, limit=QUAD_LIMIT)
    return cs


def compute_producer_surplus_E(eq: EquilibriumE) -> float:
    """
    Producer surplus under exclusive subscriptions.

    PS = E[s_i(γ) + p^M_i(γ) · P_θ|γ(v_i(θ) ≥ p^M_i(γ))]

    Firm i collects subscription fee s_i(γ) from type γ, plus strike price
    p^M_i(γ) if consumer makes a purchase.

    WARNING: PARTIAL IMPLEMENTATION
    Subscription schedules not fully implemented.

    Args:
        eq: Solved E equilibrium

    Returns:
        Expected producer surplus (APPROXIMATE)

    Reference: Payoffs specification page 4, Proposition 3 pages 10-11
    """
    import warnings
    warnings.warn(
        "compute_producer_surplus_E: Subscription schedules not fully implemented. "
        "Result may be incorrect."
    )

    v_0 = eq.v_0
    F = eq.F
    G = eq.G
    gamma_hat = eq.gamma_hat
    gamma_min, gamma_max = G.support()

    def integrand_gamma(gamma: float) -> float:
        """Revenue from type γ."""
        eps_min, eps_max = F.support()

        if gamma <= gamma_hat:
            # Firm A's revenue
            s = eq.s_A(gamma)  # Currently 0
            p = eq.p_M_A(gamma)

            # Compute purchase probability Q^M_A(p|γ) = P(v_A(θ) ≥ p)
            # v_A(θ) = v_0 - θ ≥ p => θ ≤ v_0 - p
            # θ = γ + ε => ε ≤ v_0 - p - γ
            Q = F.cdf(v_0 - p - gamma)

            revenue = s + p * Q
        else:
            # Firm B's revenue
            s = eq.s_B(gamma)  # Currently 0
            p = eq.p_M_B(gamma)

            # Compute purchase probability Q^M_B(p|γ) = P(v_B(θ) ≥ p)
            # v_B(θ) = v_0 + θ ≥ p => θ ≥ p - v_0
            # θ = γ + ε => ε ≥ p - v_0 - γ
            Q = F.sf(p - v_0 - gamma)

            revenue = s + p * Q

        return revenue * G.pdf(gamma)

    ps, _ = quad(integrand_gamma, gamma_min, gamma_max, limit=QUAD_LIMIT)
    return ps


# ==============================================================================
# MULTI-GOOD MONOPOLY (MM)
# ==============================================================================


def compute_total_surplus_MM(eq: EquilibriumMM) -> float:
    """
    Total surplus under multi-good monopoly.

    TS = E_θ|0[max{v_A(θ), v_B(θ)}] (by symmetry, equals E_θ over full distribution)

    From Proposition 5 (page 15): Allocation is EFFICIENT since strike prices = 0.
    Consumer always purchases from preferred firm.

    max{v_A(θ), v_B(θ)} = max{v_0 - θ, v_0 + θ} = v_0 + |θ|

    Args:
        eq: Solved MM equilibrium

    Returns:
        Expected total surplus

    Reference: Proposition 5, page 15
    """
    v_0 = eq.v_0
    F = eq.F
    G = eq.G

    # Since G is symmetric and we're computing over full distribution
    # Use convolution H = G * F
    H = ConvolutionDistribution(G, F)

    theta_min, theta_max = H.support()

    def integrand(theta: float) -> float:
        # max{v_A(θ), v_B(θ)} = v_0 + |θ|
        max_val = v_0 + abs(theta)
        return max_val * H.pdf(theta)

    ts, _ = quad(integrand, theta_min, theta_max, limit=QUAD_LIMIT)
    return ts


def compute_consumer_surplus_MM(eq: EquilibriumMM) -> float:
    """
    Consumer surplus under multi-good monopoly.

    CS = 0 (full surplus extraction via subscription fee).

    Args:
        eq: Solved MM equilibrium

    Returns:
        0.0
    """
    return 0.0


def compute_producer_surplus_MM(eq: EquilibriumMM) -> float:
    """
    Producer surplus under multi-good monopoly.

    PS = TS (monopolist extracts all surplus via subscription fee s).

    Args:
        eq: Solved MM equilibrium

    Returns:
        Expected producer surplus (equals total surplus)
    """
    return compute_total_surplus_MM(eq)


# ==============================================================================
# CONVENIENCE FUNCTIONS (for your requested API)
# ==============================================================================


def compute_all_welfare_NE(v_0: float, F: Distribution, G: Distribution) -> dict:
    """
    Convenience wrapper: solve equilibrium and compute all welfare metrics.

    WARNING: This re-solves equilibrium. If computing multiple metrics,
    use solve_equilibrium_NE once and pass the result to individual functions.

    Returns:
        dict with keys 'TS', 'CS', 'PS'
    """
    from .equilibrium import solve_equilibrium_NE
    eq = solve_equilibrium_NE(v_0, F, G)
    return {
        'TS': compute_total_surplus_NE(eq),
        'CS': compute_consumer_surplus_NE(eq),
        'PS': compute_producer_surplus_NE(eq)
    }


def compute_all_welfare_SP(v_0: float, F: Distribution, G: Distribution) -> dict:
    """Convenience wrapper for spot pricing."""
    from .equilibrium import solve_equilibrium_SP
    eq = solve_equilibrium_SP(v_0, F, G)
    return {
        'TS': compute_total_surplus_SP(eq),
        'CS': compute_consumer_surplus_SP(eq),
        'PS': compute_producer_surplus_SP(eq)
    }


def compute_all_welfare_E(v_0: float, F: Distribution, G: Distribution) -> dict:
    """Convenience wrapper for exclusive subscriptions."""
    from .equilibrium import solve_equilibrium_E
    eq = solve_equilibrium_E(v_0, F, G)
    return {
        'TS': compute_total_surplus_E(eq),
        'CS': compute_consumer_surplus_E(eq),
        'PS': compute_producer_surplus_E(eq)
    }


def compute_all_welfare_MM(v_0: float, F: Distribution, G: Distribution) -> dict:
    """Convenience wrapper for multi-good monopoly."""
    from .equilibrium import solve_equilibrium_MM
    eq = solve_equilibrium_MM(v_0, F, G)
    return {
        'TS': compute_total_surplus_MM(eq),
        'CS': compute_consumer_surplus_MM(eq),
        'PS': compute_producer_surplus_MM(eq)
    }


def compute_all_welfare(v_0: float, F: Distribution, G: Distribution) -> dict:
    """
    Master convenience function: compute TS for all 4 market settings.

    This is the recommended function for quick comparisons.

    Args:
        v_0: Average valuation
        F: Taste shock distribution (information precision)
        G: Type distribution (ex-ante heterogeneity)

    Returns:
        dict with keys: TS_NE, TS_SP, TS_E, TS_MM, CS_SP, CS_MM, PS_SP, PS_MM
        (CS and PS only for SP and MM, as NE and E are approximate)

    Example:
        >>> from src import Uniform, Normal
        >>> results = compute_all_welfare(6.0, Normal(0,1), Uniform(-1,1))
        >>> print(f"SP Total Surplus: {results['TS_SP']:.3f}")
    """
    import warnings

    # Suppress warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Compute welfare for all settings
        ne_results = compute_all_welfare_NE(v_0, F, G)
        sp_results = compute_all_welfare_SP(v_0, F, G)
        e_results = compute_all_welfare_E(v_0, F, G)
        mm_results = compute_all_welfare_MM(v_0, F, G)

    return {
        # Total Surplus (all accurate)
        'TS_NE': ne_results['TS'],
        'TS_SP': sp_results['TS'],
        'TS_E': e_results['TS'],
        'TS_MM': mm_results['TS'],

        # Consumer/Producer Surplus (SP and MM only - accurate)
        'CS_SP': sp_results['CS'],
        'CS_MM': mm_results['CS'],
        'PS_SP': sp_results['PS'],
        'PS_MM': mm_results['PS'],

        # Note: CS/PS for NE and E are approximate (not included)
    }
