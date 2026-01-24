"""
Parameter validation and assumption checking.

Ground Truth Engineering:
- NEVER trust assumptions without verification
- Log warnings when assumptions are violated
- Provide actionable diagnostics
"""

from typing import Dict, List, Tuple
import numpy as np
from ..core.distributions import Distribution


class AssumptionViolation(Exception):
    """Raised when critical model assumptions are violated."""
    pass


def validate_parameters(v_0: float, F: Distribution, G: Distribution,
                       strict: bool = True) -> Dict[str, Tuple[bool, str]]:
    """
    Validate all model parameters and assumptions.

    Args:
        v_0: Average valuation (should be positive)
        F: Taste shock distribution
        G: Type distribution
        strict: If True, raise exception on critical violations

    Returns:
        Dict mapping assumption names to (passed, message) tuples

    Raises:
        AssumptionViolation: If strict=True and critical assumption fails
    """
    results = {}
    critical_failures = []

    # v_0 must be positive
    v0_positive = v_0 > 0
    results['v0_positive'] = (
        v0_positive,
        f"v_0 must be positive (got {v_0})"
    )
    if not v0_positive:
        critical_failures.append("v0_positive")

    # F: symmetric around 0
    if hasattr(F, 'is_symmetric'):
        F_symmetric = F.is_symmetric(0.0)
    else:
        # Numerical check
        F_symmetric = check_symmetry_numerical(F, center=0.0)

    results['F_symmetric'] = (
        F_symmetric,
        "F must be symmetric around 0"
    )
    if not F_symmetric:
        critical_failures.append("F_symmetric")

    # F: mean zero
    F_mean = F.mean()
    F_mean_zero = abs(F_mean) < 1e-6
    results['F_mean_zero'] = (
        F_mean_zero,
        f"F must have mean 0 (got {F_mean:.6e})"
    )
    if not F_mean_zero:
        critical_failures.append("F_mean_zero")

    # F: log-concave (declared)
    F_log_concave = F.is_log_concave()
    results['F_log_concave'] = (
        F_log_concave,
        "F must be log-concave (check distribution definition)"
    )

    # G: log-concave (declared)
    G_log_concave = G.is_log_concave()
    results['G_log_concave'] = (
        G_log_concave,
        "G must be log-concave (check distribution definition)"
    )

    # G: support should be bounded (common assumption)
    g_min, g_max = G.support()
    G_bounded = np.isfinite(g_min) and np.isfinite(g_max)
    results['G_bounded'] = (
        G_bounded,
        f"G should have bounded support (got [{g_min}, {g_max}])"
    )

    # Raise exception if critical failures in strict mode
    if strict and critical_failures:
        failed = ", ".join(critical_failures)
        raise AssumptionViolation(
            f"Critical assumption violations: {failed}\n" +
            "\n".join(f"  - {k}: {v[1]}" for k, v in results.items() if not v[0])
        )

    return results


def check_coverage_NE(v_0: float, G: Distribution, num_points: int = 100) -> Tuple[bool, float]:
    """
    Check coverage condition for non-exclusive equilibrium.

    Condition: v_0 >= max_γ (1 / g(γ))

    Args:
        v_0: Average valuation
        G: Type distribution
        num_points: Number of points to check

    Returns:
        (is_covered, max_inverse_g)
    """
    gamma_min, gamma_max = G.support()
    gamma_grid = np.linspace(gamma_min, gamma_max, num_points)

    max_inverse_g = 0.0
    for gamma in gamma_grid:
        g_val = G.pdf(gamma)
        if g_val > 1e-12:
            inverse_g = 1.0 / g_val
            max_inverse_g = max(max_inverse_g, inverse_g)
        else:
            max_inverse_g = np.inf
            break

    is_covered = v_0 >= max_inverse_g

    return is_covered, max_inverse_g


def check_coverage_SP(v_0: float, theta_star: float, H) -> bool:
    """
    Check coverage condition for spot pricing.

    Condition: v_0 >= 1 / h(θ*)

    Args:
        v_0: Average valuation
        theta_star: Equilibrium critical position
        H: Convolution distribution

    Returns:
        True if market is covered
    """
    h_star = H.pdf(theta_star)
    if h_star < 1e-12:
        return False
    return v_0 >= 1.0 / h_star


def check_symmetry_numerical(dist: Distribution, center: float = 0.0,
                             num_points: int = 50, tol: float = 1e-6) -> bool:
    """
    Numerically check if distribution is symmetric around center.

    WARNING: This is approximate and may give false negatives.

    Args:
        dist: Distribution to check
        center: Center point for symmetry
        num_points: Number of points to check
        tol: Tolerance for PDF differences

    Returns:
        True if distribution appears symmetric
    """
    lower, upper = dist.support()

    if not (np.isfinite(lower) and np.isfinite(upper)):
        # For unbounded support, check within reasonable range
        std = 3.0  # Assume something like normal distribution
        lower = center - 3 * std
        upper = center + 3 * std

    # Check symmetry: f(center + x) == f(center - x)
    test_points = np.linspace(0, min(upper - center, center - lower), num_points)

    for x in test_points[1:]:  # Skip x=0
        left = dist.pdf(center - x)
        right = dist.pdf(center + x)

        if abs(left - right) > tol:
            return False

    return True


def diagnose_equilibrium_failure(v_0: float, F: Distribution, G: Distribution,
                                 setting: str) -> str:
    """
    Provide diagnostic information when equilibrium solving fails.

    Args:
        v_0: Average valuation
        F: Taste shock distribution
        G: Type distribution
        setting: Market setting ('NE', 'SP', 'E', 'MM')

    Returns:
        Diagnostic message string
    """
    diag = [f"Equilibrium solving failed for {setting} setting.\n"]

    # Check assumptions
    try:
        results = validate_parameters(v_0, F, G, strict=False)
        failed = [k for k, (passed, _) in results.items() if not passed]

        if failed:
            diag.append("Failed assumptions:")
            for k in failed:
                diag.append(f"  - {k}: {results[k][1]}")
        else:
            diag.append("All assumptions passed.")

    except Exception as e:
        diag.append(f"Could not validate assumptions: {e}")

    # Check coverage
    if setting in ['NE', 'SP', 'E']:
        is_covered, threshold = check_coverage_NE(v_0, G)
        if not is_covered:
            diag.append(f"\nCoverage condition violated:")
            diag.append(f"  Need v_0 >= {threshold:.4f}, got v_0 = {v_0:.4f}")

    # Setting-specific diagnostics
    if setting == 'MM':
        if hasattr(G, 'is_symmetric'):
            if not G.is_symmetric():
                diag.append("\nMM setting requires symmetric G")

    return "\n".join(diag)


def check_welfare_consistency(ts: float, cs: float, ps: float,
                              tol: float = 1e-3) -> Tuple[bool, str]:
    """
    Verify that TS = CS + PS (accounting identity).

    Args:
        ts: Total surplus
        cs: Consumer surplus
        ps: Producer surplus
        tol: Absolute tolerance

    Returns:
        (is_consistent, message)
    """
    computed_ts = cs + ps
    diff = abs(ts - computed_ts)

    is_consistent = diff < tol

    if is_consistent:
        msg = f"Welfare accounting consistent (TS = {ts:.6f}, CS+PS = {computed_ts:.6f})"
    else:
        msg = (f"Welfare accounting ERROR: TS = {ts:.6f}, "
               f"CS+PS = {computed_ts:.6f}, diff = {diff:.6f}")

    return is_consistent, msg
