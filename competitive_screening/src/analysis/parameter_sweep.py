"""
Parameter sweep utilities for Total Surplus analysis.

This module provides functions to sweep over parameter ranges and compute
total surplus across all four market settings.
"""

from typing import List, Dict, Tuple, Callable
import numpy as np
from dataclasses import dataclass
import warnings

from ..core.distributions import Distribution, Uniform, Normal, Logistic
from ..core.equilibrium import (
    solve_equilibrium_NE, solve_equilibrium_SP,
    solve_equilibrium_E, solve_equilibrium_MM
)
from ..core.welfare import (
    compute_total_surplus_NE, compute_total_surplus_SP,
    compute_total_surplus_E, compute_total_surplus_MM
)


@dataclass
class SweepResult:
    """Results from a parameter sweep."""
    parameter_name: str
    parameter_values: np.ndarray
    TS_NE: np.ndarray
    TS_SP: np.ndarray
    TS_E: np.ndarray
    TS_MM: np.ndarray

    def get_ranking(self, index: int) -> List[Tuple[str, float]]:
        """Get TS ranking at given index, sorted high to low."""
        results = [
            ('NE', self.TS_NE[index]),
            ('SP', self.TS_SP[index]),
            ('E', self.TS_E[index]),
            ('MM', self.TS_MM[index])
        ]
        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_all_rankings(self) -> List[List[str]]:
        """Get TS rankings (setting names only) for all parameter values."""
        return [[s for s, _ in self.get_ranking(i)]
                for i in range(len(self.parameter_values))]


def sweep_v0(v0_values: List[float],
             F: Distribution,
             G: Distribution,
             verbose: bool = True) -> SweepResult:
    """
    Sweep over v_0 values and compute TS for all settings.

    Args:
        v0_values: List of average valuation values to test
        F: Taste shock distribution (fixed)
        G: Type distribution (fixed)
        verbose: Print progress

    Returns:
        SweepResult with TS values for each setting

    Example:
        >>> from src import Uniform, Normal
        >>> G = Uniform(-1, 1)
        >>> F = Normal(0, 1)
        >>> result = sweep_v0([2, 4, 6, 8, 10], F, G)
        >>> rankings = result.get_all_rankings()
    """
    n = len(v0_values)
    TS_NE = np.zeros(n)
    TS_SP = np.zeros(n)
    TS_E = np.zeros(n)
    TS_MM = np.zeros(n)

    for i, v0 in enumerate(v0_values):
        if verbose:
            print(f"Computing v_0 = {v0:.2f} ({i+1}/{n})...", end=" ")

        try:
            # Suppress warnings for cleaner output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                eq_NE = solve_equilibrium_NE(v0, F, G)
                TS_NE[i] = compute_total_surplus_NE(eq_NE)

                eq_SP = solve_equilibrium_SP(v0, F, G)
                TS_SP[i] = compute_total_surplus_SP(eq_SP)

                eq_E = solve_equilibrium_E(v0, F, G)
                TS_E[i] = compute_total_surplus_E(eq_E)

                eq_MM = solve_equilibrium_MM(v0, F, G)
                TS_MM[i] = compute_total_surplus_MM(eq_MM)

            if verbose:
                ranking = SweepResult('v0', np.array([v0]), np.array([TS_NE[i]]),
                                     np.array([TS_SP[i]]), np.array([TS_E[i]]),
                                     np.array([TS_MM[i]])).get_ranking(0)
                print(f"✓ Ranking: {[s for s,_ in ranking]}")

        except Exception as e:
            if verbose:
                print(f"✗ Failed: {e}")
            TS_NE[i] = TS_SP[i] = TS_E[i] = TS_MM[i] = np.nan

    return SweepResult('v_0', np.array(v0_values), TS_NE, TS_SP, TS_E, TS_MM)


def sweep_information_precision(sigma_values: List[float],
                                v0: float,
                                G: Distribution,
                                verbose: bool = True) -> SweepResult:
    """
    Sweep over taste shock precision (σ in F ~ N(0, σ²)).

    Lower σ = more informative pre-contractual signal (γ predicts θ better)
    Higher σ = less informative signal (large taste shock)

    Args:
        sigma_values: List of standard deviations for F ~ Normal(0, σ²)
        v0: Average valuation (fixed)
        G: Type distribution (fixed)
        verbose: Print progress

    Returns:
        SweepResult with TS values for each σ

    Example:
        >>> result = sweep_information_precision([0.1, 0.5, 1.0, 2.0, 5.0],
        ...                                       v0=6, G=Uniform(-1,1))
    """
    n = len(sigma_values)
    TS_NE = np.zeros(n)
    TS_SP = np.zeros(n)
    TS_E = np.zeros(n)
    TS_MM = np.zeros(n)

    for i, sigma in enumerate(sigma_values):
        if verbose:
            print(f"Computing σ = {sigma:.2f} ({i+1}/{n})...", end=" ")

        try:
            F = Normal(0, sigma)  # Create new F with this σ

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                eq_NE = solve_equilibrium_NE(v0, F, G)
                TS_NE[i] = compute_total_surplus_NE(eq_NE)

                eq_SP = solve_equilibrium_SP(v0, F, G)
                TS_SP[i] = compute_total_surplus_SP(eq_SP)

                eq_E = solve_equilibrium_E(v0, F, G)
                TS_E[i] = compute_total_surplus_E(eq_E)

                eq_MM = solve_equilibrium_MM(v0, F, G)
                TS_MM[i] = compute_total_surplus_MM(eq_MM)

            if verbose:
                ranking = SweepResult('sigma', np.array([sigma]), np.array([TS_NE[i]]),
                                     np.array([TS_SP[i]]), np.array([TS_E[i]]),
                                     np.array([TS_MM[i]])).get_ranking(0)
                print(f"✓ Ranking: {[s for s,_ in ranking]}")

        except Exception as e:
            if verbose:
                print(f"✗ Failed: {e}")
            TS_NE[i] = TS_SP[i] = TS_E[i] = TS_MM[i] = np.nan

    return SweepResult('σ (F precision)', np.array(sigma_values), TS_NE, TS_SP, TS_E, TS_MM)


def sweep_type_distribution_width(width_values: List[float],
                                  v0: float,
                                  F: Distribution,
                                  center: float = 0.0,
                                  verbose: bool = True) -> SweepResult:
    """
    Sweep over type distribution width (G ~ Uniform[center-w, center+w]).

    Larger width = more ex-ante heterogeneity in consumer preferences

    Args:
        width_values: List of half-widths for G ~ Uniform
        v0: Average valuation (fixed)
        F: Taste shock distribution (fixed)
        center: Center point of uniform distribution
        verbose: Print progress

    Returns:
        SweepResult with TS values for each width

    Example:
        >>> result = sweep_type_distribution_width([0.5, 1.0, 2.0, 5.0],
        ...                                         v0=6, F=Normal(0,1))
    """
    n = len(width_values)
    TS_NE = np.zeros(n)
    TS_SP = np.zeros(n)
    TS_E = np.zeros(n)
    TS_MM = np.zeros(n)

    for i, width in enumerate(width_values):
        if verbose:
            print(f"Computing width = {width:.2f} ({i+1}/{n})...", end=" ")

        try:
            G = Uniform(center - width, center + width)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                eq_NE = solve_equilibrium_NE(v0, F, G)
                TS_NE[i] = compute_total_surplus_NE(eq_NE)

                eq_SP = solve_equilibrium_SP(v0, F, G)
                TS_SP[i] = compute_total_surplus_SP(eq_SP)

                eq_E = solve_equilibrium_E(v0, F, G)
                TS_E[i] = compute_total_surplus_E(eq_E)

                eq_MM = solve_equilibrium_MM(v0, F, G)
                TS_MM[i] = compute_total_surplus_MM(eq_MM)

            if verbose:
                ranking = SweepResult('width', np.array([width]), np.array([TS_NE[i]]),
                                     np.array([TS_SP[i]]), np.array([TS_E[i]]),
                                     np.array([TS_MM[i]])).get_ranking(0)
                print(f"✓ Ranking: {[s for s,_ in ranking]}")

        except Exception as e:
            if verbose:
                print(f"✗ Failed: {e}")
            TS_NE[i] = TS_SP[i] = TS_E[i] = TS_MM[i] = np.nan

    return SweepResult('G width', np.array(width_values), TS_NE, TS_SP, TS_E, TS_MM)


def compare_distributions(scenarios: Dict[str, Tuple[float, Distribution, Distribution]],
                         verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Compare TS across multiple distribution scenarios.

    Args:
        scenarios: Dict mapping scenario name to (v0, F, G) tuple
        verbose: Print results

    Returns:
        Dict mapping scenario name to dict of TS values by setting

    Example:
        >>> scenarios = {
        ...     'Baseline': (6, Normal(0, 1), Uniform(-1, 1)),
        ...     'High info': (6, Normal(0, 0.5), Uniform(-1, 1)),
        ...     'Low info': (6, Normal(0, 2), Uniform(-1, 1))
        ... }
        >>> results = compare_distributions(scenarios)
    """
    results = {}

    for name, (v0, F, G) in scenarios.items():
        if verbose:
            print(f"\nScenario: {name}")
            print(f"  v0={v0}, F={F.__class__.__name__}, G={G.__class__.__name__}")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                eq_NE = solve_equilibrium_NE(v0, F, G)
                TS_NE = compute_total_surplus_NE(eq_NE)

                eq_SP = solve_equilibrium_SP(v0, F, G)
                TS_SP = compute_total_surplus_SP(eq_SP)

                eq_E = solve_equilibrium_E(v0, F, G)
                TS_E = compute_total_surplus_E(eq_E)

                eq_MM = solve_equilibrium_MM(v0, F, G)
                TS_MM = compute_total_surplus_MM(eq_MM)

            results[name] = {
                'NE': TS_NE,
                'SP': TS_SP,
                'E': TS_E,
                'MM': TS_MM
            }

            if verbose:
                ranking = sorted(results[name].items(), key=lambda x: x[1], reverse=True)
                print("  Rankings:")
                for i, (setting, ts) in enumerate(ranking, 1):
                    print(f"    {i}. {setting}: {ts:.4f}")

        except Exception as e:
            if verbose:
                print(f"  ✗ Failed: {e}")
            results[name] = {'NE': np.nan, 'SP': np.nan, 'E': np.nan, 'MM': np.nan}

    return results


def analyze_ranking_transitions(sweep_result: SweepResult,
                                threshold: float = 0.01) -> List[Tuple[int, str]]:
    """
    Identify parameter values where rankings change.

    Args:
        sweep_result: Result from a parameter sweep
        threshold: Minimum TS difference to consider rankings different

    Returns:
        List of (index, description) tuples for transition points

    Example:
        >>> result = sweep_v0([2, 4, 6, 8, 10], F, G)
        >>> transitions = analyze_ranking_transitions(result)
        >>> for idx, desc in transitions:
        ...     print(f"At {result.parameter_values[idx]}: {desc}")
    """
    rankings = sweep_result.get_all_rankings()
    transitions = []

    for i in range(1, len(rankings)):
        if rankings[i] != rankings[i-1]:
            # Ranking changed
            prev_ranking = sweep_result.get_ranking(i-1)
            curr_ranking = sweep_result.get_ranking(i)

            desc = (f"Ranking changed from {[s for s,_ in prev_ranking]} "
                   f"to {[s for s,_ in curr_ranking]}")
            transitions.append((i, desc))

    return transitions
