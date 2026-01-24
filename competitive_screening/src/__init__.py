"""
Competitive Sequential Screening - Computational Framework

This package implements the models from "Competitive Sequential Screening"
by Ball, Kattwinkel, and Knoepfle.
"""

__version__ = "0.1.0"

# Export main API
from .core.distributions import (
    Distribution,
    Uniform,
    Normal,
    Logistic,
    ConvolutionDistribution
)

from .core.equilibrium import (
    EquilibriumNE,
    EquilibriumSP,
    EquilibriumE,
    EquilibriumMM,
    solve_equilibrium_NE,
    solve_equilibrium_SP,
    solve_equilibrium_E,
    solve_equilibrium_MM
)

from .core.welfare import (
    compute_total_surplus_NE,
    compute_consumer_surplus_NE,
    compute_producer_surplus_NE,
    compute_total_surplus_SP,
    compute_consumer_surplus_SP,
    compute_producer_surplus_SP,
    compute_total_surplus_E,
    compute_consumer_surplus_E,
    compute_producer_surplus_E,
    compute_total_surplus_MM,
    compute_consumer_surplus_MM,
    compute_producer_surplus_MM,
    compute_all_welfare_NE,
    compute_all_welfare_SP,
    compute_all_welfare_E,
    compute_all_welfare_MM,
    compute_all_welfare
)

from .utils.validation import (
    validate_parameters,
    check_coverage_NE,
    check_coverage_SP,
    check_welfare_consistency,
    AssumptionViolation
)

__all__ = [
    # Distributions
    'Distribution', 'Uniform', 'Normal', 'Logistic', 'ConvolutionDistribution',
    # Equilibria
    'EquilibriumNE', 'EquilibriumSP', 'EquilibriumE', 'EquilibriumMM',
    'solve_equilibrium_NE', 'solve_equilibrium_SP', 'solve_equilibrium_E', 'solve_equilibrium_MM',
    # Welfare
    'compute_total_surplus_NE', 'compute_consumer_surplus_NE', 'compute_producer_surplus_NE',
    'compute_total_surplus_SP', 'compute_consumer_surplus_SP', 'compute_producer_surplus_SP',
    'compute_total_surplus_E', 'compute_consumer_surplus_E', 'compute_producer_surplus_E',
    'compute_total_surplus_MM', 'compute_consumer_surplus_MM', 'compute_producer_surplus_MM',
    'compute_all_welfare_NE', 'compute_all_welfare_SP', 'compute_all_welfare_E', 'compute_all_welfare_MM',
    'compute_all_welfare',  # Master convenience function
    # Validation
    'validate_parameters', 'check_coverage_NE', 'check_coverage_SP',
    'check_welfare_consistency', 'AssumptionViolation'
]
