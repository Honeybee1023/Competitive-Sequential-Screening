"""
Global configuration for numerical tolerances.

Per user request: Single variable that can be adjusted to update all tolerances.
"""

# Master tolerance setting
NUMERICAL_TOLERANCE = 1e-8

# Derived tolerances for different operations
ROOT_FINDING_TOL = NUMERICAL_TOLERANCE
INTEGRATION_REL_TOL = NUMERICAL_TOLERANCE * 100  # Slightly looser for integration
INTEGRATION_ABS_TOL = NUMERICAL_TOLERANCE * 100
ASSUMPTION_CHECK_TOL = NUMERICAL_TOLERANCE * 100

# Integration parameters
QUAD_LIMIT = 50  # Number of quadrature points for adaptive integration

# Parameter ranges (reasonable defaults, can be adjusted)
DEFAULT_V0_RANGE = (1.0, 20.0)  # Average valuation range
DEFAULT_GAMMA_RANGE = (-5.0, 5.0)  # Type distribution support
DEFAULT_SIGMA_RANGE = (0.1, 5.0)  # Taste shock standard deviation range
