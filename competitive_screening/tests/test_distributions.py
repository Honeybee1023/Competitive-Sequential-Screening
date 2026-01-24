"""
Tests for distribution classes.

Ground Truth Engineering:
- Test actual behavior, not assumptions
- Test edge cases and numerical stability
"""

import pytest
import numpy as np
from src.core.distributions import Uniform, Normal, Logistic, ConvolutionDistribution


class TestUniform:
    def test_basic_properties(self):
        """Test basic properties of Uniform distribution."""
        U = Uniform(-1, 1)

        assert U.mean() == 0.0
        assert U.support() == (-1, 1)
        assert U.is_log_concave()
        assert U.is_symmetric(0.0)

    def test_pdf_cdf_consistency(self):
        """Test that PDF integrates to CDF."""
        U = Uniform(-1, 1)

        # CDF at midpoint should be 0.5
        assert abs(U.cdf(0.0) - 0.5) < 1e-9

        # PDF should be constant
        assert abs(U.pdf(-0.5) - U.pdf(0.5)) < 1e-9

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            Uniform(1, -1)  # a > b


class TestNormal:
    def test_basic_properties(self):
        """Test basic properties of Normal distribution."""
        N = Normal(0, 1)

        assert abs(N.mean()) < 1e-9
        assert N.is_log_concave()
        assert N.is_symmetric(0.0)

    def test_pdf_cdf_relationship(self):
        """Test PDF and CDF relationship."""
        N = Normal(0, 1)

        # At mean, CDF should be 0.5
        assert abs(N.cdf(0.0) - 0.5) < 1e-9

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            Normal(0, -1)  # negative sigma


class TestConvolutionDistribution:
    def test_mean_addition(self):
        """Test that mean of convolution is sum of means."""
        G = Uniform(-1, 1)
        F = Normal(0, 1)

        H = ConvolutionDistribution(G, F)

        # E[θ] = E[γ] + E[ε] = 0 + 0 = 0
        assert abs(H.mean()) < 1e-6

    def test_log_concavity_preservation(self):
        """Test that convolution of log-concave is log-concave."""
        G = Uniform(-1, 1)
        F = Normal(0, 1)

        H = ConvolutionDistribution(G, F)

        assert H.is_log_concave()

    @pytest.mark.slow
    def test_pdf_integration(self):
        """Test that PDF integrates to approximately 1."""
        from scipy.integrate import quad

        G = Uniform(-1, 1)
        F = Normal(0, 0.5)

        H = ConvolutionDistribution(G, F)

        # Integrate PDF over support
        theta_min, theta_max = H.support()
        integral, _ = quad(H.pdf, theta_min, theta_max, limit=100)

        assert abs(integral - 1.0) < 1e-3


class TestDistributionAssumptions:
    def test_F_assumptions(self):
        """Test that F distributions satisfy model assumptions."""
        distributions = [
            Normal(0, 1),
            Logistic(0, 1),
        ]

        for F in distributions:
            # Mean zero
            assert abs(F.mean()) < 1e-6, f"{F} should have mean 0"

            # Symmetric
            assert F.is_symmetric(0.0), f"{F} should be symmetric"

            # Log-concave
            assert F.is_log_concave(), f"{F} should be log-concave"

    def test_G_assumptions(self):
        """Test that G distributions satisfy model assumptions."""
        distributions = [
            Uniform(-1, 1),
            Uniform(0, 1),
        ]

        for G in distributions:
            # Log-concave
            assert G.is_log_concave(), f"{G} should be log-concave"

            # Bounded support
            lower, upper = G.support()
            assert np.isfinite(lower) and np.isfinite(upper), \
                f"{G} should have bounded support"
