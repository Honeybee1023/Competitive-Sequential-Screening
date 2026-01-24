"""
Tests validating propositions from the paper.

These tests verify theoretical claims from "Competitive Sequential Screening"
by Ball, Kattwinkel, and Knoepfle.
"""

import pytest
import numpy as np
from src import (
    Uniform, Normal, Logistic,
    solve_equilibrium_NE, solve_equilibrium_SP, solve_equilibrium_E, solve_equilibrium_MM,
    compute_total_surplus_NE, compute_total_surplus_SP,
    compute_total_surplus_E, compute_total_surplus_MM,
    compute_consumer_surplus_SP, compute_producer_surplus_SP,
    compute_consumer_surplus_MM, compute_producer_surplus_MM
)


class TestProposition2_SpotPricing:
    """Test Proposition 2: Spot Pricing equilibrium (page 10)."""

    def test_symmetric_case_theta_star_zero(self):
        """
        For symmetric F (mean 0) and symmetric G, θ* should be 0.

        Reference: Discussion on page 10, "If m = 0, then θ* = 0"
        where m is the median of H.
        """
        v_0 = 6.0
        G = Uniform(-1, 1)  # Symmetric around 0
        F = Normal(0, 1)     # Symmetric around 0

        eq_SP = solve_equilibrium_SP(v_0, F, G)

        assert abs(eq_SP.theta_star) < 1e-3, \
            f"For symmetric case, θ* should be ~0, got {eq_SP.theta_star}"

    def test_symmetric_prices_equal(self):
        """For symmetric case, p_A* should equal p_B* when θ* = 0."""
        v_0 = 6.0
        G = Uniform(-1, 1)
        F = Normal(0, 1)

        eq_SP = solve_equilibrium_SP(v_0, F, G)

        assert abs(eq_SP.p_A - eq_SP.p_B) < 1e-3, \
            f"For symmetric case, prices should be equal: p_A={eq_SP.p_A}, p_B={eq_SP.p_B}"

    def test_efficient_allocation_symmetric(self):
        """
        For symmetric case (θ* = 0), allocation is efficient.

        Total surplus should equal efficient benchmark (MM).
        """
        v_0 = 6.0
        G = Uniform(-1, 1)
        F = Normal(0, 1)

        eq_SP = solve_equilibrium_SP(v_0, F, G)
        eq_MM = solve_equilibrium_MM(v_0, F, G)

        TS_SP = compute_total_surplus_SP(eq_SP)
        TS_MM = compute_total_surplus_MM(eq_MM)

        assert abs(TS_SP - TS_MM) < 1e-2, \
            f"SP should be efficient for symmetric case: TS_SP={TS_SP}, TS_MM={TS_MM}"

    def test_welfare_accounting(self):
        """Verify TS = CS + PS for spot pricing."""
        v_0 = 6.0
        G = Uniform(-1, 1)
        F = Normal(0, 1)

        eq_SP = solve_equilibrium_SP(v_0, F, G)

        TS = compute_total_surplus_SP(eq_SP)
        CS = compute_consumer_surplus_SP(eq_SP)
        PS = compute_producer_surplus_SP(eq_SP)

        assert abs(TS - (CS + PS)) < 1e-3, \
            f"Welfare accounting failed: TS={TS}, CS+PS={CS+PS}"


class TestProposition4_EfficiencyComparison:
    """Test Proposition 4: NE more efficient than E for symmetric G (page 13)."""

    def test_ne_more_efficient_than_e_symmetric(self):
        """
        Proposition 4: If G is symmetric, NE allocation is pointwise more
        efficient than E allocation.

        Therefore: TS^NE > TS^E

        Reference: Proposition 4, page 13
        """
        v_0 = 6.0
        G = Uniform(-1, 1)  # Symmetric
        F = Normal(0, 1)

        eq_NE = solve_equilibrium_NE(v_0, F, G)
        eq_E = solve_equilibrium_E(v_0, F, G)

        TS_NE = compute_total_surplus_NE(eq_NE)
        TS_E = compute_total_surplus_E(eq_E)

        assert TS_NE > TS_E, \
            f"Proposition 4 violated: TS_NE ({TS_NE}) should be > TS_E ({TS_E})"

        # Check magnitude of difference is reasonable
        efficiency_gain = (TS_NE - TS_E) / TS_E * 100
        assert efficiency_gain > 1, \
            f"Efficiency gain should be substantial, got {efficiency_gain:.2f}%"

    def test_proposition4_multiple_distributions(self):
        """Test Proposition 4 holds for different symmetric G distributions."""
        v_0 = 6.0
        F = Normal(0, 1)

        # Test with different symmetric G distributions
        test_cases = [
            ("Uniform[-1,1]", Uniform(-1, 1)),
            ("Uniform[-2,2]", Uniform(-2, 2)),
            ("Uniform[-0.5,0.5]", Uniform(-0.5, 0.5)),
        ]

        for name, G in test_cases:
            eq_NE = solve_equilibrium_NE(v_0, F, G)
            eq_E = solve_equilibrium_E(v_0, F, G)

            TS_NE = compute_total_surplus_NE(eq_NE)
            TS_E = compute_total_surplus_E(eq_E)

            assert TS_NE > TS_E, \
                f"Proposition 4 failed for G={name}: TS_NE={TS_NE}, TS_E={TS_E}"


class TestProposition5_Monopoly:
    """Test Proposition 5: Multi-Good Monopoly (page 15)."""

    def test_efficient_allocation(self):
        """
        MM achieves efficient allocation (prices = 0).
        Therefore: TS^MM should be maximal.
        """
        v_0 = 6.0
        G = Uniform(-1, 1)
        F = Normal(0, 1)

        eq_NE = solve_equilibrium_NE(v_0, F, G)
        eq_SP = solve_equilibrium_SP(v_0, F, G)
        eq_E = solve_equilibrium_E(v_0, F, G)
        eq_MM = solve_equilibrium_MM(v_0, F, G)

        TS_NE = compute_total_surplus_NE(eq_NE)
        TS_SP = compute_total_surplus_SP(eq_SP)
        TS_E = compute_total_surplus_E(eq_E)
        TS_MM = compute_total_surplus_MM(eq_MM)

        # MM should be at least as efficient as all others
        assert TS_MM >= TS_NE - 1e-3, f"TS_MM ({TS_MM}) < TS_NE ({TS_NE})"
        assert TS_MM >= TS_SP - 1e-3, f"TS_MM ({TS_MM}) < TS_SP ({TS_SP})"
        assert TS_MM >= TS_E - 1e-3, f"TS_MM ({TS_MM}) < TS_E ({TS_E})"

    def test_full_surplus_extraction(self):
        """
        Proposition 5: Monopoly extracts all surplus.
        Therefore: CS = 0, PS = TS
        """
        v_0 = 6.0
        G = Uniform(-1, 1)
        F = Normal(0, 1)

        eq_MM = solve_equilibrium_MM(v_0, F, G)

        TS = compute_total_surplus_MM(eq_MM)
        CS = compute_consumer_surplus_MM(eq_MM)
        PS = compute_producer_surplus_MM(eq_MM)

        assert abs(CS) < 1e-6, f"CS should be 0, got {CS}"
        assert abs(PS - TS) < 1e-3, f"PS should equal TS: PS={PS}, TS={TS}"

    def test_requires_symmetric_g(self):
        """MM is only defined for symmetric G."""
        v_0 = 6.0
        G_asymmetric = Uniform(0, 2)  # Not symmetric around 0
        F = Normal(0, 1)

        with pytest.raises(ValueError, match="symmetric"):
            solve_equilibrium_MM(v_0, F, G_asymmetric)


class TestProposition6_MonopolySurplus:
    """Test Proposition 6: Monopoly surplus comparisons (page 16)."""

    def test_mm_lowest_cs(self):
        """
        Proposition 6: MM yields strictly lower consumer surplus than
        NE, SP, and E.
        """
        v_0 = 6.0
        G = Uniform(-1, 1)
        F = Normal(0, 1)

        eq_SP = solve_equilibrium_SP(v_0, F, G)
        eq_MM = solve_equilibrium_MM(v_0, F, G)

        CS_SP = compute_consumer_surplus_SP(eq_SP)
        CS_MM = compute_consumer_surplus_MM(eq_MM)

        # MM extracts all surplus, so CS_MM = 0 < CS_SP
        assert CS_MM < CS_SP, \
            f"MM should have lower CS: CS_MM={CS_MM}, CS_SP={CS_SP}"

    def test_mm_highest_ps(self):
        """
        Proposition 6: MM yields strictly higher producer surplus than
        NE, SP, and E.
        """
        v_0 = 6.0
        G = Uniform(-1, 1)
        F = Normal(0, 1)

        eq_SP = solve_equilibrium_SP(v_0, F, G)
        eq_MM = solve_equilibrium_MM(v_0, F, G)

        PS_SP = compute_producer_surplus_SP(eq_SP)
        PS_MM = compute_producer_surplus_MM(eq_MM)

        # MM extracts all surplus
        assert PS_MM > PS_SP, \
            f"MM should have higher PS: PS_MM={PS_MM}, PS_SP={PS_SP}"


class TestEfficiencyRankings:
    """General tests for efficiency rankings across settings."""

    def test_efficient_settings(self):
        """SP (symmetric) and MM should both be efficient."""
        v_0 = 6.0
        G = Uniform(-1, 1)
        F = Normal(0, 1)

        eq_SP = solve_equilibrium_SP(v_0, F, G)
        eq_MM = solve_equilibrium_MM(v_0, F, G)

        TS_SP = compute_total_surplus_SP(eq_SP)
        TS_MM = compute_total_surplus_MM(eq_MM)

        # Should be approximately equal (both efficient)
        assert abs(TS_SP - TS_MM) < 1e-2, \
            f"SP and MM should be equally efficient: TS_SP={TS_SP}, TS_MM={TS_MM}"

    def test_e_least_efficient(self):
        """
        Exclusive contracting should be least efficient
        (most lock-in distortion).
        """
        v_0 = 6.0
        G = Uniform(-1, 1)
        F = Normal(0, 1)

        eq_NE = solve_equilibrium_NE(v_0, F, G)
        eq_SP = solve_equilibrium_SP(v_0, F, G)
        eq_E = solve_equilibrium_E(v_0, F, G)
        eq_MM = solve_equilibrium_MM(v_0, F, G)

        TS_NE = compute_total_surplus_NE(eq_NE)
        TS_SP = compute_total_surplus_SP(eq_SP)
        TS_E = compute_total_surplus_E(eq_E)
        TS_MM = compute_total_surplus_MM(eq_MM)

        # E should be least efficient
        assert TS_E < TS_NE, f"E should be less efficient than NE"
        assert TS_E < TS_SP, f"E should be less efficient than SP"
        assert TS_E < TS_MM, f"E should be less efficient than MM"

    def test_ne_intermediate_efficiency(self):
        """NE should be between E (least efficient) and SP/MM (efficient)."""
        v_0 = 6.0
        G = Uniform(-1, 1)
        F = Normal(0, 1)

        eq_NE = solve_equilibrium_NE(v_0, F, G)
        eq_SP = solve_equilibrium_SP(v_0, F, G)
        eq_E = solve_equilibrium_E(v_0, F, G)

        TS_NE = compute_total_surplus_NE(eq_NE)
        TS_SP = compute_total_surplus_SP(eq_SP)
        TS_E = compute_total_surplus_E(eq_E)

        # Ranking: SP ≈ MM > NE > E
        assert TS_E < TS_NE < TS_SP + 1e-2, \
            f"Expected E < NE < SP, got TS_E={TS_E}, TS_NE={TS_NE}, TS_SP={TS_SP}"


class TestParameterRobustness:
    """Test that rankings are robust across parameter values."""

    @pytest.mark.parametrize("v_0", [4.0, 6.0, 8.0, 10.0])
    def test_proposition4_various_v0(self, v_0):
        """Proposition 4 should hold for different v_0 values."""
        G = Uniform(-1, 1)
        F = Normal(0, 1)

        eq_NE = solve_equilibrium_NE(v_0, F, G)
        eq_E = solve_equilibrium_E(v_0, F, G)

        TS_NE = compute_total_surplus_NE(eq_NE)
        TS_E = compute_total_surplus_E(eq_E)

        assert TS_NE > TS_E, \
            f"Prop 4 failed for v_0={v_0}: TS_NE={TS_NE}, TS_E={TS_E}"

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
    def test_mm_efficient_various_sigma(self, sigma):
        """MM should always be efficient regardless of σ."""
        v_0 = 6.0
        G = Uniform(-1, 1)
        F = Normal(0, sigma)

        eq_NE = solve_equilibrium_NE(v_0, F, G)
        eq_MM = solve_equilibrium_MM(v_0, F, G)

        TS_NE = compute_total_surplus_NE(eq_NE)
        TS_MM = compute_total_surplus_MM(eq_MM)

        # MM should be at least as efficient
        assert TS_MM >= TS_NE - 1e-3, \
            f"MM not efficient for σ={sigma}: TS_MM={TS_MM}, TS_NE={TS_NE}"


class TestNumericalStability:
    """Test that implementations are numerically stable."""

    def test_small_sigma(self):
        """Test with very precise signal (small σ)."""
        v_0 = 6.0
        G = Uniform(-1, 1)
        F = Normal(0, 0.1)  # Very small noise

        # Should not raise errors
        eq_NE = solve_equilibrium_NE(v_0, F, G)
        eq_SP = solve_equilibrium_SP(v_0, F, G)
        eq_E = solve_equilibrium_E(v_0, F, G)
        eq_MM = solve_equilibrium_MM(v_0, F, G)

        # Should produce finite TS values
        assert np.isfinite(compute_total_surplus_NE(eq_NE))
        assert np.isfinite(compute_total_surplus_SP(eq_SP))
        assert np.isfinite(compute_total_surplus_E(eq_E))
        assert np.isfinite(compute_total_surplus_MM(eq_MM))

    def test_large_sigma(self):
        """Test with very imprecise signal (large σ)."""
        v_0 = 10.0  # Need larger v_0 for coverage
        G = Uniform(-1, 1)
        F = Normal(0, 5.0)  # Large noise

        # Should not raise errors
        eq_NE = solve_equilibrium_NE(v_0, F, G)
        eq_SP = solve_equilibrium_SP(v_0, F, G)
        eq_E = solve_equilibrium_E(v_0, F, G)
        eq_MM = solve_equilibrium_MM(v_0, F, G)

        # Should produce finite TS values
        assert np.isfinite(compute_total_surplus_NE(eq_NE))
        assert np.isfinite(compute_total_surplus_SP(eq_SP))
        assert np.isfinite(compute_total_surplus_E(eq_E))
        assert np.isfinite(compute_total_surplus_MM(eq_MM))

    def test_narrow_type_distribution(self):
        """Test with narrow type distribution."""
        v_0 = 6.0
        G = Uniform(-0.1, 0.1)  # Very narrow
        F = Normal(0, 1)

        # Should not raise errors
        eq_NE = solve_equilibrium_NE(v_0, F, G)
        eq_SP = solve_equilibrium_SP(v_0, F, G)
        eq_E = solve_equilibrium_E(v_0, F, G)
        eq_MM = solve_equilibrium_MM(v_0, F, G)

        # Compute TS for all settings
        TS_NE = compute_total_surplus_NE(eq_NE)
        TS_SP = compute_total_surplus_SP(eq_SP)
        TS_E = compute_total_surplus_E(eq_E)
        TS_MM = compute_total_surplus_MM(eq_MM)

        # With narrow G, SP and MM should be close (both efficient)
        # NE should be close to them as well
        # E can still be less efficient due to lock-in even with narrow G
        assert abs(TS_SP - TS_MM) < 0.1, \
            f"SP and MM should be nearly equal for narrow G: SP={TS_SP:.3f}, MM={TS_MM:.3f}"
        assert abs(TS_NE - TS_SP) < 0.1, \
            f"NE and SP should be close for narrow G: NE={TS_NE:.3f}, SP={TS_SP:.3f}"
        # E should still be less efficient (lock-in inefficiency persists)
        assert TS_E < TS_SP, \
            f"E should be less efficient even with narrow G: E={TS_E:.3f} < SP={TS_SP:.3f}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
