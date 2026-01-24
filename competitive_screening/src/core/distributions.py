"""
Probability distribution abstractions for the sequential screening model.

Key Requirements:
- F (taste shock): symmetric around 0, log-concave, mean 0
- G (type signal): log-concave, support Γ
- H (realized type θ = γ + ε): convolution of G and F
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
from scipy import stats, integrate
from functools import lru_cache


class Distribution(ABC):
    """
    Abstract base class for probability distributions.

    Ground Truth Engineering Note:
    We CANNOT numerically verify log-concavity in general.
    Subclasses should document their theoretical properties.
    """

    @abstractmethod
    def pdf(self, x: float) -> float:
        """Probability density function."""
        pass

    @abstractmethod
    def cdf(self, x: float) -> float:
        """Cumulative distribution function."""
        pass

    @abstractmethod
    def sf(self, x: float) -> float:
        """Survival function: 1 - CDF(x)."""
        pass

    @abstractmethod
    def quantile(self, p: float) -> float:
        """Inverse CDF (quantile function)."""
        pass

    @abstractmethod
    def support(self) -> Tuple[float, float]:
        """Return (lower, upper) bounds of support. Use ±inf if unbounded."""
        pass

    @abstractmethod
    def mean(self) -> float:
        """Expected value."""
        pass

    @abstractmethod
    def is_log_concave(self) -> bool:
        """
        Return True if distribution is KNOWN to be log-concave.

        WARNING: This is a declaration, not a verification.
        For custom distributions, users must ensure this mathematically.
        """
        pass

    def hazard_rate(self, x: float) -> float:
        """
        Hazard rate: h(x) = f(x) / (1 - F(x))

        Used in equilibrium conditions (e.g., Proposition 2).
        """
        pdf_val = self.pdf(x)
        sf_val = self.sf(x)
        if sf_val < 1e-12:  # Numerical stability
            return np.inf
        return pdf_val / sf_val

    def inverse_hazard_rate(self, x: float) -> float:
        """
        Inverse hazard rate: (1 - F(x)) / f(x)

        Appears in strike price formulas.
        """
        pdf_val = self.pdf(x)
        if pdf_val < 1e-12:
            return np.inf
        return self.sf(x) / pdf_val

    def virtual_valuation(self, x: float) -> float:
        """
        Virtual valuation from mechanism design: x - (1-F(x))/f(x)

        PLACEHOLDER: Verify if this is used in the paper.
        """
        return x - self.inverse_hazard_rate(x)


class Uniform(Distribution):
    """Uniform distribution U(a, b). Log-concave by definition."""

    def __init__(self, a: float, b: float):
        if a >= b:
            raise ValueError(f"Invalid bounds: {a} >= {b}")
        self.a = a
        self.b = b
        self._dist = stats.uniform(loc=a, scale=b-a)

    def pdf(self, x: float) -> float:
        return self._dist.pdf(x)

    def cdf(self, x: float) -> float:
        return self._dist.cdf(x)

    def sf(self, x: float) -> float:
        return self._dist.sf(x)

    def quantile(self, p: float) -> float:
        return self._dist.ppf(p)

    def support(self) -> Tuple[float, float]:
        return (self.a, self.b)

    def mean(self) -> float:
        return (self.a + self.b) / 2

    def is_log_concave(self) -> bool:
        return True

    def is_symmetric(self, center: float = 0.0) -> bool:
        """Check if distribution is symmetric around center."""
        return abs((self.a + self.b) / 2 - center) < 1e-9


class Normal(Distribution):
    """Normal distribution N(μ, σ²). Log-concave by definition."""

    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive: {sigma}")
        self.mu = mu
        self.sigma = sigma
        self._dist = stats.norm(loc=mu, scale=sigma)

    def pdf(self, x: float) -> float:
        return self._dist.pdf(x)

    def cdf(self, x: float) -> float:
        return self._dist.cdf(x)

    def sf(self, x: float) -> float:
        return self._dist.sf(x)

    def quantile(self, p: float) -> float:
        return self._dist.ppf(p)

    def support(self) -> Tuple[float, float]:
        return (-np.inf, np.inf)

    def mean(self) -> float:
        return self.mu

    def is_log_concave(self) -> bool:
        return True

    def is_symmetric(self, center: Optional[float] = None) -> bool:
        if center is None:
            center = self.mu
        return abs(self.mu - center) < 1e-9


class Logistic(Distribution):
    """Logistic distribution. Log-concave by definition."""

    def __init__(self, mu: float, s: float):
        if s <= 0:
            raise ValueError(f"Scale parameter s must be positive: {s}")
        self.mu = mu
        self.s = s
        self._dist = stats.logistic(loc=mu, scale=s)

    def pdf(self, x: float) -> float:
        return self._dist.pdf(x)

    def cdf(self, x: float) -> float:
        return self._dist.cdf(x)

    def sf(self, x: float) -> float:
        return self._dist.sf(x)

    def quantile(self, p: float) -> float:
        return self._dist.ppf(p)

    def support(self) -> Tuple[float, float]:
        return (-np.inf, np.inf)

    def mean(self) -> float:
        return self.mu

    def is_log_concave(self) -> bool:
        return True

    def is_symmetric(self, center: Optional[float] = None) -> bool:
        if center is None:
            center = self.mu
        return abs(self.mu - center) < 1e-9


class ConvolutionDistribution(Distribution):
    """
    Distribution of θ = γ + ε where γ ~ G, ε ~ F.

    Used for H in spot pricing (Proposition 2).

    PERFORMANCE WARNING: PDF/CDF computed via numerical convolution.
    Should be cached aggressively.
    """

    def __init__(self, G: Distribution, F: Distribution, cache_size: int = 1000):
        self.G = G
        self.F = F
        self._cache_size = cache_size

        # Determine support
        g_min, g_max = G.support()
        f_min, f_max = F.support()
        self._support = (g_min + f_min, g_max + f_max)

    @lru_cache(maxsize=1000)
    def pdf(self, x: float) -> float:
        """
        h(x) = ∫ g(γ) * f(x - γ) dγ

        PLACEHOLDER: Optimize integration bounds based on actual supports.
        """
        def integrand(gamma):
            return self.G.pdf(gamma) * self.F.pdf(x - gamma)

        g_min, g_max = self.G.support()
        result, _ = integrate.quad(integrand, g_min, g_max, limit=50)
        return result

    @lru_cache(maxsize=1000)
    def cdf(self, x: float) -> float:
        """
        H(x) = ∫ G(x - ε) * f(ε) dε

        PLACEHOLDER: Check if there's a more efficient formulation.
        """
        def integrand(epsilon):
            return self.G.cdf(x - epsilon) * self.F.pdf(epsilon)

        f_min, f_max = self.F.support()
        result, _ = integrate.quad(integrand, f_min, f_max, limit=50)
        return result

    def sf(self, x: float) -> float:
        return 1.0 - self.cdf(x)

    def quantile(self, p: float) -> float:
        """Numerical inversion of CDF. SLOW."""
        from scipy.optimize import brentq
        return brentq(lambda x: self.cdf(x) - p, self._support[0], self._support[1])

    def support(self) -> Tuple[float, float]:
        return self._support

    def mean(self) -> float:
        return self.G.mean() + self.F.mean()

    def is_log_concave(self) -> bool:
        """
        Convolution of log-concave distributions is log-concave.
        This is a theorem (Prékopa, 1973).
        """
        return self.G.is_log_concave() and self.F.is_log_concave()


def verify_distribution_assumptions(F: Distribution, G: Distribution) -> dict:
    """
    Check model assumptions for F and G.

    Returns:
        dict: Keys are assumption names, values are (bool, str) tuples
              where bool is pass/fail and str is explanation.
    """
    results = {}

    # F assumptions
    results['F_log_concave'] = (
        F.is_log_concave(),
        "F must be log-concave (declared by class)"
    )

    results['F_mean_zero'] = (
        abs(F.mean()) < 1e-6,
        f"F must have mean 0 (actual: {F.mean():.6f})"
    )

    # Check symmetry for Normal/Logistic
    if hasattr(F, 'is_symmetric'):
        results['F_symmetric'] = (
            F.is_symmetric(0.0),
            "F should be symmetric around 0"
        )

    # G assumptions
    results['G_log_concave'] = (
        G.is_log_concave(),
        "G must be log-concave (declared by class)"
    )

    return results
