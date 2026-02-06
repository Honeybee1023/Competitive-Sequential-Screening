# Competitive Sequential Screening

> **A complete computational framework for analyzing welfare across competing market designs with sequential information revelation**

Implementation of models from ["Competitive Sequential Screening"](https://arxiv.org/abs/2409.02878) by Ball, Kattwinkel, and Knoepfle (2024).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Overview

This framework compares **welfare outcomes** across four market settings where firms can offer contracts before consumers learn their preferences:

| Market Setting | Consumer Choice | Key Feature |
|---------------|-----------------|-------------|
| 🟡 **Non-Exclusive (NE)** | Subscribe to both firms, choose later | Maximum flexibility |
| 🟢 **Spot Pricing (SP)** | No subscriptions, full information | Efficient benchmark |
| 🔴 **Exclusive (E)** | Lock-in to one firm before learning | Allocation inefficiency |
| 🔵 **Multi-Good Monopoly (MM)** | Single firm, both products | Full surplus extraction |

### Key Capabilities

✅ **Complete Equilibrium Computation**
- Strike prices and subscription schedules for all settings
- Handles non-exclusive, exclusive, spot pricing, and monopoly
- Accurate implementation of Theorems 1-5 from the paper

✅ **Precise Welfare Analysis**
- Total Surplus (TS), Consumer Surplus (CS), Producer Surplus (PS)
- All metrics satisfy welfare identity: `CS + PS = TS`
- Subscription schedules via envelope theorem

✅ **Interactive Web Interface**
- Real-time parameter exploration with Streamlit
- Dynamic visualization of welfare rankings
- Distribution preview and formula reference
- Export results to JSON/CSV

✅ **Flexible Distribution Support**
- Uniform, Normal, and Logistic distributions
- Log-concave distributions (as required by theory)
- Symmetric and asymmetric configurations

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/competitive-sequential-screening.git
cd competitive-sequential-screening

# Install dependencies
pip install -r requirements.txt
```

### Launch Interactive Frontend

```bash
./run_app.sh
```

Or manually:
```bash
streamlit run competitive_screening/frontend/app.py
```

The web interface will open at `http://localhost:8501`

### Python API Usage

```python
from competitive_screening.src import Uniform, Normal, compute_all_welfare

# Define parameters
v_0 = 6.0                    # Average valuation
G = Uniform(-1.0, 1.0)       # Type distribution (ex-ante heterogeneity)
F = Normal(0.0, 1.0)         # Taste shock (information precision)

# Compute welfare for all market settings
results = compute_all_welfare(v_0, F, G)

# Access results
print(f"Non-Exclusive TS: {results['TS_NE']:.4f}")
print(f"Spot Pricing TS:  {results['TS_SP']:.4f}")
print(f"Exclusive TS:     {results['TS_E']:.4f}")
print(f"Monopoly TS:      {results['TS_MM']:.4f}")

# Consumer and Producer Surplus available for all settings
print(f"\nNE Consumer Surplus: {results['CS_NE']:.4f}")
print(f"NE Producer Surplus: {results['PS_NE']:.4f}")
```

---

## 📊 Features

### 1. Equilibrium Computation

Complete implementation of all equilibrium concepts from the paper:

**Non-Exclusive (Theorem 1)**
- Strike prices: `p*_A(γ) = 2G(γ)/g(γ)`
- Subscription schedules via envelope theorem
- Boundary utility + demand integral

**Exclusive (Propositions 3-4)**
- Critical type γ̂ solving indifference
- Monopoly strike prices with exclusivity constraints
- Subscription fees with opportunity costs

**Spot Pricing (Proposition 2)**
- Hotelling equilibrium with full information
- Critical position θ* from convolved distribution

**Multi-Good Monopoly (Proposition 5)**
- Optimal subscription fee with zero strike prices
- Efficient allocation, full surplus extraction

### 2. Welfare Analysis

All welfare metrics computed with numerical precision:

```python
from competitive_screening.src.core.equilibrium import solve_equilibrium_NE
from competitive_screening.src.core.welfare import (
    compute_total_surplus_NE,
    compute_consumer_surplus_NE,
    compute_producer_surplus_NE
)

# Solve equilibrium once (efficient)
eq = solve_equilibrium_NE(v_0, F, G)

# Compute individual metrics
TS = compute_total_surplus_NE(eq)
CS = compute_consumer_surplus_NE(eq)
PS = compute_producer_surplus_NE(eq)

# Verify welfare identity
assert abs(TS - (CS + PS)) < 1e-6
```

### 3. Interactive Frontend

The Streamlit web interface provides:

- **Real-time computation**: Adjust parameters and see results instantly
- **Visual rankings**: Color-coded cards showing welfare ordering
- **Distribution preview**: PDF plots with statistics
- **Formula reference**: Complete mathematical formulas from paper
- **Export functionality**: Download results as JSON or CSV
- **Preset configurations**: Quick access to standard scenarios

**Consistent Color Scheme:**
- 🟡 Yellow: Non-Exclusive (NE)
- 🟢 Green: Spot Pricing (SP)
- 🔴 Red: Exclusive (E)
- 🔵 Blue: Multi-Good Monopoly (MM)

### 4. Distribution Framework

```python
from competitive_screening.src import Uniform, Normal, Logistic

# Built-in distributions (all log-concave)
G = Uniform(a=-1.0, b=1.0)           # Uniform on [a,b]
F = Normal(mu=0.0, sigma=1.0)        # Normal with mean μ, std σ
H = Logistic(mu=0.0, s=1.0)          # Logistic with location μ, scale s

# Distribution properties
print(f"Support: {G.support()}")      # (-1.0, 1.0)
print(f"Mean: {G.mean()}")            # 0.0
print(f"Log-concave: {G.is_log_concave()}")  # True

# PDF, CDF, quantile functions
g = G.pdf(0.0)                        # Density at 0
p = G.cdf(0.5)                        # Cumulative probability
q = G.quantile(0.95)                  # 95th percentile
```

---

## 📁 Project Structure

```
competitive_screening/
├── src/
│   ├── core/
│   │   ├── distributions.py      # Distribution classes (Uniform, Normal, Logistic)
│   │   ├── equilibrium.py        # Equilibrium solvers (NE, SP, E, MM)
│   │   └── welfare.py            # Welfare computations (TS, CS, PS)
│   └── analysis/
│       ├── parameter_sweep.py    # Multi-parameter analysis tools
│       └── visualization.py      # Plotting utilities
├── frontend/
│   └── app.py                    # Streamlit web interface
├── presets/
│   ├── baseline.json             # Standard configuration (v_0=6)
│   ├── high_valuation.json       # High consumer value (v_0=10)
│   └── *.json                    # Additional preset scenarios
├── tests/
│   ├── test_distributions.py    # Distribution unit tests
│   ├── test_equilibrium.py      # Equilibrium solver tests
│   └── test_welfare.py           # Welfare computation tests
└── requirements.txt              # Python dependencies
```

---

## 🔬 Mathematical Foundation

### Model Setup

- **Two firms** (A and B) offering **two products**
- **Sequential information revelation**:
  1. Period 1: Consumer observes type γ, firms offer contracts
  2. Period 2: Consumer learns taste shock ε, realizes θ = γ + ε
- **Consumer valuations**: v_A(θ) = v_0 - θ, v_B(θ) = v_0 + θ
- **Contracts**: Subscription fee s + strike price p (option contract)

### Theoretical Results Implemented

**Theorem 1 (Non-Exclusive Equilibrium)**
- Strike prices: Double monopoly prices due to competition
- Subscription schedules: Envelope theorem with boundary conditions
- Reference: Pages 16-18

**Proposition 2 (Spot Pricing)**
- Hotelling equilibrium with convolved distribution H = G * F
- Efficient allocation based on realized preferences
- Reference: Page 10

**Proposition 3-4 (Exclusive Equilibrium)**
- Critical type γ̂ splits market between firms
- Monopoly pricing with lock-in inefficiency
- Reference: Pages 10-11, 23-25

**Proposition 5 (Multi-Good Monopoly)**
- Optimal: Zero strike prices, positive subscription fee
- Full surplus extraction, efficient allocation
- Reference: Page 15

### Coverage Conditions

For accurate welfare calculations, parameters must satisfy:
- **NE**: v_0 ≥ max_γ(1/g(γ))
- **SP**: Market coverage via convolution distribution
- **E**: Individual firm coverage conditions

All presets are configured to satisfy coverage requirements.

---

## 📈 Example: Welfare Comparison

```python
from competitive_screening.src import Uniform, Normal, compute_all_welfare

# Baseline configuration from paper
v_0 = 6.0
G = Uniform(-1.0, 1.0)
F = Normal(0.0, 1.0)

results = compute_all_welfare(v_0, F, G)

# Total Surplus ranking
print("Total Surplus:")
print(f"  SP: {results['TS_SP']:.4f}  (Spot Pricing)")
print(f"  MM: {results['TS_MM']:.4f}  (Monopoly)")
print(f"  NE: {results['TS_NE']:.4f}  (Non-Exclusive)")
print(f"  E:  {results['TS_E']:.4f}   (Exclusive)")

# Consumer Surplus ranking
print("\nConsumer Surplus:")
print(f"  NE: {results['CS_NE']:.4f}  (Non-Exclusive)")
print(f"  E:  {results['CS_E']:.4f}   (Exclusive)")
print(f"  SP: {results['CS_SP']:.4f}  (Spot Pricing)")
print(f"  MM: {results['CS_MM']:.4f}  (Monopoly)")

# Verify welfare identities
for setting in ['NE', 'SP', 'E', 'MM']:
    ts = results[f'TS_{setting}']
    cs = results[f'CS_{setting}']
    ps = results[f'PS_{setting}']
    error = abs(ts - (cs + ps))
    print(f"\n{setting}: CS + PS = TS? (error = {error:.2e})")
```

**Expected Output:**
```
Total Surplus:
  SP: 6.9247  (Spot Pricing)
  MM: 6.9247  (Monopoly)
  NE: 6.7772  (Non-Exclusive)
  E:  6.5000  (Exclusive)

Consumer Surplus:
  NE: 4.5545  (Non-Exclusive)
  E:  4.5000  (Exclusive)
  SP: 3.9951  (Spot Pricing)
  MM: 0.0000  (Monopoly)

NE: CS + PS = TS? (error = 7.34e-07)
SP: CS + PS = TS? (error = 2.66e-15)
E:  CS + PS = TS? (error = 5.14e-10)
MM: CS + PS = TS? (error = 0.00e+00)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest competitive_screening/tests/

# Run with coverage report
pytest --cov=competitive_screening/src competitive_screening/tests/

# Test specific module
pytest competitive_screening/tests/test_welfare.py

# Run tests in parallel (faster)
pytest -n auto competitive_screening/tests/
```

**Test Coverage:**
- ✅ Distribution properties and operations
- ✅ Equilibrium uniqueness and existence
- ✅ Welfare identity verification (CS + PS = TS)
- ✅ Subscription schedule accuracy
- ✅ Edge cases and parameter validation

---

## ⚙️ Advanced Usage

### Parameter Sweeps

```python
from competitive_screening.src.analysis.parameter_sweep import (
    sweep_v0, sweep_information_precision
)

# Sweep over average valuation
results = sweep_v0(
    v0_values=[4.0, 5.0, 6.0, 7.0, 8.0],
    G=Uniform(-1.0, 1.0),
    F=Normal(0.0, 1.0)
)

# Sweep over information precision (F variance)
results = sweep_information_precision(
    sigma_values=[0.5, 1.0, 1.5, 2.0],
    v_0=6.0,
    G=Uniform(-1.0, 1.0)
)
```

### Custom Distributions

```python
from competitive_screening.src.core.distributions import Distribution
import scipy.stats as stats

class CustomDistribution(Distribution):
    def __init__(self, params):
        self._dist = stats.your_distribution(**params)

    def pdf(self, x):
        return self._dist.pdf(x)

    def cdf(self, x):
        return self._dist.cdf(x)

    def support(self):
        return (lower, upper)

    def mean(self):
        return self._dist.mean()

    def is_log_concave(self):
        # Verify mathematically for your distribution
        return True
```

### Efficient Computation Pattern

For large parameter sweeps, solve equilibria once and reuse:

```python
from competitive_screening.src.core.equilibrium import (
    solve_equilibrium_NE, solve_equilibrium_SP,
    solve_equilibrium_E, solve_equilibrium_MM
)

# Solve all equilibria once
eq_NE = solve_equilibrium_NE(v_0, F, G)
eq_SP = solve_equilibrium_SP(v_0, F, G)
eq_E = solve_equilibrium_E(v_0, F, G)
eq_MM = solve_equilibrium_MM(v_0, F, G)

# Compute multiple welfare metrics (fast)
from competitive_screening.src.core.welfare import *

ts_ne = compute_total_surplus_NE(eq_NE)
cs_ne = compute_consumer_surplus_NE(eq_NE)
ps_ne = compute_producer_surplus_NE(eq_NE)
# ... etc
```

---

## 🎨 Visualization

The framework includes publication-quality plotting utilities:

```python
from competitive_screening.src.analysis.visualization import (
    plot_ts_comparison, plot_welfare_decomposition
)

# Compare total surplus across settings
fig = plot_ts_comparison(
    sweep_results,
    title="Total Surplus vs Average Valuation",
    xlabel="v₀"
)
fig.savefig("ts_comparison.pdf")

# Decompose welfare into CS and PS
fig = plot_welfare_decomposition(
    results,
    setting='NE',
    title="Non-Exclusive Welfare Decomposition"
)
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{ball2024competitive,
  title={Competitive Sequential Screening},
  author={Ball, Ian and Kattwinkel, Deniz and Knoepfle, Daniel},
  journal={arXiv preprint arXiv:2409.02878},
  year={2024}
}
```

**Paper:** https://arxiv.org/abs/2409.02878

---

## 🛠️ Technical Details

### Numerical Methods

- **Equilibrium solving**: Brent's method (scipy.optimize.brentq)
- **Integration**: Adaptive Gauss-Kronrod quadrature (scipy.integrate.quad)
- **Convolution**: Cached numerical integration with LRU cache
- **Tolerances**: Root-finding (1e-8), Integration (1e-8 absolute/relative)

### Performance

- **Single equilibrium**: ~0.1-1 second
- **Welfare computation**: ~0.01-0.1 seconds given equilibrium
- **Parameter sweep (100 points)**: ~1-10 minutes
- **Frontend computation**: Real-time for individual parameter sets

### Dependencies

Core requirements:
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing and optimization
- `streamlit>=1.20.0` - Interactive web interface
- `matplotlib>=3.4.0` - Visualization

See `requirements.txt` for complete list.

---

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- Additional distribution families (e.g., Beta, Gamma)
- Asymmetric market settings
- Multi-dimensional type spaces
- Computational optimizations
- Additional validation tests

Please open an issue to discuss before submitting pull requests.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This implementation is based on the theoretical work by Ian Ball, Deniz Kattwinkel, and Daniel Knoepfle. The computational framework was developed to enable empirical analysis and validation of their theoretical results.

Special thanks to the open-source community for the excellent scientific computing tools (NumPy, SciPy, Streamlit) that made this implementation possible.

---

## 📧 Contact

For questions, issues, or collaboration opportunities, please:
- Open an issue on GitHub
- Email: [your email]
- Twitter: [@yourhandle]

---

**Status:** ✅ Production Ready | **Version:** 1.0.0 | **Last Updated:** February 2026
