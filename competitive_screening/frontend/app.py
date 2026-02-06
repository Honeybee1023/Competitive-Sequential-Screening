"""
Interactive Frontend for Competitive Sequential Screening Framework

A Streamlit-based web interface for exploring welfare comparisons across
market settings: Non-Exclusive (NE), Spot Pricing (SP), Exclusive (E),
and Multi-Good Monopoly (MM).

Launch: streamlit run app.py
"""

import streamlit as st
import sys
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    Uniform, Normal, Logistic,
    compute_all_welfare,
    solve_equilibrium_SP, solve_equilibrium_MM,
    solve_equilibrium_NE, solve_equilibrium_E
)
from src.analysis.visualization import (
    COLORS, MARKERS
)

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Competitive Screening Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    /* Setting-specific card colors (consistent across all rankings) */
    .sp-card {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .mm-card {
        background-color: #d1ecf1;
        border-color: #0173B2;
    }
    .ne-card {
        background-color: #fff3cd;
        border-color: #FFC107;
    }
    .e-card {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .formula-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PRESET MANAGEMENT
# ============================================================================

def load_builtin_presets():
    """Load all built-in preset configurations."""
    presets_dir = Path(__file__).parent.parent / "presets"
    presets = {}

    if presets_dir.exists():
        for preset_file in presets_dir.glob("*.json"):
            with open(preset_file, 'r') as f:
                preset_data = json.load(f)
                presets[preset_data['name']] = preset_data

    return presets

def save_user_preset(name, v_0, G_config, F_config):
    """Save user-defined preset to session state."""
    if 'user_presets' not in st.session_state:
        st.session_state.user_presets = {}

    preset = {
        'name': name,
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'v_0': v_0,
            'G': G_config,
            'F': F_config
        }
    }

    st.session_state.user_presets[name] = preset

def load_preset_to_session(preset):
    """Load preset parameters into session state."""
    params = preset['parameters']

    # Load v_0
    st.session_state.v_0 = params['v_0']

    # Load G parameters
    st.session_state.G_type = params['G']['type']
    if params['G']['type'] == 'Uniform':
        st.session_state.G_uniform_a = params['G']['params']['a']
        st.session_state.G_uniform_b = params['G']['params']['b']
    elif params['G']['type'] == 'Normal':
        st.session_state.G_normal_mu = params['G']['params']['mu']
        st.session_state.G_normal_sigma = params['G']['params']['sigma']

    # Load F parameters
    st.session_state.F_type = params['F']['type']
    if params['F']['type'] == 'Uniform':
        st.session_state.F_uniform_a = params['F']['params']['a']
        st.session_state.F_uniform_b = params['F']['params']['b']
    elif params['F']['type'] == 'Normal':
        st.session_state.F_normal_mu = params['F']['params']['mu']
        st.session_state.F_normal_sigma = params['F']['params']['sigma']

# ============================================================================
# DISTRIBUTION BUILDING
# ============================================================================

def create_distribution(dist_type, params):
    """Create distribution object from type and parameters."""
    if dist_type == "Uniform":
        return Uniform(params['a'], params['b'])
    elif dist_type == "Normal":
        return Normal(params['mu'], params['sigma'])
    elif dist_type == "Logistic":
        return Logistic(params['mu'], params['scale'])
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

def plot_distribution(dist, title="Distribution"):
    """Plot PDF of distribution."""
    fig, ax = plt.subplots(figsize=(6, 3))

    # Get variance safely (some distributions don't have variance() method)
    def get_variance(d):
        """Get variance, handling distributions without variance() method."""
        try:
            return d.variance()
        except AttributeError:
            # Fall back to scipy distribution's var() method
            if hasattr(d, '_dist'):
                return d._dist.var()
            # Manual calculation for Uniform: σ² = (b-a)²/12
            a, b = d.support()
            if not (np.isinf(a) or np.isinf(b)):
                return ((b - a) ** 2) / 12
            return None

    a, b = dist.support()
    # Handle infinite support
    variance = get_variance(dist)
    if np.isinf(a) and variance is not None:
        a = dist.mean() - 4 * np.sqrt(variance)
    if np.isinf(b) and variance is not None:
        b = dist.mean() + 4 * np.sqrt(variance)

    x = np.linspace(a, b, 200)
    y = [dist.pdf(xi) for xi in x]

    ax.plot(x, y, linewidth=2, color='#1f77b4')
    ax.fill_between(x, y, alpha=0.3, color='#1f77b4')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics
    if variance is not None:
        stats_text = f"μ = {dist.mean():.3f}\nσ² = {variance:.3f}"
    else:
        # Fallback if variance can't be computed
        a_supp, b_supp = dist.support()
        if not (np.isinf(a_supp) or np.isinf(b_supp)):
            stats_text = f"μ = {dist.mean():.3f}\nRange: [{a_supp:.2f}, {b_supp:.2f}]"
        else:
            stats_text = f"μ = {dist.mean():.3f}"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.5), fontsize=9)

    plt.tight_layout()
    return fig

# ============================================================================
# WELFARE COMPUTATION
# ============================================================================

@st.cache_data
def compute_welfare_cached(v_0, G_config, F_config):
    """Compute welfare with caching for performance."""
    # Create distributions
    G = create_distribution(G_config['type'], G_config['params'])
    F = create_distribution(F_config['type'], F_config['params'])

    # Suppress warnings for clean UI
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = compute_all_welfare(v_0, F, G)

    return results

def get_ranking(results, metric='TS'):
    """Get ranked list of settings for given metric."""
    if metric == 'TS':
        scores = {
            'NE': results['TS_NE'],
            'SP': results['TS_SP'],
            'E': results['TS_E'],
            'MM': results['TS_MM']
        }
    elif metric == 'CS':
        scores = {
            'NE': results.get('CS_NE'),
            'SP': results['CS_SP'],
            'E': results.get('CS_E'),
            'MM': results['CS_MM']
        }
    elif metric == 'PS':
        scores = {
            'NE': results.get('PS_NE'),
            'SP': results['PS_SP'],
            'E': results.get('PS_E'),
            'MM': results['PS_MM']
        }

    # Sort by score (descending), handling None values
    ranked = sorted(
        [(k, v) for k, v in scores.items() if v is not None],
        key=lambda x: x[1],
        reverse=True
    )

    return ranked, scores

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar():
    """Render the sidebar with input controls."""
    st.sidebar.markdown("## 📊 Configuration")

    # ===== PRESET MANAGEMENT =====
    st.sidebar.markdown("### 🎯 Presets")

    # Load built-in presets
    builtin_presets = load_builtin_presets()

    # Preset selector
    preset_names = ["Custom"] + list(builtin_presets.keys())

    if 'selected_preset' not in st.session_state:
        st.session_state.selected_preset = "Baseline (Paper Example)"

    selected = st.sidebar.selectbox(
        "Load Preset",
        preset_names,
        index=preset_names.index(st.session_state.selected_preset)
              if st.session_state.selected_preset in preset_names else 0,
        key='preset_selector'
    )

    # Load preset if selected
    if selected != "Custom" and selected in builtin_presets:
        if st.sidebar.button("📥 Load Preset", use_container_width=True):
            load_preset_to_session(builtin_presets[selected])
            st.session_state.selected_preset = selected
            st.rerun()

    st.sidebar.markdown("---")

    # ===== PARAMETERS =====
    st.sidebar.markdown("### ⚙️ Parameters")

    # v_0 (Average Valuation)
    v_0 = st.sidebar.slider(
        "Average Valuation (v₀)",
        min_value=1.0,
        max_value=20.0,
        value=st.session_state.get('v_0', 6.0),
        step=0.1,
        help="Average product valuation"
    )
    st.session_state.v_0 = v_0

    st.sidebar.markdown("---")

    # ===== G DISTRIBUTION (Type) =====
    st.sidebar.markdown("### 📈 Type Distribution (G)")
    st.sidebar.caption("Ex-ante consumer heterogeneity")

    G_type = st.sidebar.selectbox(
        "Distribution Type",
        ["Uniform", "Normal", "Logistic"],
        index=["Uniform", "Normal", "Logistic"].index(
            st.session_state.get('G_type', 'Uniform')
        ),
        key='G_type_select'
    )
    st.session_state.G_type = G_type

    G_config = {}
    if G_type == "Uniform":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            a = st.number_input(
                "Min (a)",
                value=st.session_state.get('G_uniform_a', -1.0),
                step=0.1,
                key='G_uniform_a_input'
            )
        with col2:
            b = st.number_input(
                "Max (b)",
                value=st.session_state.get('G_uniform_b', 1.0),
                step=0.1,
                key='G_uniform_b_input'
            )

        if a >= b:
            st.sidebar.error("❌ Min must be < Max")
            return None

        G_config = {'type': 'Uniform', 'params': {'a': a, 'b': b}}
        st.session_state.G_uniform_a = a
        st.session_state.G_uniform_b = b

    elif G_type == "Normal":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            mu = st.number_input(
                "Mean (μ)",
                value=st.session_state.get('G_normal_mu', 0.0),
                step=0.1,
                key='G_normal_mu_input'
            )
        with col2:
            sigma = st.number_input(
                "Std Dev (σ)",
                value=st.session_state.get('G_normal_sigma', 1.0),
                step=0.1,
                min_value=0.01,
                key='G_normal_sigma_input'
            )

        G_config = {'type': 'Normal', 'params': {'mu': mu, 'sigma': sigma}}
        st.session_state.G_normal_mu = mu
        st.session_state.G_normal_sigma = sigma

    st.sidebar.markdown("---")

    # ===== F DISTRIBUTION (Taste Shock) =====
    st.sidebar.markdown("### 📉 Taste Shock Distribution (F)")
    st.sidebar.caption("Information precision (lower σ = more informative)")

    F_type = st.sidebar.selectbox(
        "Distribution Type",
        ["Normal", "Uniform", "Logistic"],
        index=["Normal", "Uniform", "Logistic"].index(
            st.session_state.get('F_type', 'Normal')
        ),
        key='F_type_select'
    )
    st.session_state.F_type = F_type

    F_config = {}
    if F_type == "Normal":
        st.sidebar.info("ℹ️ For symmetric F, mean is fixed at μ=0")
        sigma = st.sidebar.slider(
            "Std Dev (σ)",
            min_value=0.1,
            max_value=5.0,
            value=st.session_state.get('F_normal_sigma', 1.0),
            step=0.1,
            help="Lower σ = more precise information"
        )

        F_config = {'type': 'Normal', 'params': {'mu': 0.0, 'sigma': sigma}}
        st.session_state.F_normal_sigma = sigma

    elif F_type == "Uniform":
        st.sidebar.info("ℹ️ Symmetric uniform centered at 0")
        width = st.sidebar.slider(
            "Half-width",
            min_value=0.1,
            max_value=5.0,
            value=st.session_state.get('F_uniform_width', 1.0),
            step=0.1
        )

        F_config = {'type': 'Uniform', 'params': {'a': -width, 'b': width}}
        st.session_state.F_uniform_width = width

    st.sidebar.markdown("---")

    # ===== ACTION BUTTONS =====
    col1, col2 = st.sidebar.columns(2)

    with col1:
        compute_button = st.button(
            "🚀 Compute",
            use_container_width=True,
            type="primary"
        )

    with col2:
        clear_button = st.button(
            "🔄 Reset",
            use_container_width=True
        )

    if clear_button:
        # Reset to baseline
        st.session_state.v_0 = 6.0
        st.session_state.G_type = 'Uniform'
        st.session_state.G_uniform_a = -1.0
        st.session_state.G_uniform_b = 1.0
        st.session_state.F_type = 'Normal'
        st.session_state.F_normal_sigma = 1.0
        st.session_state.selected_preset = "Baseline (Paper Example)"
        st.rerun()

    # Save preset
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Save Configuration")

    preset_name = st.sidebar.text_input("Preset Name", "My Configuration")
    if st.sidebar.button("Save as Preset", use_container_width=True):
        save_user_preset(preset_name, v_0, G_config, F_config)
        st.sidebar.success(f"✓ Saved '{preset_name}'")

    return compute_button, v_0, G_config, F_config

def render_ranking_card(rank, setting, score, metric, is_approximate=False):
    """Render a single ranking card with consistent setting-based colors."""
    # Determine card style based on setting type (consistent colors)
    card_class = {
        'NE': 'ne-card',
        'SP': 'sp-card',
        'E': 'e-card',
        'MM': 'mm-card'
    }.get(setting, 'metric-card')

    # Icons
    rank_icon = {
        1: "🥇",
        2: "🥈",
        3: "🥉",
        4: "4️⃣"
    }.get(rank, "")

    status_icon = "⚠️" if is_approximate else "✓"

    # Full names
    full_names = {
        'NE': 'Non-Exclusive',
        'SP': 'Spot Pricing',
        'E': 'Exclusive',
        'MM': 'Multi-Good Monopoly'
    }

    html = f"""
    <div class="metric-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 1.5rem;">{rank_icon}</span>
                <span style="font-size: 1.1rem; font-weight: 600; margin-left: 0.5rem;">
                    {full_names.get(setting, setting)}
                </span>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.3rem; font-weight: 700; color: #333;">
                    {score:.4f}
                </div>
                <div style="font-size: 0.9rem; color: #666;">
                    {status_icon} {'Approximate' if is_approximate else 'Accurate'}
                </div>
            </div>
        </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)

def render_results(results):
    """Render the 3x4 welfare grid."""
    st.markdown("## 📊 Results: Welfare Rankings")

    col1, col2, col3 = st.columns(3)

    # ===== TOTAL SURPLUS =====
    with col1:
        st.markdown("### 💰 Total Surplus (TS)")
        st.caption("Overall economic efficiency")

        ranked, scores = get_ranking(results, 'TS')

        for rank, (setting, score) in enumerate(ranked, 1):
            render_ranking_card(rank, setting, score, 'TS', is_approximate=False)

        # Summary
        winner = ranked[0][0]
        st.success(f"**Winner:** {winner} achieves maximum total surplus!")

    # ===== CONSUMER SURPLUS =====
    with col2:
        st.markdown("### 👥 Consumer Surplus (CS)")
        st.caption("Consumer welfare")

        ranked, scores = get_ranking(results, 'CS')

        for rank, (setting, score) in enumerate(ranked, 1):
            render_ranking_card(rank, setting, score, 'CS', is_approximate=False)

    # ===== PRODUCER SURPLUS =====
    with col3:
        st.markdown("### 🏢 Producer Surplus (PS)")
        st.caption("Firm profits")

        ranked, scores = get_ranking(results, 'PS')

        for rank, (setting, score) in enumerate(ranked, 1):
            render_ranking_card(rank, setting, score, 'PS', is_approximate=False)

        # Interpretation
        if scores['MM'] == max(s for s in scores.values() if s is not None):
            st.warning("📌 MM achieves full surplus extraction (CS=0)")

def render_visualizations(results, G, F):
    """Render visualization tabs."""
    st.markdown("---")
    st.markdown("## 📈 Visualizations")

    tab1, tab2, tab3 = st.tabs([
        "📊 Comparison Charts",
        "📉 Distribution Previews",
        "📋 Formula Reference"
    ])

    # ===== TAB 1: Comparison Charts =====
    with tab1:
        st.markdown("### Total Surplus Comparison")

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        settings = ['NE', 'SP', 'E', 'MM']
        ts_values = [results['TS_NE'], results['TS_SP'], results['TS_E'], results['TS_MM']]
        colors_list = [COLORS[s] for s in settings]

        bars = ax.bar(settings, ts_values, color=colors_list, alpha=0.7, edgecolor='black')

        # Add value labels on bars
        for bar, value in zip(bars, ts_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Total Surplus', fontsize=12, fontweight='bold')
        ax.set_title('Total Surplus Across Market Settings', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add legend with full names
        legend_labels = [
            'NE: Non-Exclusive',
            'SP: Spot Pricing',
            'E: Exclusive',
            'MM: Multi-Good Monopoly'
        ]
        ax.legend(bars, legend_labels, loc='upper right', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

        # CS and PS charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Consumer Surplus")
            fig_cs, ax_cs = plt.subplots(figsize=(5, 4))
            cs_settings = ['NE', 'SP', 'E', 'MM']
            cs_values = [
                results.get('CS_NE', 0),
                results['CS_SP'],
                results.get('CS_E', 0),
                results['CS_MM']
            ]
            cs_colors = [COLORS[s] for s in cs_settings]

            bars_cs = ax_cs.bar(cs_settings, cs_values, color=cs_colors, alpha=0.7, edgecolor='black')
            for bar, value in zip(bars_cs, cs_values):
                height = bar.get_height()
                ax_cs.text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.3f}',
                          ha='center', va='bottom', fontsize=10)

            ax_cs.set_ylabel('Consumer Surplus', fontsize=10, fontweight='bold')
            ax_cs.set_title('Consumer Surplus Comparison', fontsize=11, fontweight='bold')
            ax_cs.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig_cs)

        with col2:
            st.markdown("### Producer Surplus")
            fig_ps, ax_ps = plt.subplots(figsize=(5, 4))
            ps_settings = ['NE', 'SP', 'E', 'MM']
            ps_values = [
                results.get('PS_NE', 0),
                results['PS_SP'],
                results.get('PS_E', 0),
                results['PS_MM']
            ]
            ps_colors = [COLORS[s] for s in ps_settings]

            bars_ps = ax_ps.bar(ps_settings, ps_values, color=ps_colors, alpha=0.7, edgecolor='black')
            for bar, value in zip(bars_ps, ps_values):
                height = bar.get_height()
                ax_ps.text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.3f}',
                          ha='center', va='bottom', fontsize=10)

            ax_ps.set_ylabel('Producer Surplus', fontsize=10, fontweight='bold')
            ax_ps.set_title('Producer Surplus Comparison', fontsize=11, fontweight='bold')
            ax_ps.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig_ps)

    # ===== TAB 2: Distribution Previews =====
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Type Distribution (G)")
            fig_g = plot_distribution(G, "G: Ex-Ante Heterogeneity")
            st.pyplot(fig_g)

            # Get variance safely
            try:
                g_var = G.variance()
            except AttributeError:
                g_var = G._dist.var() if hasattr(G, '_dist') else None

            variance_str = f"{g_var:.3f}" if g_var is not None else "N/A"

            st.info(f"""
            **Properties:**
            - Support: {G.support()}
            - Mean: {G.mean():.3f}
            - Variance: {variance_str}
            - Symmetric: {G.is_symmetric()}
            """)

        with col2:
            st.markdown("### Taste Shock Distribution (F)")
            fig_f = plot_distribution(F, "F: Information Precision")
            st.pyplot(fig_f)

            # Get variance safely
            try:
                f_var = F.variance()
            except AttributeError:
                f_var = F._dist.var() if hasattr(F, '_dist') else None

            variance_str = f"{f_var:.3f}" if f_var is not None else "N/A"

            st.info(f"""
            **Properties:**
            - Support: {F.support()}
            - Mean: {F.mean():.3f}
            - Variance: {variance_str}
            - Symmetric: {F.is_symmetric()}
            """)

    # ===== TAB 3: Formula Reference =====
    with tab3:
        render_formula_reference()

def render_formula_reference():
    """Render expandable formula reference."""
    st.markdown("### 📚 Formula Reference")
    st.caption("Mathematical formulas from the paper")

    # Spot Pricing
    with st.expander("🟢 Spot Pricing (SP) Equilibrium"):
        st.latex(r"\theta^* = \frac{1 - 2H(\theta^*)}{h(\theta^*)}")
        st.markdown("""
        **Critical position** where consumer is indifferent between A and B.

        **Prices:**
        """)
        st.latex(r"p_A^* = \frac{2H(\theta^*)}{h(\theta^*)}, \quad p_B^* = \frac{2(1-H(\theta^*))}{h(\theta^*)}")
        st.markdown("""
        - **H** = G * F (convolution of type and taste shock)
        - **h** = density of H
        - For symmetric distributions: θ* = 0

        **Reference:** Proposition 2, page 10, equation (4)
        """)

    # Multi-Good Monopoly
    with st.expander("🔵 Multi-Good Monopoly (MM) Equilibrium"):
        st.latex(r"(p_A^*, p_B^*, s^*) = (0, 0, \mathbb{E}_{\theta|0}[\max\{v_A(\theta), v_B(\theta)\}])")
        st.markdown("""
        **Optimal contract**: Zero strike prices, full surplus extraction via subscription fee.

        **Subscription fee:**
        """)
        st.latex(r"s^* = \mathbb{E}[v_0 + |\theta|]")
        st.markdown("""
        - Achieves **efficient allocation** (match to best product)
        - **Consumer surplus = 0** (full extraction)
        - Only defined for **symmetric G**

        **Reference:** Proposition 5, page 15
        """)

    # Non-Exclusive
    with st.expander("🟡 Non-Exclusive (NE) Equilibrium"):
        st.markdown("""
        **Strike prices** (Theorem 1, pages 16-18):
        """)
        st.latex(r"p_A^*(\gamma) = \frac{2G(\gamma)}{g(\gamma)}, \quad p_B^*(\gamma) = \frac{2(1-G(\gamma))}{g(\gamma)}")
        st.markdown("""
        - Factor of 2 compared to monopoly prices (competitive effect)
        - Consumers subscribe to **both firms**
        - Exercise option for product with higher realized value

        **Upper bounds on strike prices:**
        """)
        st.latex(r"\bar{p}_A = \frac{2}{g(\underline{\gamma})}, \quad \bar{p}_B = \frac{2}{g(\bar{\gamma})}")
        st.markdown("""
        where γ̲ = γ_min and γ̄ = γ_max.

        **Subscription schedules** (via envelope theorem):
        """)
        st.latex(r"""
        s_A^*(p_A) = \mathbb{E}_{\theta|\bar{\gamma}}\left[\left(v_A(\theta) - \bar{p}_A - (v_B(\theta))_+\right)_+\right] + \int_{p_A}^{\bar{p}_A} Q_A^*(p') \, dp'
        """)
        st.latex(r"""
        s_B^*(p_B) = \mathbb{E}_{\theta|\underline{\gamma}}\left[\left(v_B(\theta) - \bar{p}_B - (v_A(\theta))_+\right)_+\right] + \int_{p_B}^{\bar{p}_B} Q_B^*(p') \, dp'
        """)
        st.markdown("""
        - First term: Boundary utility at extreme types
        - Second term: Integral of interim demand over strike prices
        - Ensures incentive compatibility across type space

        **Reference:** Theorem 1, pages 16-18
        """)

    # Exclusive
    with st.expander("🔴 Exclusive (E) Equilibrium"):
        st.markdown("""
        **Indifference condition** for critical type γ̂:
        """)
        st.latex(r"""
        \mathbb{E}_{\theta|\hat{\gamma}}[(v_A(\theta) - p_A^M(\hat{\gamma}))_+]
        = \mathbb{E}_{\theta|\hat{\gamma}}[(v_B(\theta) - p_B^M(\hat{\gamma}))_+]
        """)
        st.markdown("""
        **Monopoly strike prices:**
        """)
        st.latex(r"p_A^M(\gamma) = \frac{G(\gamma)}{g(\gamma)}, \quad p_B^M(\gamma) = \frac{1-G(\gamma)}{g(\gamma)}")
        st.markdown("""
        - Consumers **locked in** to one firm
        - Creates allocation inefficiency (can't switch after ε revealed)
        - γ̂ splits market between firms A and B

        **Equilibrium strike prices** at critical type:
        """)
        st.latex(r"\hat{p}_A = p_A^M(\hat{\gamma}), \quad \hat{p}_B = p_B^M(\hat{\gamma})")
        st.markdown("""
        **Subscription schedules** (Proposition 4, pages 23-25):
        """)
        st.latex(r"""
        s_A^*(p_A) = \hat{p}_A \cdot Q_B^M(\hat{p}_B|\hat{\gamma}) + \int_{p_A}^{\hat{p}_A} Q_A^*(p') \, dp'
        """)
        st.latex(r"""
        s_B^*(p_B) = \hat{p}_B \cdot Q_A^M(\hat{p}_A|\hat{\gamma}) + \int_{p_B}^{\hat{p}_B} Q_B^*(p') \, dp'
        """)
        st.markdown("""
        - First term: Boundary term (opportunity cost at γ̂)
        - Second term: Integral of monopoly interim demand
        - Only types γ ≤ γ̂ subscribe to A; types γ > γ̂ subscribe to B

        **Reference:** Proposition 3 (pages 10-11), Proposition 4 (pages 23-25)
        """)

    # Total Surplus
    with st.expander("📊 Total Surplus Calculation"):
        st.markdown("""
        **General formula:**
        """)
        st.latex(r"TS = \mathbb{E}[\text{realized valuation for purchased good}]")
        st.markdown("""
        **For each setting:**

        - **SP**: Efficient allocation based on realized θ
        - **MM**: Efficient allocation (matches consumer to best product)
        - **NE**: Some inefficiency from option exercise decisions
        - **E**: Inefficiency from lock-in (can't switch firms)

        **Theoretical ranking** (symmetric distributions):
        """)
        st.info("SP = MM > NE > E")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""

    # Title
    st.markdown('<div class="main-title">Competitive Sequential Screening Explorer</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Interactive welfare analysis across market settings</div>',
                unsafe_allow_html=True)

    # Initialize session state
    if 'first_run' not in st.session_state:
        st.session_state.first_run = True
        # Load baseline preset by default
        builtin_presets = load_builtin_presets()
        if "Baseline (Paper Example)" in builtin_presets:
            load_preset_to_session(builtin_presets["Baseline (Paper Example)"])

    # Render sidebar and get inputs
    result = render_sidebar()

    if result is None:
        st.error("❌ Invalid parameter configuration. Please check your inputs.")
        return

    compute_button, v_0, G_config, F_config = result

    # Quick start guide for first-time users
    if st.session_state.first_run:
        st.info("""
        👋 **Welcome!** This tool compares welfare across 4 market settings:
        - **NE** (Non-Exclusive): Consumers subscribe to both firms
        - **SP** (Spot Pricing): No pre-contractual commitments
        - **E** (Exclusive): Consumers locked in to one firm
        - **MM** (Multi-Good Monopoly): Single firm controls both products

        👈 **Configure parameters in the sidebar**, then click **🚀 Compute** to see results!
        """)

    # Compute welfare if button clicked
    if compute_button:
        st.session_state.first_run = False

        with st.spinner("⏳ Computing equilibria and welfare metrics..."):
            try:
                # Compute welfare
                results = compute_welfare_cached(v_0, G_config, F_config)

                # Create distributions for visualization
                G = create_distribution(G_config['type'], G_config['params'])
                F = create_distribution(F_config['type'], F_config['params'])

                # Store in session state
                st.session_state.results = results
                st.session_state.G = G
                st.session_state.F = F
                st.session_state.v_0_computed = v_0

                st.success("✅ Computation complete!")

            except Exception as e:
                st.error(f"❌ Error during computation: {str(e)}")
                st.exception(e)
                return

    # Display results if available
    if 'results' in st.session_state:
        render_results(st.session_state.results)
        render_visualizations(
            st.session_state.results,
            st.session_state.G,
            st.session_state.F
        )

        # Export results
        st.markdown("---")
        st.markdown("## 💾 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            # JSON export
            export_data = {
                'parameters': {
                    'v_0': st.session_state.v_0_computed,
                    'G': G_config,
                    'F': F_config
                },
                'results': {
                    'TS_NE': float(st.session_state.results['TS_NE']),
                    'TS_SP': float(st.session_state.results['TS_SP']),
                    'TS_E': float(st.session_state.results['TS_E']),
                    'TS_MM': float(st.session_state.results['TS_MM']),
                    'CS_SP': float(st.session_state.results['CS_SP']),
                    'CS_MM': float(st.session_state.results['CS_MM']),
                    'PS_SP': float(st.session_state.results['PS_SP']),
                    'PS_MM': float(st.session_state.results['PS_MM'])
                },
                'timestamp': datetime.now().isoformat()
            }

            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"welfare_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        with col2:
            # CSV export
            csv_data = "Setting,Total_Surplus,Consumer_Surplus,Producer_Surplus\n"
            csv_data += f"NE,{st.session_state.results['TS_NE']:.6f},N/A,N/A\n"
            csv_data += f"SP,{st.session_state.results['TS_SP']:.6f},{st.session_state.results['CS_SP']:.6f},{st.session_state.results['PS_SP']:.6f}\n"
            csv_data += f"E,{st.session_state.results['TS_E']:.6f},N/A,N/A\n"
            csv_data += f"MM,{st.session_state.results['TS_MM']:.6f},{st.session_state.results['CS_MM']:.6f},{st.session_state.results['PS_MM']:.6f}\n"

            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"welfare_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Built with ❤️ using Streamlit | Framework by Ball, Kattwinkel, & Knoepfle (2025)</p>
        <p>📚 <a href="https://github.com/yourusername/competitive-screening" target="_blank">Documentation</a> |
        🐛 <a href="https://github.com/yourusername/competitive-screening/issues" target="_blank">Report Issue</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
