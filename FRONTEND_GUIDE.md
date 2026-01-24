# Competitive Screening Explorer - Frontend Guide

## Quick Start

### Launch the App (30 seconds)

**Option 1: Shell script (Mac/Linux)**
```bash
cd /Users/honjar/Downloads/Competitive_Sequential_Screening
./run_app.sh
```

**Option 2: Python script (All platforms)**
```bash
cd /Users/honjar/Downloads/Competitive_Sequential_Screening
python3 run_frontend.py
```

**Option 3: Direct Streamlit (if you know what you're doing)**
```bash
cd competitive_screening/frontend
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## User Interface Overview

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Competitive Sequential Screening Explorer                  │
│  Interactive welfare analysis across market settings        │
├──────────────────┬──────────────────────────────────────────┤
│                  │                                          │
│   SIDEBAR        │          MAIN CONTENT                    │
│   (Controls)     │          (Results & Visualizations)      │
│                  │                                          │
│  • Presets       │  • Welfare Rankings (3×4 Grid)          │
│  • Parameters    │  • Comparison Charts                     │
│  • Distributions │  • Distribution Previews                │
│  • Action Buttons│  • Formula Reference                     │
│  • Save Config   │  • Export Options                        │
│                  │                                          │
└──────────────────┴──────────────────────────────────────────┘
```

---

## Step-by-Step Guide

### 1. First-Time User Flow

**When you first launch the app:**

1. You'll see a welcome message with an overview
2. The "Baseline (Paper Example)" preset is pre-loaded
3. Click the **🚀 Compute** button in the sidebar
4. Results appear in ~2-3 seconds
5. Explore the different tabs and visualizations

**That's it!** You've run your first welfare analysis.

---

### 2. Changing Parameters

#### **Average Valuation (v₀)**

Located at the top of the sidebar:
- **Slider**: Range 1.0 to 20.0, step 0.1
- **Default**: 6.0
- **Interpretation**: Higher v₀ means consumers value products more
- **Effect**: Increases TS for all settings proportionally

#### **Type Distribution (G)**

Controls ex-ante heterogeneity:

**Uniform Distribution:**
- **Parameters**: Min (a) and Max (b)
- **Example**: Uniform(-1, 1) means types range from -1 to 1
- **Narrower range** (e.g., -0.5 to 0.5): Less heterogeneity
- **Wider range** (e.g., -2 to 2): More heterogeneity

**Normal Distribution:**
- **Parameters**: Mean (μ) and Standard Deviation (σ)
- **Symmetric**: For theoretical results, use μ=0
- **Effect of σ**: Larger σ = more dispersed types

**Logistic Distribution:**
- Similar to Normal but with heavier tails

#### **Taste Shock Distribution (F)**

Controls information precision:

**Normal Distribution (Default):**
- **Mean fixed at 0** (for symmetry)
- **Adjust σ only**: Lower σ = more precise information
- **σ = 0.5**: High precision (consumers learn a lot)
- **σ = 2.0**: Low precision (consumers learn little)

**Uniform Distribution:**
- **Symmetric around 0**: Adjust half-width
- **Smaller width**: More precise information

---

### 3. Using Presets

#### **Built-in Presets**

Located at the top of the sidebar:

1. **Baseline (Paper Example)**
   - Standard configuration: G ~ Uniform(-1,1), F ~ Normal(0,1), v₀=6
   - Use this as your starting point

2. **Narrow Types**
   - G ~ Uniform(-0.5, 0.5)
   - See how reduced heterogeneity affects rankings

3. **High Information Precision**
   - F ~ Normal(0, 0.5)
   - More informative signals

4. **Low Information Precision**
   - F ~ Normal(0, 2)
   - Less informative signals

5. **High Valuation**
   - v₀ = 10
   - See effect of higher baseline value

6. **Wide Types**
   - G ~ Uniform(-2, 2)
   - More ex-ante heterogeneity

**To load a preset:**
1. Select from dropdown
2. Click **📥 Load Preset** button
3. Parameters update automatically
4. Click **🚀 Compute** to see results

#### **Saving Your Own Presets**

At the bottom of the sidebar:

1. Configure your desired parameters
2. Enter a name in "Preset Name" field (e.g., "My Research Configuration")
3. Click **Save as Preset**
4. Success message confirms save
5. Preset is saved to your browser session

**Note:** Presets are session-based. To persist:
- Use **Export Results** to download JSON
- Keep the JSON file for later import

---

### 4. Interpreting Results

#### **Welfare Rankings Grid (Main Display)**

Three columns showing rankings for each metric:

**Column 1: Total Surplus (TS)** 💰
- Overall economic efficiency
- Sum of consumer and producer surplus
- **Winner** maximizes social welfare
- All values are **accurate** ✓

**Column 2: Consumer Surplus (CS)** 👥
- Consumer welfare
- Only SP and MM are accurate
- NE and E are approximate ⚠️

**Column 3: Producer Surplus (PS)** 🏢
- Firm profits
- Only SP and MM are accurate
- NE and E are approximate ⚠️

**Reading a card:**
```
┌──────────────────────────────────┐
│ 🥇 Spot Pricing            6.925 │
│                         ✓ Accurate│
└──────────────────────────────────┘
```
- **Icon**: 🥇 (1st), 🥈 (2nd), 🥉 (3rd), 4️⃣ (4th)
- **Value**: Exact welfare measure
- **Status**: ✓ Accurate or ⚠️ Approximate

**Color coding:**
- **Green** (🥇 winner): Best performance
- **Blue** (🥈 second): Second best
- **Yellow** (🥉 third): Third place
- **Red** (4️⃣ fourth): Last place

#### **Theoretical Predictions**

For **symmetric distributions** (F and G centered at 0):
- **Expected TS ranking**: SP = MM > NE > E
- **Why?**
  - SP and MM: Efficient allocation
  - NE: Some inefficiency (option exercise)
  - E: Most inefficiency (lock-in effect)

If your results differ, check:
- Are distributions symmetric?
- Are parameters in reasonable ranges?
- Any error messages?

---

### 5. Visualizations

Click the tabs below the results grid:

#### **Tab 1: 📊 Comparison Charts**

**Total Surplus Bar Chart:**
- All 4 settings side-by-side
- Color-coded by setting
- Values labeled on bars
- Immediately see which setting dominates

**Consumer Surplus Chart:**
- Only SP and MM (accurate values)
- Shows who benefits more: consumers or firms

**Producer Surplus Chart:**
- Only SP and MM (accurate values)
- MM typically has highest PS (extracts all surplus)

#### **Tab 2: 📉 Distribution Previews**

**Type Distribution (G):**
- PDF plot with shaded area
- Shows support, mean, variance
- Visualize ex-ante heterogeneity

**Taste Shock Distribution (F):**
- PDF plot with shaded area
- Shows information precision (width)
- Narrower = more precise signals

**Use this to:**
- Verify your parameter choices
- Understand the model setup
- Compare different distributions visually

#### **Tab 3: 📋 Formula Reference**

Expandable sections for each market setting:

**What's included:**
- Mathematical formulas (LaTeX-rendered)
- Plain English descriptions
- Paper references (proposition numbers)
- Interpretation of results

**How to use:**
- Click any section to expand
- See exact formulas used for computation
- Understand theoretical foundations
- Reference paper sections for details

---

### 6. Exporting Results

At the bottom of the main content area:

#### **Download JSON**
- Complete parameter configuration
- All welfare values
- Timestamp
- Use for: Reproducibility, sharing, archiving

#### **Download CSV**
- Tabular format: Setting, TS, CS, PS
- Use for: Spreadsheet analysis, plotting in other tools

**Filename format:**
```
welfare_results_20260123_143022.json
welfare_results_20260123_143022.csv
```
(Includes timestamp for easy organization)

---

## Common Workflows

### Workflow 1: Quick Exploration

**Goal:** See basic rankings for standard parameters

1. Launch app (loads Baseline preset automatically)
2. Click **🚀 Compute**
3. Note winner in each category
4. Done!

**Time:** 30 seconds

---

### Workflow 2: Parameter Sensitivity

**Goal:** See how changing one parameter affects rankings

1. Load Baseline preset
2. Compute baseline results
3. Change v₀ slider to 10
4. Click **🚀 Compute** again
5. Compare results
6. Try different σ values (0.5, 1, 2, 3)
7. Observe pattern

**Insight:** TS scales linearly with v₀; rankings stay consistent

**Time:** 2-3 minutes

---

### Workflow 3: Distribution Comparison

**Goal:** Compare Narrow vs. Wide types

1. Load "Narrow Types" preset
2. Compute and note results
3. Load "Wide Types" preset
4. Compute and note results
5. Compare TS values

**Insight:** Wider types increase TS for all settings (more opportunities for matching)

**Time:** 1-2 minutes

---

### Workflow 4: Create Custom Configuration

**Goal:** Set up and save your own scenario

1. Start with any preset (or clear to defaults)
2. Adjust parameters:
   - Set v₀ = 8
   - G ~ Uniform(-1.5, 1.5)
   - F ~ Normal(0, 1.2)
3. Click **🚀 Compute**
4. Verify results look reasonable
5. Enter name: "My Research Config"
6. Click **Save as Preset**
7. Export JSON for permanent storage

**Time:** 2-3 minutes

---

### Workflow 5: Research Paper Analysis

**Goal:** Generate figures for publication

1. Load "Baseline" preset
2. Compute results
3. Navigate to "Comparison Charts" tab
4. Right-click chart → Save Image
5. Change to "High Precision" preset
6. Compute and save chart
7. Compare side-by-side in your paper

**Note:** For publication-quality plots, use the Python API (see `examples/parameter_sweep_example.py`)

---

## Understanding the Metrics

### Total Surplus (TS)

**Definition:** Expected total value created in the market

**Formula:** E[realized valuation of purchased good]

**Interpretation:**
- Higher is better (more value created)
- Efficient settings: SP, MM
- TS = CS + PS (welfare decomposition)

**Accuracy:** ✓ Accurate for all settings

---

### Consumer Surplus (CS)

**Definition:** Consumer utility minus payments

**Formula:** E[value received - (strike price + subscription fee)]

**Interpretation:**
- Higher is better for consumers
- MM typically lowest (full extraction)
- SP typically highest (competitive pricing)

**Accuracy:**
- ✓ Accurate for SP and MM
- ⚠️ Approximate for NE and E (subscription schedules incomplete)

---

### Producer Surplus (PS)

**Definition:** Firm profits (revenues minus costs, assuming zero marginal cost)

**Formula:** E[strike price + subscription fee - cost]

**Interpretation:**
- Higher is better for firms
- MM typically highest (monopoly power)
- Equals firm revenue (zero marginal cost assumption)

**Accuracy:**
- ✓ Accurate for SP and MM
- ⚠️ Approximate for NE and E

---

## Market Settings Explained

### Non-Exclusive (NE) 🔵

**Structure:**
- Consumers subscribe to **both firms**
- Pay subscription fees to both
- See taste shock ε, then choose product
- Exercise option with lower strike price

**Characteristics:**
- Competitive information revelation
- Some allocation inefficiency
- Intermediate efficiency

**When it wins:**
- Rarely (SP and MM typically dominate)

---

### Spot Pricing (SP) 🟠

**Structure:**
- **No pre-contractual commitments**
- Consumers learn full information (θ = γ + ε)
- Choose product with highest net value
- Pay spot prices

**Characteristics:**
- Efficient allocation (for symmetric distributions)
- Competitive pricing
- High consumer surplus

**When it wins:**
- Almost always ties with MM for TS
- Often highest CS

---

### Exclusive (E) 🟢

**Structure:**
- Consumers subscribe to **one firm** only
- **Locked in** (can't switch after ε revealed)
- Pay subscription fee + strike price

**Characteristics:**
- Allocation inefficiency (lock-in)
- Market segmentation (firms split consumers)
- Lowest TS

**When it wins:**
- Rarely (typically least efficient)

---

### Multi-Good Monopoly (MM) 🟣

**Structure:**
- **Single firm** controls both products
- Sets p_A = p_B = 0 (zero strike prices)
- Extracts all surplus via subscription fee

**Characteristics:**
- Efficient allocation (monopolist matches optimally)
- Full surplus extraction (CS = 0)
- Benchmark for efficiency

**When it wins:**
- Always ties with SP for TS
- Always highest PS
- Always lowest CS

---

## Troubleshooting

### App won't launch

**Issue:** `ModuleNotFoundError` or import errors

**Solution:**
```bash
pip3 install streamlit numpy scipy matplotlib
```

---

### "Invalid parameter configuration"

**Issue:** Min ≥ Max for Uniform distribution

**Solution:** Ensure Min < Max. Click **🔄 Reset** to restore defaults.

---

### Results seem wrong

**Issue:** Unexpected ranking

**Check:**
1. Are distributions symmetric? (For theoretical predictions)
2. Are parameters reasonable? (v₀ > 0, σ > 0)
3. Are you comparing the right metric? (TS vs CS vs PS)

**Solution:** Try loading Baseline preset and verify that gives expected results (SP = MM > NE > E).

---

### Computation is slow

**Issue:** Takes >10 seconds to compute

**Causes:**
- Very wide distributions (support too large)
- Very small σ (< 0.1)
- Very large σ (> 5)

**Solution:**
- Use reasonable parameter ranges
- Streamlit caches results automatically (second computation is instant)

---

### Formulas not rendering

**Issue:** LaTeX shows as raw text

**Solution:** Streamlit should render LaTeX automatically. If not:
1. Check internet connection (MathJax CDN required)
2. Try refreshing browser (Ctrl+R or Cmd+R)
3. Clear browser cache

---

### Can't save presets

**Issue:** "Save as Preset" doesn't work

**Solution:**
- Presets are session-based (lost on refresh)
- Use **Export Results** to download JSON permanently
- Store JSON files for later reference

---

### Browser won't open automatically

**Issue:** App launches but browser doesn't open

**Solution:**
1. Manually navigate to `http://localhost:8501`
2. If port is in use, Streamlit will try 8502, 8503, etc.
3. Check terminal output for exact URL

---

## Advanced Tips

### Tip 1: Keyboard Shortcuts

- **R**: Rerun app (after changing code)
- **C**: Clear cache
- **Ctrl+C**: Stop server (in terminal)

---

### Tip 2: Multiple Configurations

Want to compare 2 configurations side-by-side?

1. Compute first configuration
2. Export JSON
3. Change parameters
4. Compute second configuration
5. Export second JSON
6. Open both JSON files to compare numerically

*(Future enhancement: built-in comparison mode)*

---

### Tip 3: Custom Distributions

To add new distributions (e.g., Beta, Exponential):

1. Implement in `competitive_screening/src/core/distributions.py`
2. Add to dropdown options in `app.py` (lines ~180 and ~240)
3. Add parameter inputs (follow existing pattern)

See developer comments in `app.py` for details.

---

### Tip 4: Parameter Sweeps

For **multiple values** of a parameter (e.g., v₀ = [2,4,6,8,10]):

Use the **Python API** instead:
```python
from src.analysis import sweep_v0
result = sweep_v0([2,4,6,8,10], F, G)
```

See `examples/parameter_sweep_example.py` for full workflow.

The frontend is for **single-point exploration**, not batch analysis.

---

### Tip 5: Publication Figures

For **publication-quality plots** (300 DPI, customizable):

Use the visualization API:
```python
from src.analysis.visualization import plot_ts_comparison
fig = plot_ts_comparison(result, save_path='figure.png')
```

Frontend charts are for **quick visual inspection**.

---

## FAQ

**Q: Can I run parameter sweeps in the frontend?**
A: Not directly. Use the Python API (`sweep_v0`, etc.) for batch analysis. The frontend is for single-configuration exploration.

**Q: Why are CS and PS approximate for NE and E?**
A: Subscription schedule formulas require boundary conditions not in the paper excerpt. TS is fully accurate.

**Q: Can I compare multiple configurations at once?**
A: Not in current version (P2 feature). Use Export → compare JSON files manually.

**Q: What's the difference between G and F?**
A: G = Type distribution (ex-ante, permanent), F = Taste shock (information, realized later). See paper for model details.

**Q: Can I use this for teaching?**
A: Yes! Perfect for demonstrating welfare concepts interactively.

**Q: How do I cite this tool?**
A: Cite the original paper: Ball, Kattwinkel, and Knoepfle (2025) "Competitive Sequential Screening"

**Q: Can I modify the code?**
A: Yes! MIT License (check LICENSE file). Contributions welcome.

**Q: Does this work on Windows?**
A: Yes, use `python3 run_frontend.py` or `python run_frontend.py`

**Q: Can I deploy this online?**
A: Yes! Deploy to Streamlit Cloud for free. See [Streamlit deployment guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

---

## Getting Help

**Documentation:**
- This guide (FRONTEND_GUIDE.md)
- Main README (README.md)
- API documentation (GETTING_STARTED.md)
- Implementation details (IMPLEMENTATION_REPORT.md)

**Issues:**
- Check troubleshooting section above
- Verify parameters are valid
- Try loading Baseline preset

**Contact:**
- Report bugs: GitHub Issues
- Feature requests: GitHub Discussions
- Questions: Stack Overflow (tag: competitive-screening)

---

## Appendix: File Structure

```
Competitive_Sequential_Screening/
├── run_frontend.py          # Main launcher (recommended)
├── run_app.sh              # Shell launcher (Mac/Linux)
├── FRONTEND_GUIDE.md       # This file
│
├── competitive_screening/
│   ├── frontend/
│   │   └── app.py          # Streamlit app (900+ lines)
│   │
│   ├── presets/            # Built-in presets (6 JSON files)
│   │   ├── baseline.json
│   │   ├── narrow_types.json
│   │   ├── high_precision.json
│   │   ├── low_precision.json
│   │   ├── high_valuation.json
│   │   └── wide_types.json
│   │
│   └── src/                # Backend (DO NOT MODIFY from frontend)
│       ├── core/
│       │   ├── distributions.py
│       │   ├── equilibrium.py
│       │   └── welfare.py
│       └── analysis/
│           ├── parameter_sweep.py
│           └── visualization.py
│
└── requirements.txt        # Dependencies
```

---

## Changelog

**Version 1.0 (2026-01-23)**
- Initial release
- ✓ P0: Working computation, presets, ranking grid, one-command launch
- ✓ P1: Preset saving, formula reference, distribution previews, export
- ⏳ P2: Comparison mode (future), sensitivity sliders (future)

---

**Happy Exploring!** 🚀

If you find this tool useful, please cite the original paper and consider contributing improvements via GitHub.
