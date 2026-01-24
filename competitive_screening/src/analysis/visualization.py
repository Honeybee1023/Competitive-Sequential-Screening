"""
Visualization tools for Total Surplus analysis.

Creates publication-quality plots comparing welfare across market settings.
"""

from typing import Optional, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .parameter_sweep import SweepResult


# Color scheme for market settings (colorblind-friendly)
COLORS = {
    'NE': '#0173B2',  # Blue
    'SP': '#DE8F05',  # Orange
    'E': '#029E73',   # Green
    'MM': '#CC78BC'   # Purple
}

LINESTYLES = {
    'NE': '-',
    'SP': '--',
    'E': '-.',
    'MM': ':'
}

MARKERS = {
    'NE': 'o',
    'SP': 's',
    'E': '^',
    'MM': 'D'
}


def plot_ts_comparison(sweep_result: SweepResult,
                       title: Optional[str] = None,
                       xlabel: Optional[str] = None,
                       ylabel: str = "Total Surplus",
                       figsize: tuple = (10, 6),
                       save_path: Optional[str] = None) -> Figure:
    """
    Plot TS values for all settings as a function of parameter.

    Args:
        sweep_result: Result from parameter sweep
        title: Plot title (auto-generated if None)
        xlabel: X-axis label (uses parameter name if None)
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib Figure object

    Example:
        >>> from src.analysis import sweep_v0
        >>> result = sweep_v0([2, 4, 6, 8, 10], F, G)
        >>> fig = plot_ts_comparison(result, save_path='ts_vs_v0.png')
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = sweep_result.parameter_values

    # Plot each setting
    ax.plot(x, sweep_result.TS_NE, label='Non-Exclusive (NE)',
            color=COLORS['NE'], linestyle=LINESTYLES['NE'],
            marker=MARKERS['NE'], linewidth=2, markersize=6)

    ax.plot(x, sweep_result.TS_SP, label='Spot Pricing (SP)',
            color=COLORS['SP'], linestyle=LINESTYLES['SP'],
            marker=MARKERS['SP'], linewidth=2, markersize=6)

    ax.plot(x, sweep_result.TS_E, label='Exclusive (E)',
            color=COLORS['E'], linestyle=LINESTYLES['E'],
            marker=MARKERS['E'], linewidth=2, markersize=6)

    ax.plot(x, sweep_result.TS_MM, label='Multi-Good Monopoly (MM)',
            color=COLORS['MM'], linestyle=LINESTYLES['MM'],
            marker=MARKERS['MM'], linewidth=2, markersize=6)

    # Labels and formatting
    if xlabel is None:
        xlabel = sweep_result.parameter_name
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    if title is None:
        title = f"Total Surplus vs. {sweep_result.parameter_name}"
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_ts_rankings(sweep_result: SweepResult,
                    title: Optional[str] = None,
                    xlabel: Optional[str] = None,
                    figsize: tuple = (12, 7),
                    save_path: Optional[str] = None) -> Figure:
    """
    Plot TS rankings with colored regions showing which setting is best.

    Args:
        sweep_result: Result from parameter sweep
        title: Plot title
        xlabel: X-axis label
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib Figure

    Example:
        >>> fig = plot_ts_rankings(result, save_path='rankings.png')
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

    x = sweep_result.parameter_values

    # Top panel: TS values
    ax1.plot(x, sweep_result.TS_NE, label='NE', color=COLORS['NE'],
            linestyle=LINESTYLES['NE'], marker=MARKERS['NE'], linewidth=2, markersize=6)
    ax1.plot(x, sweep_result.TS_SP, label='SP', color=COLORS['SP'],
            linestyle=LINESTYLES['SP'], marker=MARKERS['SP'], linewidth=2, markersize=6)
    ax1.plot(x, sweep_result.TS_E, label='E', color=COLORS['E'],
            linestyle=LINESTYLES['E'], marker=MARKERS['E'], linewidth=2, markersize=6)
    ax1.plot(x, sweep_result.TS_MM, label='MM', color=COLORS['MM'],
            linestyle=LINESTYLES['MM'], marker=MARKERS['MM'], linewidth=2, markersize=6)

    ax1.set_ylabel('Total Surplus', fontsize=12, fontweight='bold')
    if title:
        ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Winner regions
    rankings = sweep_result.get_all_rankings()
    winners = [ranking[0] for ranking in rankings]

    # Create color map for winners
    winner_colors = {'NE': 0, 'SP': 1, 'E': 2, 'MM': 3}
    winner_indices = [winner_colors[w] for w in winners]

    # Plot as step function
    for i in range(len(x) - 1):
        winner = winners[i]
        ax2.axvspan(x[i], x[i+1], facecolor=COLORS[winner], alpha=0.3)
        ax2.text((x[i] + x[i+1]) / 2, 0.5, winner,
                ha='center', va='center', fontsize=10, fontweight='bold')

    # Handle last segment
    if len(x) > 1:
        winner = winners[-1]
        ax2.axvspan(x[-2], x[-1], facecolor=COLORS[winner], alpha=0.3)

    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    if xlabel is None:
        xlabel = sweep_result.parameter_name
    ax2.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax2.set_ylabel('Best', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_ts_differences(sweep_result: SweepResult,
                       baseline: str = 'SP',
                       title: Optional[str] = None,
                       xlabel: Optional[str] = None,
                       figsize: tuple = (10, 6),
                       save_path: Optional[str] = None) -> Figure:
    """
    Plot TS differences relative to baseline setting.

    Useful for seeing efficiency gains/losses compared to a reference.

    Args:
        sweep_result: Result from parameter sweep
        baseline: Setting to use as baseline ('NE', 'SP', 'E', or 'MM')
        title: Plot title
        xlabel: X-axis label
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib Figure

    Example:
        >>> # Show efficiency gains relative to Spot Pricing
        >>> fig = plot_ts_differences(result, baseline='SP')
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = sweep_result.parameter_values

    # Get baseline values
    baseline_map = {
        'NE': sweep_result.TS_NE,
        'SP': sweep_result.TS_SP,
        'E': sweep_result.TS_E,
        'MM': sweep_result.TS_MM
    }

    if baseline not in baseline_map:
        raise ValueError(f"baseline must be one of {list(baseline_map.keys())}")

    baseline_ts = baseline_map[baseline]

    # Plot differences
    for setting in ['NE', 'SP', 'E', 'MM']:
        if setting == baseline:
            continue  # Don't plot baseline against itself

        ts_values = baseline_map[setting]
        diff = ts_values - baseline_ts

        ax.plot(x, diff, label=f'{setting} - {baseline}',
               color=COLORS[setting], linestyle=LINESTYLES[setting],
               marker=MARKERS[setting], linewidth=2, markersize=6)

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Labels
    if xlabel is None:
        xlabel = sweep_result.parameter_name
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(f'TS Difference (relative to {baseline})', fontsize=12, fontweight='bold')

    if title is None:
        title = f'Total Surplus Differences vs. {baseline}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_efficiency_ratios(sweep_result: SweepResult,
                           title: Optional[str] = None,
                           xlabel: Optional[str] = None,
                           figsize: tuple = (10, 6),
                           save_path: Optional[str] = None) -> Figure:
    """
    Plot efficiency ratios: TS / TS_MM (MM is efficient benchmark).

    Shows what percentage of efficient surplus is achieved by each setting.

    Args:
        sweep_result: Result from parameter sweep
        title: Plot title
        xlabel: X-axis label
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib Figure

    Example:
        >>> fig = plot_efficiency_ratios(result)
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = sweep_result.parameter_values

    # Compute ratios relative to MM (efficient benchmark)
    ratio_NE = sweep_result.TS_NE / sweep_result.TS_MM * 100
    ratio_SP = sweep_result.TS_SP / sweep_result.TS_MM * 100
    ratio_E = sweep_result.TS_E / sweep_result.TS_MM * 100

    ax.plot(x, ratio_NE, label='NE', color=COLORS['NE'],
           linestyle=LINESTYLES['NE'], marker=MARKERS['NE'],
           linewidth=2, markersize=6)

    ax.plot(x, ratio_SP, label='SP', color=COLORS['SP'],
           linestyle=LINESTYLES['SP'], marker=MARKERS['SP'],
           linewidth=2, markersize=6)

    ax.plot(x, ratio_E, label='E', color=COLORS['E'],
           linestyle=LINESTYLES['E'], marker=MARKERS['E'],
           linewidth=2, markersize=6)

    # 100% line (efficient allocation)
    ax.axhline(y=100, color=COLORS['MM'], linestyle=LINESTYLES['MM'],
              linewidth=2, label='MM (100% efficient)', alpha=0.7)

    # Labels
    if xlabel is None:
        xlabel = sweep_result.parameter_name
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Efficiency (% of MM surplus)', fontsize=12, fontweight='bold')

    if title is None:
        title = 'Efficiency Relative to Multi-Good Monopoly'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([min(0, ax.get_ylim()[0]), 105])  # Show from 0 to slightly above 100%

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def create_dashboard(sweep_results: Dict[str, SweepResult],
                    save_path: Optional[str] = None) -> Figure:
    """
    Create a dashboard with multiple sweep results.

    Args:
        sweep_results: Dict mapping plot titles to SweepResult objects
        save_path: Path to save figure

    Returns:
        Matplotlib Figure with subplots

    Example:
        >>> results = {
        ...     'Varying v_0': sweep_v0([2,4,6,8,10], F, G),
        ...     'Varying σ': sweep_information_precision([0.5,1,2], 6, G)
        ... }
        >>> fig = create_dashboard(results, 'dashboard.png')
    """
    n_plots = len(sweep_results)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 5*n_plots))

    if n_plots == 1:
        axes = [axes]

    for ax, (title, result) in zip(axes, sweep_results.items()):
        x = result.parameter_values

        ax.plot(x, result.TS_NE, label='NE', color=COLORS['NE'],
               linestyle=LINESTYLES['NE'], marker=MARKERS['NE'], linewidth=2, markersize=6)
        ax.plot(x, result.TS_SP, label='SP', color=COLORS['SP'],
               linestyle=LINESTYLES['SP'], marker=MARKERS['SP'], linewidth=2, markersize=6)
        ax.plot(x, result.TS_E, label='E', color=COLORS['E'],
               linestyle=LINESTYLES['E'], marker=MARKERS['E'], linewidth=2, markersize=6)
        ax.plot(x, result.TS_MM, label='MM', color=COLORS['MM'],
               linestyle=LINESTYLES['MM'], marker=MARKERS['MM'], linewidth=2, markersize=6)

        ax.set_xlabel(result.parameter_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Total Surplus', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dashboard to {save_path}")

    return fig
