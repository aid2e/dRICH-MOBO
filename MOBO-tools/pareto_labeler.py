"""
pareto_labeler.py

Automates Pareto front point labeling. Given a PlotGenerator (pg) object and
the observed frontiers from get_observed_pareto_frontiers(), this module:
  1. Numbers each Pareto point left-to-right on the x-axis
  2. Draws a labeled matplotlib figure (draft quality, not final)
  3. Prints / returns a DataFrame with trial index, parameters, and objectives

Usage (in mobo_analysis.ipynb):
    from pareto_labeler import label_pareto_fronts
    tables = label_pareto_fronts(pg_march7_2026)

Returns a dict keyed by "(primary_metric, secondary_metric)" with DataFrames.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from ax.plot.pareto_utils import get_observed_pareto_frontiers


def _get_trial_params(experiment, arm_name):
    """Return parameter dict for a given arm_name from the experiment."""
    for trial in experiment.trials.values():
        for arm in trial.arms:
            if arm.name == arm_name:
                return arm.parameters
    return {}


def label_pareto_fronts(pg, save_dir=None, figsize=(7, 5), fontsize=12,
                        cmap="viridis", color_pad=0):
    """
    Label all Pareto fronts for a PlotGenerator experiment.

    Parameters
    ----------
    pg : PlotGenerator
        An initialised PlotGenerator object (has pg.experiment, pg.objectives_names,
        pg.parameters_names, pg.yaxis_label_dict, pg.xaxis_label_dict).
    save_dir : str or None
        If given, each figure is saved as a PDF to this directory.
        If None, figures are just displayed.
    figsize : tuple
        Matplotlib figure size.
    fontsize : int
        Font size for axis labels and annotations.
    cmap : str
        Matplotlib colormap name (e.g. "viridis", "plasma", "turbo", "rainbow").
    color_pad : int
        Extra padding added to the colormap normalization range, stretching the
        color spread so adjacent points look more distinct. 0 = default range.

    Returns
    -------
    tables : dict
        Keys are "(primary_metric) vs (secondary_metric)" strings.
        Values are DataFrames with columns:
            label, arm_name, trial_index,
            <secondary_metric>, <primary_metric>,
            <param1>, <param2>, ...
    """
    experiment = pg.experiment
    observed_frontiers = get_observed_pareto_frontiers(experiment=experiment, rel=False)

    # Build arm_name -> trial_index lookup from the experiment
    arm_to_trial = {}
    for trial_idx, trial in experiment.trials.items():
        for arm in trial.arms:
            arm_to_trial[arm.name] = trial_idx

    tables = {}

    for frontier in observed_frontiers:
        primary = frontier.primary_metric    # y-axis
        secondary = frontier.secondary_metric  # x-axis

        arm_names = frontier.arm_names
        means = frontier.means  # dict: metric_name -> list of floats

        x_vals = np.array(means[secondary])
        y_vals = np.array(means[primary])

        # Sort points left to right by x (secondary metric)
        order = np.argsort(x_vals)
        x_sorted = x_vals[order]
        y_sorted = y_vals[order]
        arms_sorted = [arm_names[i] for i in order]

        # --- Precompute colors for each point ---
        n_pts = len(x_sorted)
        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=1 - color_pad, vmax=n_pts + color_pad)
        hex_colors = [mcolors.to_hex(cmap_obj(norm(i))) for i in range(1, n_pts + 1)]

        # --- Build the DataFrame ---
        rows = []
        for label_num, (arm, x, y) in enumerate(zip(arms_sorted, x_sorted, y_sorted), start=1):
            trial_idx = arm_to_trial.get(arm, "?")
            params = _get_trial_params(experiment, arm)
            row = {
                "label": label_num,
                "arm_name": arm,
                "trial_index": trial_idx,
                secondary: round(x, 5),
                primary: round(y, 5),
                "hex_color": hex_colors[label_num - 1],
            }
            for p in pg.parameters_names:
                row[p] = round(params.get(p, float("nan")), 4) if isinstance(params.get(p), float) else params.get(p, "?")
            rows.append(row)

        df = pd.DataFrame(rows).set_index("label")
        key = f"{primary} vs {secondary}"
        tables[key] = df

        # --- Print table ---
        xlabel = pg.yaxis_label_dict.get(secondary, secondary)
        ylabel = pg.yaxis_label_dict.get(primary, primary)
        print(f"\n{'='*70}")
        print(f"  Pareto front: {ylabel}  vs  {xlabel}")
        print(f"{'='*70}")
        print(df.to_string())

        # --- Draw labeled matplotlib figure ---
        fig, ax = plt.subplots(figsize=figsize)

        # Separate status_quo from regular points
        is_sq = [arm == "status_quo" for arm in arms_sorted]
        reg_mask = [not sq for sq in is_sq]

        # Plot regular points with colormap
        reg_x = x_sorted[reg_mask]
        reg_y = y_sorted[reg_mask]
        reg_indices = [i + 1 for i, sq in enumerate(is_sq) if not sq]
        sc = ax.scatter(reg_x, reg_y, c=reg_indices, cmap=cmap_obj, norm=norm, s=60, zorder=3)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Trial number", fontsize=fontsize)

        # Plot status_quo as black star
        sq_x = x_sorted[is_sq]
        sq_y = y_sorted[is_sq]
        if len(sq_x) > 0:
            ax.scatter(sq_x, sq_y, marker="*", c="black", s=200, zorder=4)

        for label_num, (x, y, arm) in enumerate(zip(x_sorted, y_sorted, arms_sorted), start=1):
            trial_idx = arm_to_trial.get(arm, "?")
            label_text = "SQ" if arm == "status_quo" else str(label_num)
            ax.annotate(
                label_text,
                xy=(x, y),
                xytext=(0, 8),
                ha="center",
                textcoords="offset points",
                fontsize=fontsize,
                fontweight="bold",
            )

        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        x_margin = (x_sorted.max() - x_sorted.min()) * 0.08
        y_margin = (y_sorted.max() - y_sorted.min()) * 0.12  # extra top margin for labels
        ax.set_xlim(x_sorted.min() - x_margin, x_sorted.max() + x_margin)
        ax.set_ylim(y_sorted.min() - y_margin * 0.6, y_sorted.max() + y_margin)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            import os
            fname = os.path.join(save_dir, f"labeled__{primary}__vs__{secondary}.pdf")
            fig.savefig(fname, format="pdf")
            print(f"\nSaved: {fname}")
        else:
            plt.show()

    return tables


def plot_dual_pareto(pg, save_path=None, figsize=(14, 5.5), fontsize=12,
                     cmap="viridis", color_pad=0):
    """
    Plot two Pareto front projections side-by-side with consistent labeling.

    Left panel: low_RMSE vs high_RMSE.  Right panel: low_muID_auc vs high_muID_auc.
    Points are labeled 1..N sorted by low_RMSE ascending (left-to-right on left panel).
    The same label refers to the same trial in both panels.  Status quo shown as
    a black star labeled "SQ".

    Parameters
    ----------
    pg : PlotGenerator
    save_path : str or None
        If given, figure saved as PDF to this path.
    figsize, fontsize, cmap, color_pad : see label_pareto_fronts

    Returns
    -------
    df : DataFrame
        Unified table with consistent labels, all 4 objectives, parameters, hex_color.
    """
    experiment = pg.experiment
    observed_frontiers = get_observed_pareto_frontiers(experiment=experiment, rel=False)

    # --- Find the two frontiers we need ---
    frontier_rmse = None
    frontier_muid = None
    for f in observed_frontiers:
        metrics = {f.primary_metric, f.secondary_metric}
        if "low_RMSE" in metrics and "high_RMSE" in metrics:
            frontier_rmse = f
        elif "low_muID_auc" in metrics and "high_muID_auc" in metrics:
            frontier_muid = f

    if frontier_rmse is None or frontier_muid is None:
        raise ValueError(
            f"Could not find both RMSE and muID frontiers. "
            f"Available metric pairs: "
            f"{[(f.primary_metric, f.secondary_metric) for f in observed_frontiers]}"
        )

    # --- Build arm_name -> trial_index lookup ---
    arm_to_trial = {}
    for trial_idx, trial in experiment.trials.items():
        for arm in trial.arms:
            arm_to_trial[arm.name] = trial_idx

    # --- Get unified arm list and all 4 objective values ---
    # Both frontiers share the same arms (Ax computes one 4D Pareto front)
    arm_names = frontier_rmse.arm_names
    low_rmse_vals = np.array(frontier_rmse.means["low_RMSE"])
    high_rmse_vals = np.array(frontier_rmse.means["high_RMSE"])
    low_muid_vals = np.array(frontier_muid.means["low_muID_auc"])
    high_muid_vals = np.array(frontier_muid.means["high_muID_auc"])

    # --- Sort by low_RMSE ascending for consistent labeling ---
    order = np.argsort(low_rmse_vals)
    arm_names_sorted = [arm_names[i] for i in order]
    low_rmse_sorted = low_rmse_vals[order]
    high_rmse_sorted = high_rmse_vals[order]
    low_muid_sorted = low_muid_vals[order]
    high_muid_sorted = high_muid_vals[order]

    # --- Assign labels: 1..N, status_quo -> "SQ" ---
    n_pts = len(arm_names_sorted)
    labels = []
    label_counter = 1
    for arm in arm_names_sorted:
        if arm == "status_quo":
            labels.append("SQ")
        else:
            labels.append(str(label_counter))
            label_counter += 1

    # --- Precompute colors (by label order, skipping SQ) ---
    n_regular = sum(1 for l in labels if l != "SQ")
    cmap_obj = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=1 - color_pad, vmax=n_regular + color_pad)

    # --- Build DataFrame ---
    rows = []
    reg_counter = 0
    for i, arm in enumerate(arm_names_sorted):
        trial_idx = arm_to_trial.get(arm, "?")
        params = _get_trial_params(experiment, arm)
        if arm == "status_quo":
            hex_color = "#000000"
        else:
            reg_counter += 1
            hex_color = mcolors.to_hex(cmap_obj(norm(reg_counter)))
        row = {
            "label": labels[i],
            "arm_name": arm,
            "trial_index": trial_idx,
            "low_RMSE": round(low_rmse_sorted[i], 5),
            "high_RMSE": round(high_rmse_sorted[i], 5),
            "low_muID_auc": round(low_muid_sorted[i], 5),
            "high_muID_auc": round(high_muid_sorted[i], 5),
            "hex_color": hex_color,
        }
        for p in pg.parameters_names:
            val = params.get(p, float("nan"))
            row[p] = round(val, 4) if isinstance(val, float) else val
        rows.append(row)

    df = pd.DataFrame(rows).set_index("label")

    # --- Print table ---
    print(f"\n{'='*90}")
    print(f"  Unified Pareto front (sorted by low_RMSE)")
    print(f"{'='*90}")
    print(df.to_string())

    # --- Helper to plot one panel ---
    def _plot_panel(ax, x_vals, y_vals, xlabel, ylabel):
        is_sq = [arm == "status_quo" for arm in arm_names_sorted]
        reg_mask = np.array([not sq for sq in is_sq])

        # Regular points
        reg_x = x_vals[reg_mask]
        reg_y = y_vals[reg_mask]
        reg_colors = list(range(1, n_regular + 1))
        sc = ax.scatter(reg_x, reg_y, c=reg_colors, cmap=cmap_obj, norm=norm,
                        s=60, zorder=3, edgecolors="none")

        # Status quo
        sq_mask = np.array(is_sq)
        if sq_mask.any():
            ax.scatter(x_vals[sq_mask], y_vals[sq_mask], marker="*",
                       c="black", s=200, zorder=4)

        # Labels
        for i, (x, y, arm) in enumerate(zip(x_vals, y_vals, arm_names_sorted)):
            label_text = "SQ" if arm == "status_quo" else labels[i]
            ax.annotate(
                label_text, xy=(x, y), xytext=(0, 8), ha="center",
                textcoords="offset points", fontsize=fontsize, fontweight="bold",
            )

        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        x_margin = (x_vals.max() - x_vals.min()) * 0.08
        y_margin = (y_vals.max() - y_vals.min()) * 0.12
        ax.set_xlim(x_vals.min() - x_margin, x_vals.max() + x_margin)
        ax.set_ylim(y_vals.min() - y_margin * 0.6, y_vals.max() + y_margin)
        ax.grid(True, alpha=0.3)
        return sc

    # --- Create figure with nested gridspecs: plots together, colorbar close ---
    fig = plt.figure(figsize=figsize)
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.025], wspace=0.02)
    inner_gs = outer_gs[0].subgridspec(1, 2, wspace=0.25)
    ax1 = fig.add_subplot(inner_gs[0])
    ax2 = fig.add_subplot(inner_gs[1])
    cax = fig.add_subplot(outer_gs[1])

    sc1 = _plot_panel(
        ax1, low_rmse_sorted, high_rmse_sorted,
        pg.yaxis_label_dict.get("low_RMSE", "low_RMSE"),
        pg.yaxis_label_dict.get("high_RMSE", "high_RMSE"),
    )
    sc2 = _plot_panel(
        ax2, low_muid_sorted, high_muid_sorted,
        pg.yaxis_label_dict.get("low_muID_auc", "low_muID_auc"),
        pg.yaxis_label_dict.get("high_muID_auc", "high_muID_auc"),
    )

    # Colorbar in dedicated axis
    cbar = fig.colorbar(sc1, cax=cax)
    cbar.set_label("Trial number", fontsize=fontsize)

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"\nSaved: {save_path}")
    else:
        plt.show()

    return df
