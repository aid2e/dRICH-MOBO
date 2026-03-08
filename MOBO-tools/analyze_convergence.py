#!/usr/bin/env python3
"""
MOBO Convergence Analysis
=========================
Plots hypervolume indicator (HVI) and per-objective best-so-far curves
to assess whether the optimization converged and whether more trials
would have helped.

Usage:
    source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
    python3 analyze_convergence.py                              # uses default CSV
    python3 analyze_convergence.py --csv path/to/scheduler.csv
    python3 analyze_convergence.py --csv exp1.csv --csv exp2.csv  # compare runs

Outputs (in plots/convergence/):
    hvi_curve.pdf          - Hypervolume indicator vs. trial number
    best_so_far.pdf        - Per-objective best-so-far curves
    pareto_scatter.pdf     - Pairwise Pareto front scatter (feasible trials)
    convergence_report.txt - Numeric summary (HVI gain by phase, plateau trial)
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ---------------------------------------------------------------------------
# Objective configuration
# MOBO minimizes RMSE, maximizes AUC.  For hypervolume all objectives must be
# framed as MAXIMIZE, so we negate the RMSE objectives.
# ---------------------------------------------------------------------------
OBJECTIVES = {
    "low_RMSE":    {"direction": "minimize", "label": "Low-E RMSE (GeV)", "color": "#e66101"},
    "high_RMSE":   {"direction": "minimize", "label": "High-E RMSE (GeV)", "color": "#d01c8b"},
    "low_muID_auc":  {"direction": "maximize", "label": "Low-E muID AUC",  "color": "#4dac26"},
    "high_muID_auc": {"direction": "maximize", "label": "High-E muID AUC", "color": "#0571b0"},
}

# Reference point for hypervolume (nadir point, slightly beyond worst feasible).
# Defined in the MAXIMIZED space (negated RMSE).
# These are set conservatively; edit if your runs explore a wider range.
HV_REFERENCE = np.array([-1.2, -1.2, 0.85, 0.85])  # [-max_low_RMSE, -max_high_RMSE, min_low_AUC, min_high_AUC]


def load_csv(path):
    df = pd.read_csv(path, index_col=0)
    return df


def feasible_completed(df):
    mask = (df["trial_status"] == "COMPLETED") & (df["is_feasible"] == True)
    return df[mask].copy()


def to_maximized(df):
    """Return array of shape (N, 4) in maximized space (negate RMSE)."""
    cols = list(OBJECTIVES.keys())
    arr = df[cols].values.astype(float).copy()
    for i, (col, cfg) in enumerate(OBJECTIVES.items()):
        if cfg["direction"] == "minimize":
            arr[:, i] = -arr[:, i]
    return arr


def compute_hypervolume(points, ref):
    """
    Compute exact hypervolume for small N using botorch if available,
    otherwise fall back to a Monte Carlo estimate.
    """
    try:
        import torch
        from botorch.utils.multi_objective.hypervolume import Hypervolume
        hv = Hypervolume(ref_point=torch.tensor(ref))
        result = hv.compute(torch.tensor(points))
        return result.item() if hasattr(result, "item") else float(result)
    except ImportError:
        # Simple Monte Carlo fallback (less accurate but dependency-free)
        n_samples = 200_000
        rng = np.random.default_rng(0)
        lb = ref
        ub = points.max(axis=0) + 1e-6
        sample = rng.uniform(lb, ub, size=(n_samples, len(ref)))
        dominated = np.any(
            np.all(sample[None, :, :] <= points[:, None, :], axis=2), axis=0
        )
        vol = np.prod(ub - lb)
        return vol * dominated.mean()


def cumulative_hvi(df_all, ref):
    """
    For each completed trial (in order), compute HVI of feasible set up to
    and including that trial.
    """
    completed = df_all[df_all["trial_status"] == "COMPLETED"].sort_values("trial_index")
    hvi_vals = []
    trial_indices = []

    feasible_so_far = []
    for _, row in completed.iterrows():
        idx = int(row["trial_index"])
        if row["is_feasible"] == True and not pd.isna(row.get("low_RMSE", np.nan)):
            vals = []
            for col, cfg in OBJECTIVES.items():
                v = float(row[col])
                vals.append(-v if cfg["direction"] == "minimize" else v)
            feasible_so_far.append(vals)

        if feasible_so_far:
            pts = np.array(feasible_so_far)
            hv = compute_hypervolume(pts, ref)
        else:
            hv = 0.0

        trial_indices.append(idx)
        hvi_vals.append(hv)

    return np.array(trial_indices), np.array(hvi_vals)


def best_so_far(df_all):
    """Per-objective cumulative best (only over feasible completed trials)."""
    completed = df_all[df_all["trial_status"] == "COMPLETED"].sort_values("trial_index")
    bests = {col: [] for col in OBJECTIVES}
    trial_indices = []
    running = {col: None for col in OBJECTIVES}

    for _, row in completed.iterrows():
        idx = int(row["trial_index"])
        trial_indices.append(idx)

        if row["is_feasible"] == True and not pd.isna(row.get("low_RMSE", np.nan)):
            for col, cfg in OBJECTIVES.items():
                v = float(row[col])
                if running[col] is None:
                    running[col] = v
                else:
                    if cfg["direction"] == "minimize":
                        running[col] = min(running[col], v)
                    else:
                        running[col] = max(running[col], v)

        for col in OBJECTIVES:
            bests[col].append(running[col])

    return np.array(trial_indices), bests


def find_plateau_trial(trial_indices, hvi_vals, window=5, threshold_frac=0.005):
    """
    Return the trial index at which HVI improvement (over a rolling window)
    drops below threshold_frac of total HVI gain.
    Returns None if never plateaued.
    """
    total_gain = hvi_vals[-1] - hvi_vals[0]
    if total_gain <= 0:
        return None
    for i in range(window, len(hvi_vals)):
        recent_gain = hvi_vals[i] - hvi_vals[i - window]
        if recent_gain < threshold_frac * total_gain:
            return int(trial_indices[i])
    return None


def sobol_cutoff(df):
    """Return the last trial index that used Sobol generation."""
    sobol = df[df["generation_method"] == "Sobol"]
    if sobol.empty:
        return None
    return int(sobol["trial_index"].max())


def plot_hvi(datasets, out_path):
    """Plot HVI curves for one or more experiments."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1b7837", "#762a83", "#d73027", "#4393c3"]

    for i, (label, trial_indices, hvi_vals, sobol_end) in enumerate(datasets):
        color = colors[i % len(colors)]
        ax.plot(trial_indices, hvi_vals, color=color, lw=2, label=label)
        ax.fill_between(trial_indices, hvi_vals, alpha=0.10, color=color)
        if sobol_end is not None:
            ax.axvline(sobol_end + 0.5, color=color, ls="--", lw=1, alpha=0.6,
                       label=f"Sobol→BoTorch ({label})" if len(datasets) > 1 else "Sobol→BoTorch")

    ax.set_xlabel("Trial index", fontsize=13)
    ax.set_ylabel("Hypervolume indicator", fontsize=13)
    ax.set_title("MOBO convergence — Hypervolume indicator vs. trial", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_best_so_far(trial_indices, bests, sobol_end, out_path):
    """2×2 grid of per-objective best-so-far."""
    cols = list(OBJECTIVES.keys())
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    for i, (col, cfg) in enumerate(OBJECTIVES.items()):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        vals = bests[col]
        ax.plot(trial_indices, vals, color=cfg["color"], lw=2)
        ax.fill_between(trial_indices, vals, alpha=0.15, color=cfg["color"])
        if sobol_end is not None:
            ax.axvline(sobol_end + 0.5, color="gray", ls="--", lw=1, label="Sobol→BoTorch")
            ax.legend(fontsize=8)
        ax.set_xlabel("Trial", fontsize=11)
        ax.set_ylabel(cfg["label"], fontsize=11)
        direction_str = "↓ minimize" if cfg["direction"] == "minimize" else "↑ maximize"
        ax.set_title(f"{col}  ({direction_str})", fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Best-so-far per objective (feasible trials only)", fontsize=13)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_pareto_scatter(df_feasible, sobol_end, out_path):
    """Pairwise scatter: low_RMSE vs high_RMSE, colored by generation phase."""
    fig, ax = plt.subplots(figsize=(7, 6))

    sobol_mask = df_feasible["generation_method"] == "Sobol"
    botorch_mask = ~sobol_mask

    ax.scatter(df_feasible.loc[sobol_mask, "low_RMSE"],
               df_feasible.loc[sobol_mask, "high_RMSE"],
               c="steelblue", s=50, alpha=0.7, label="Sobol", zorder=3)
    ax.scatter(df_feasible.loc[botorch_mask, "low_RMSE"],
               df_feasible.loc[botorch_mask, "high_RMSE"],
               c="darkorange", s=50, alpha=0.7, label="BoTorch", zorder=3)

    # Mark status_quo
    sq = df_feasible[df_feasible.get("arm_name", pd.Series()).eq("status_quo")] if "arm_name" in df_feasible else pd.DataFrame()
    if not sq.empty:
        ax.scatter(sq["low_RMSE"], sq["high_RMSE"], c="red", s=120, marker="*",
                   label="Status quo", zorder=5)

    ax.set_xlabel("Low-E RMSE (GeV)  ↓ better", fontsize=12)
    ax.set_ylabel("High-E RMSE (GeV)  ↓ better", fontsize=12)
    ax.set_title("RMSE trade-off: Sobol vs BoTorch trials", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def write_report(datasets, df_all, out_path):
    feas = feasible_completed(df_all)
    sobol_end = sobol_cutoff(df_all)
    n_total = len(df_all[df_all["trial_status"] == "COMPLETED"])
    n_feasible = len(feas)
    n_failed = len(df_all[df_all["trial_status"] == "FAILED"])
    n_sobol = len(df_all[df_all["generation_method"] == "Sobol"])
    n_botorch = len(df_all[df_all["generation_method"] == "BoTorch"])

    lines = []
    lines.append("=" * 60)
    lines.append("MOBO CONVERGENCE REPORT")
    lines.append("=" * 60)
    lines.append(f"Total completed trials : {n_total}")
    lines.append(f"  Feasible            : {n_feasible}")
    lines.append(f"  Failed              : {n_failed}")
    lines.append(f"Sobol trials          : {n_sobol}  (trials 1–{sobol_end})")
    lines.append(f"BoTorch trials        : {n_botorch}  (trials {sobol_end+1}–{n_total-1})")
    lines.append("")

    for label, trial_indices, hvi_vals, sobol_end_i in datasets:
        lines.append(f"--- {label} ---")
        if len(hvi_vals) < 2:
            lines.append("  Not enough data.")
            continue
        sobol_hvi = hvi_vals[sobol_end_i] if sobol_end_i < len(hvi_vals) else hvi_vals[-1]
        final_hvi = hvi_vals[-1]
        total_gain = final_hvi - hvi_vals[0]
        sobol_gain = sobol_hvi - hvi_vals[0]
        botorch_gain = final_hvi - sobol_hvi

        lines.append(f"  HVI at end of Sobol  : {sobol_hvi:.4f}")
        lines.append(f"  HVI at final trial   : {final_hvi:.4f}")
        lines.append(f"  Total HVI gain       : {total_gain:.4f}")
        lines.append(f"  Sobol gain fraction  : {100*sobol_gain/total_gain:.1f}%")
        lines.append(f"  BoTorch gain fraction: {100*botorch_gain/total_gain:.1f}%")

        plateau = find_plateau_trial(trial_indices, hvi_vals)
        if plateau is not None:
            lines.append(f"  Plateau detected at trial {plateau}  (< 0.5% gain per 5-trial window)")
            lines.append(f"  => Trials after {plateau} added little; you could reduce to ~{plateau+5} trials")
        else:
            lines.append("  No plateau detected — HVI was still improving at the last trial.")
            lines.append("  => Consider running MORE trials.")
        lines.append("")

    lines.append("--- Per-objective best (feasible) ---")
    for col, cfg in OBJECTIVES.items():
        vals = feas[col].dropna()
        if vals.empty:
            continue
        best = vals.min() if cfg["direction"] == "minimize" else vals.max()
        best_trial = int(feas.loc[vals.idxmin() if cfg["direction"] == "minimize" else vals.idxmax(), "trial_index"])
        lines.append(f"  {col:20s}: best = {best:.4f}  (trial {best_trial})")
    lines.append("")
    lines.append("--- Status quo vs. MOBO best ---")
    sq = df_all[df_all.get("arm_name", pd.Series()).eq("status_quo")] if "arm_name" in df_all else pd.DataFrame()
    if not sq.empty:
        sq = sq.iloc[0]
        for col, cfg in OBJECTIVES.items():
            if col not in sq or pd.isna(sq[col]):
                continue
            sq_val = float(sq[col])
            feas_col = feas[col].dropna()
            if feas_col.empty:
                continue
            best = feas_col.min() if cfg["direction"] == "minimize" else feas_col.max()
            delta = sq_val - best if cfg["direction"] == "minimize" else best - sq_val
            pct = 100 * delta / abs(sq_val) if sq_val != 0 else 0
            lines.append(f"  {col:20s}: SQ={sq_val:.4f}  best={best:.4f}  Δ={delta:+.4f} ({pct:+.1f}%)")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="MOBO convergence analysis")
    parser.add_argument(
        "--csv", action="append", dest="csvs",
        default=None,
        help="Path to test_scheduler_df.csv (repeat for multiple experiments)"
    )
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: plots/convergence/ next to first CSV)")
    args = parser.parse_args()

    DEFAULT_CSV = Path(__file__).parent / "experiment_high_rmse_low_rmse_high_mupi_low_mupi_march_6_2026" / "test_scheduler_df.csv"
    csvs = [Path(c) for c in args.csvs] if args.csvs else [DEFAULT_CSV]

    if args.outdir:
        out_dir = Path(args.outdir)
    else:
        out_dir = csvs[0].parent.parent / "plots" / "convergence"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir}")
    print(f"Reference point (maximized space): {HV_REFERENCE}")
    print()

    # Load all experiments
    all_dfs = [(p.stem, load_csv(p)) for p in csvs]

    # Build HVI datasets
    hvi_datasets = []
    for label, df in all_dfs:
        t_idx, hvi_vals = cumulative_hvi(df, HV_REFERENCE)
        s_end = sobol_cutoff(df)
        # Find index in t_idx corresponding to sobol_end trial
        sobol_idx = int(np.searchsorted(t_idx, s_end)) if s_end is not None else len(t_idx) // 3
        hvi_datasets.append((label, t_idx, hvi_vals, sobol_idx))

    # Use first experiment for per-objective plots
    primary_label, primary_df = all_dfs[0]
    t_idx_p, bests_p = best_so_far(primary_df)
    s_end_p = sobol_cutoff(primary_df)
    feas_p = feasible_completed(primary_df)

    # Generate plots
    print("Generating plots...")
    plot_hvi(hvi_datasets, out_dir / "hvi_curve.pdf")
    plot_best_so_far(t_idx_p, bests_p, s_end_p, out_dir / "best_so_far.pdf")
    if not feas_p.empty:
        plot_pareto_scatter(feas_p, s_end_p, out_dir / "pareto_scatter.pdf")

    # Write text report
    print("\nConvergence report:")
    write_report(hvi_datasets, primary_df, out_dir / "convergence_report.txt")


if __name__ == "__main__":
    main()
