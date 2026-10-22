"""Static matplotlib figures exported with the artifacts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

COMPOUND_COLOURS = {"SOFT": "#e10600", "MEDIUM": "#f5c400", "HARD": "#8a8a8a", "UNKNOWN": "#4a6fa5"}


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def degradation_curves_figure(curves: pd.DataFrame, path: Path) -> Path:
    """Median relative pace vs tyre age per compound with IQR band and fitted curve."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for compound, group in curves[curves["race_id"] == "all"].groupby("compound"):
        colour = COMPOUND_COLOURS.get(str(compound), "black")
        ax.fill_between(group["tyre_age"], group["q25_s"], group["q75_s"], color=colour, alpha=0.15)
        ax.plot(group["tyre_age"], group["median_s"], "o", ms=3, color=colour, label=str(compound))
        ax.plot(group["tyre_age"], group["fitted_s"], "-", color=colour)
    ax.set(
        xlabel="Tyre age (laps)",
        ylabel="Pace vs field, driver-adjusted (s)",
        title="Degradation curves (train races)",
    )
    ax.legend()
    return _save(fig, path)


def horizon_figure(horizon: pd.DataFrame, path: Path) -> Path:
    """Rollout MAE by horizon k per model (validation and test)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (model, split), group in horizon.groupby(["model", "split"]):
        ax.plot(
            group["k"],
            group["mae"],
            marker="o",
            ls="-" if split == "val" else "--",
            label=f"{model} ({split})",
        )
    ax.set(xlabel="Laps ahead (k)", ylabel="MAE (s)", title="Multi-lap forecast error")
    ax.legend(fontsize=8)
    return _save(fig, path)


def pred_vs_actual_figure(predictions: pd.DataFrame, model: str, path: Path) -> Path:
    """Scatter of predicted vs actual next-lap time on held-out races."""
    held = predictions[(predictions["model"] == model) & predictions["split"].isin(["val", "test"])]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(held["actual_s"], held["pred_s"], s=4, alpha=0.4)
    low, high = held["actual_s"].quantile([0.01, 0.99])
    ax.plot([low, high], [low, high], "k--", lw=1)
    ax.set(
        xlim=(low, high),
        ylim=(low, high),
        xlabel="Actual (s)",
        ylabel="Predicted (s)",
        title=f"Next-lap predictions — {model}",
    )
    return _save(fig, path)


def pit_loss_figure(losses: pd.DataFrame, median_s: float, path: Path) -> Path:
    """Histogram of green-flag stop losses."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(losses["loss_s"].clip(10, 40), bins=30, color="#4a6fa5")
    ax.axvline(median_s, color="k", ls="--", label=f"median {median_s:.1f} s")
    ax.set(xlabel="Pit loss (s)", ylabel="Stops", title="Green-flag pit loss")
    ax.legend()
    return _save(fig, path)


def error_distribution_figure(predictions: pd.DataFrame, model: str, path: Path) -> Path:
    """Absolute next-lap error by compound on held-out races."""
    held = predictions[(predictions["model"] == model) & predictions["split"].isin(["val", "test"])]
    held = held.assign(abs_error=(held["pred_s"] - held["actual_s"]).abs())
    compounds = sorted(held["compound"].unique())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        [held.loc[held["compound"] == c, "abs_error"] for c in compounds],
        tick_labels=compounds,
        showfliers=False,
    )
    ax.set(ylabel="|error| (s)", title=f"Next-lap error by compound — {model}")
    return _save(fig, path)
