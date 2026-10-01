"""Step 4: charts for the README / write-up."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "text.parse_math": False})
COL = {"Baseline (56-day empirical)": "#9ca3af", "LightGBM (quantile)": "#2563eb", "Hierarchical Bayes": "#ea580c"}
PCT = mt.PercentFormatter(1.0, decimals=0)
PCT1 = mt.PercentFormatter(1.0, decimals=1)


def _label(ax, bars, vals):
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:+.1%}", (b.get_x() + b.get_width() / 2, v), xytext=(0, -4 if v < 0 else 3),
                    textcoords="offset points", ha="center", va="top" if v < 0 else "bottom", fontsize=8)


def calibration():
    cal = pd.read_csv(OUT / "eval_calibration.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="#111827", lw=0.8, ls="--", label="Perfect calibration")
    for m, g in cal.groupby("Model", sort=False):
        off = {"Baseline (56-day empirical)": -0.008, "LightGBM (quantile)": 0, "Hierarchical Bayes": 0.008}[m]
        ax.vlines(g.Nominal + off, g["P(y<Q)"], g["P(y<=Q)"], color=COL[m], lw=3, alpha=0.6)
        ax.plot(g.Nominal + off, g["Coverage (mid)"], marker="o", ms=4, lw=0, color=COL[m], label=m)
    ax.set_xlabel("Forecast quantile level"); ax.set_ylabel("Share of actual demand at or below forecast")
    ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Calibration (bars span P(y<Q) to P(y<=Q))", loc="left", fontweight="bold")
    ax.text(0.98, 0.03, "Well calibrated if the bar crosses the diagonal.\nCount data: the 10% quantile is often 0,\nso its bar is long by construction.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#6b7280")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "01_calibration.png", dpi=150); plt.close(fig)


def frontier():
    nv = pd.read_csv(OUT / "eval_newsvendor_sweep.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, g in nv.groupby("Model", sort=False):
        ax.plot(g["Waste % of ordered"], g["Fill rate"], marker="o", color=COL[m], label=m)
        for _, r in g.iterrows():
            if m == "LightGBM (quantile)":
                ax.annotate(f"CR {r.CR:.0%}", (r["Waste % of ordered"], r["Fill rate"]), fontsize=8,
                            xytext=(6, -10), textcoords="offset points", color=COL[m])
    ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("Waste (% of units ordered)"); ax.set_ylabel("Fill rate (% of demand served)")
    ax.set_title("Stockouts vs waste: each point is an ordering policy (critical ratio)", loc="left", fontweight="bold")
    ax.text(0.98, 0.04, "Up and to the left is better", transform=ax.transAxes, ha="right", fontsize=9, color="#6b7280")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.08))
    fig.tight_layout(); fig.savefig(FIG / "02_tradeoff_frontier.png", dpi=150); plt.close(fig)


def cost_vs_baseline():
    nv = pd.read_csv(OUT / "eval_newsvendor_sweep.csv")
    nv = nv[nv.Model != "Baseline (56-day empirical)"]
    crs = sorted(nv.CR.unique())
    x = np.arange(len(crs)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (m, g) in enumerate(nv.groupby("Model", sort=False)):
        v = g.set_index("CR").loc[crs, "Cost vs baseline"].to_numpy()
        bars = ax.bar(x + (i - 0.5) * w, v, w, color=COL[m], label=m)
        _label(ax, bars, v)
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xticks(x, [f"{c:.0%}" for c in crs]); ax.set_xlabel("Critical ratio (target service level)")
    ax.yaxis.set_major_formatter(PCT); ax.set_ylabel("Inventory cost vs baseline")
    ax.set_title("Newsvendor cost (stockouts + waste) relative to the baseline", loc="left", fontweight="bold")
    ax.set_ylim(nv["Cost vs baseline"].min() * 1.25, 0.02)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "03_cost_vs_baseline.png", dpi=150); plt.close(fig)


def segments():
    s = pd.read_csv(OUT / "eval_segment_velocity.csv")
    piv = s.pivot(index="velocity", columns="Model", values="cost")
    order = ["Slow (<0.5/day)", "Medium (0.5-2/day)", "Fast (>2/day)"]
    piv = piv.loc[order]
    rel = piv.div(piv["Baseline (56-day empirical)"], axis=0) - 1
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(["LightGBM (quantile)", "Hierarchical Bayes"]):
        bars = ax.bar(x + (i - 0.5) * w, rel[m], w, color=COL[m], label=m)
        _label(ax, bars, rel[m].to_numpy())
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xticks(x, order); ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Fresh-grocery cost vs baseline (CR 30%)")
    ax.set_title("Where better forecasts save money, by item sales velocity", loc="left", fontweight="bold")
    ax.set_ylim(min(rel.min().min() * 1.4, -0.01), max(rel.max().max() * 1.6, 0.01))
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "04_cost_by_velocity.png", dpi=150); plt.close(fig)


def fan_charts():
    feat = pd.read_parquet(ROOT / "data" / "foods3_features.parquet", columns=["series_id", "d", "date", "sales"])
    test_ids = pd.read_parquet(OUT / "fc_lightgbm.parquet")
    vel = feat[(feat.d > 1829) & (feat.d <= 1941)].groupby("series_id", observed=True).sales.mean()
    picks = [vel[(vel > 15)].sort_values().index[len(vel[vel > 15]) // 2],
             vel[(vel > 0.4) & (vel < 0.8)].sort_values().index[len(vel[(vel > 0.4) & (vel < 0.8)]) // 2]]
    fcs = {m: pd.read_parquet(OUT / f"fc_{f}.parquet") for m, f in
           [("LightGBM (quantile)", "lightgbm"), ("Hierarchical Bayes", "hierarchical")]}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, sid in zip(axes, picks):
        h = feat[(feat.series_id == sid) & (feat.d > 1941 - 56)]
        ax.plot(h.date, h.sales, color="#111827", lw=1, label="Actual sales")
        for m, f in fcs.items():
            g = f[f.series_id == sid].sort_values("d")
            dates = feat[(feat.series_id == sid) & (feat.d >= 1942)].date.to_numpy()
            ax.fill_between(dates, g.q10, g.q90, color=COL[m], alpha=0.15)
            ax.plot(dates, g.q50, color=COL[m], lw=1.2, label=f"{m}: median, 10-90% band")
        ax.axvline(feat[feat.d == 1942].date.iloc[0], color="#6b7280", ls=":", lw=1)
        ax.set_title(f"{sid}  (avg {vel[sid]:.1f} units/day)", loc="left", fontweight="bold", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "05_example_forecasts.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    calibration(); frontier(); cost_vs_baseline(); segments(); fan_charts()
    print("charts written to", FIG)
