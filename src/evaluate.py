"""Step 3: evaluate forecasts on accuracy AND on business outcomes (newsvendor).

Newsvendor logic. Each day a store orders Q units of an item. Unsold units are wasted
(perishable, no carry-over); unmet demand is a lost sale.
  underage cost cu = lost margin per unit short
  overage  cost co = unit cost of a unit thrown away (minus any salvage)
The cost-minimising order is the quantile of demand at the critical ratio CR = cu / (cu + co).
So a probabilistic forecast turns into an ordering policy simply by reading off the
CR-quantile. Note: newsvendor cost with cu = c*p, co = (1-c)*p is exactly the price-weighted
pinball loss at quantile c, so better quantile forecasts mean lower inventory cost.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FEAT = ROOT / "data" / "foods3_features.parquet"

QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
QCOLS = [f"q{int(q * 100)}" for q in QUANTILES]
CRS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
MODELS = {"Baseline (56-day empirical)": "baseline", "LightGBM (quantile)": "lightgbm",
          "Hierarchical Bayes": "hierarchical"}

# Grocery economics (assumption, stated in the write-up)
GROSS_MARGIN = 0.30          # margin on retail price
SALVAGE = 0.0                # perishable: wasted units recover nothing


def load():
    df = pd.read_parquet(FEAT, columns=["series_id", "item_id", "store_id", "state_id", "d", "date",
                                        "sales", "sell_price", "split", "snap", "event"])
    test = df[df.split == "test"].reset_index(drop=True)
    hist = df[df.split != "test"]
    # scale for scaled pinball loss: mean absolute 1-day change in the history (M5 convention)
    scale = hist.sort_values(["series_id", "d"]).groupby("series_id", observed=True).sales \
        .apply(lambda s: s.diff().abs().mean())
    vel = hist[hist.d > 1941 - 112].groupby("series_id", observed=True).sales.mean()
    test["scale"] = test.series_id.map(scale).astype(float).clip(lower=0.05)
    test["velocity"] = pd.cut(test.series_id.map(vel).astype(float), [-1, 0.5, 2, 1e9],
                              labels=["Slow (<0.5/day)", "Medium (0.5-2/day)", "Fast (>2/day)"])
    test["horizon_week"] = "Week " + ((test.d - 1942) // 7 + 1).astype(str)
    fc = {}
    for name, f in MODELS.items():
        x = pd.read_parquet(OUT / f"fc_{f}.parquet")
        assert (x.series_id.astype(str).values == test.series_id.astype(str).values).all()
        assert (x.d.values == test.d.values).all()
        fc[name] = x[QCOLS].to_numpy(float)
    return test, fc


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


# ------------------------------------------------------------------ accuracy
def accuracy(test, fc):
    y = test.sales.to_numpy()
    rows, calib = [], []
    for name, Q in fc.items():
        pl = np.column_stack([pinball(y, Q[:, j], q) for j, q in enumerate(QUANTILES)])
        spl = (pl / test.scale.to_numpy()[:, None]).mean()
        lo, hi = Q[:, 0], Q[:, QUANTILES.index(0.9)]
        rows.append({
            "Model": name,
            "MAE (median)": np.abs(y - Q[:, QUANTILES.index(0.5)]).mean(),
            "Mean pinball loss": pl.mean(),
            "Scaled pinball loss": spl,
            "In-stock rate at q90 (target 90%)": (y <= np.round(Q[:, QUANTILES.index(0.9)])).mean(),
            "In-stock rate at q95 (target 95%)": (y <= np.round(Q[:, QUANTILES.index(0.95)])).mean(),
        })
        for j, q in enumerate(QUANTILES):
            # discrete data: nominal level should sit between P(y<Q) and P(y<=Q); report the midpoint
            below, at_or_below = (y < Q[:, j]).mean(), (y <= Q[:, j]).mean()
            calib.append({"Model": name, "Nominal": q, "P(y<Q)": below, "P(y<=Q)": at_or_below,
                          "Coverage (mid)": 0.5 * (below + at_or_below), "Pinball": pl[:, j].mean()})
    return pd.DataFrame(rows), pd.DataFrame(calib)


# ------------------------------------------------------------------ newsvendor
def order_qty(Q, cr):
    return np.round(Q[:, QUANTILES.index(cr)])


def simulate(test, order):
    y, p = test.sales.to_numpy(), test.sell_price.to_numpy()
    sold = np.minimum(order, y)
    lost = np.maximum(y - order, 0)
    waste = np.maximum(order - y, 0)
    return y, p, sold, lost, waste


def newsvendor(test, fc):
    """Sweep the critical ratio. For each CR, the implied economics are cu = CR*p, co = (1-CR)*p,
    so 'cost' = price-weighted pinball loss. Physical outcomes (fill rate, waste) don't depend on $."""
    rows = []
    for name, Q in fc.items():
        for cr in CRS:
            y, p, sold, lost, waste = simulate(test, order_qty(Q, cr))
            order = sold + waste
            rows.append({
                "Model": name, "CR": cr,
                "Units ordered": order.sum(), "Units demanded": y.sum(),
                "Fill rate": sold.sum() / y.sum(),
                "Stockout item-days %": (lost > 0).mean(),
                "Waste % of ordered": waste.sum() / max(order.sum(), 1),
                "Waste units": waste.sum(), "Lost units": lost.sum(),
                "Cost at implied economics ($)": (cr * p * lost + (1 - cr) * p * waste).sum(),
                "Fresh-grocery cost ($)": (GROSS_MARGIN * p * lost + (1 - GROSS_MARGIN) * (1 - SALVAGE) * p * waste).sum(),
                "Lost margin ($)": (GROSS_MARGIN * p * lost).sum(),
                "Waste cost ($)": ((1 - GROSS_MARGIN) * (1 - SALVAGE) * p * waste).sum(),
            })
    nv = pd.DataFrame(rows)
    base = nv[nv.Model == list(MODELS)[0]].set_index("CR")["Cost at implied economics ($)"]
    nv["Cost vs baseline"] = nv["Cost at implied economics ($)"] / nv.CR.map(base) - 1
    return nv


def waste_at_fill_rate(nv, targets=(0.80, 0.85, 0.90)):
    """Interpolate along each model's CR sweep: how much waste is needed to hit a fill rate?"""
    rows = []
    for name, g in nv.groupby("Model", sort=False):
        g = g.sort_values("Fill rate")
        for t in targets:
            if g["Fill rate"].min() <= t <= g["Fill rate"].max():
                rows.append({"Model": name, "Target fill rate": t,
                             "Waste units": np.interp(t, g["Fill rate"], g["Waste units"]),
                             "Waste % of ordered": np.interp(t, g["Fill rate"], g["Waste % of ordered"])})
    out = pd.DataFrame(rows)
    b = out[out.Model == list(MODELS)[0]].set_index("Target fill rate")["Waste units"]
    out["Waste vs baseline"] = out["Waste units"] / out["Target fill rate"].map(b) - 1
    return out


def segment_costs(test, fc, by, cr=0.3):
    """Fresh-grocery cost per segment when each model orders at the cost-optimal CR."""
    rows = []
    for name, Q in fc.items():
        y, p, sold, lost, waste = simulate(test, order_qty(Q, cr))
        t = test[[by]].copy()
        t["cost"] = GROSS_MARGIN * p * lost + (1 - GROSS_MARGIN) * p * waste
        t["pl"] = np.column_stack([pinball(y, Q[:, j], q) for j, q in enumerate(QUANTILES)]).mean(axis=1)
        t["fill_num"], t["dem"] = sold, y
        g = t.groupby(by, observed=True).agg(cost=("cost", "sum"), pinball=("pl", "mean"),
                                             sold=("fill_num", "sum"), dem=("dem", "sum"), n=("cost", "size"))
        g["fill_rate"] = g.sold / g.dem
        g["Model"] = name
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True)


def store_impact(test, fc, cr):
    """Per-store, per-day outcomes when ordering at a given CR (for the write-up)."""
    rows = []
    for name, Q in fc.items():
        y, p, sold, lost, waste = simulate(test, order_qty(Q, cr))
        t = pd.DataFrame({"store_id": test.store_id, "d": test.d, "lost_units": lost, "waste_units": waste,
                          "stockout": (lost > 0).astype(int), "lost_margin": GROSS_MARGIN * p * lost,
                          "lost_revenue": p * lost, "waste_cost": (1 - GROSS_MARGIN) * p * waste})
        daily = t.groupby(["store_id", "d"], observed=True).sum(numeric_only=True)
        g = daily.groupby(level=0, observed=True).mean()
        g["Model"], g["CR"] = name, cr
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    test, fc = load()
    acc, cal = accuracy(test, fc)
    nv = newsvendor(test, fc)
    wf = waste_at_fill_rate(nv)
    seg_v = segment_costs(test, fc, "velocity")
    seg_h = segment_costs(test, fc, "horizon_week")
    imp = pd.concat([store_impact(test, fc, 0.3), store_impact(test, fc, 0.9)])
    for n, d in [("accuracy", acc), ("calibration", cal), ("newsvendor_sweep", nv), ("waste_at_fill_rate", wf),
                 ("segment_velocity", seg_v), ("segment_horizon", seg_h), ("store_impact", imp)]:
        d.to_csv(OUT / f"eval_{n}.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.float_format", lambda v: f"{v:,.3f}")
    print(acc.to_string(index=False), "\n")
    print(nv[["Model", "CR", "Fill rate", "Stockout item-days %", "Waste % of ordered", "Cost at implied economics ($)",
              "Cost vs baseline", "Fresh-grocery cost ($)"]].to_string(index=False), "\n")
    print(wf.to_string(index=False), "\n")
    print(seg_v.pivot(index="velocity", columns="Model", values="cost").round(0), "\n")
    print(seg_h.pivot(index="horizon_week", columns="Model", values="pinball").round(4))
