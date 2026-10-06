"""Step 5: build Tableau-ready extracts from the M5 forecasts and raw data.

Writes to tableau/ :
  m5_tableau_data.xlsx            all tables below as sheets (connect Tableau to this one file)
  newsvendor_store_day.csv        store x day x model x critical ratio: units demanded/sold/lost/wasted, $
  newsvendor_item_store.csv       item x store x model x critical ratio, summed over the 28 test days
  accuracy_store_week.csv         model x store x horizon week: pinball loss, MAE, in-stock at q90
  calibration.csv                 model x quantile level: share of actuals at or below the forecast
  dq_store_day.csv                store x day over the full history: items listed, zero-sales items, closures
  dq_series.csv                   item-store profile: listed days, zero share, longest zero run, demand class

Ratios (fill rate, waste %, zero-sales rate) are NOT stored; the guide builds them in Tableau as
SUM(numerator) / SUM(denominator), so they aggregate correctly at any level (store, state, total).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
OUT = ROOT / "output"
TAB = ROOT / "tableau"

QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
QCOLS = [f"q{int(q * 100)}" for q in QUANTILES]
CRS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
MODELS = {"baseline": "Baseline", "lightgbm": "LightGBM", "hierarchical": "Hierarchical Bayes"}
GROSS_MARGIN = 0.30
STATE_NAME = {"CA": "California", "TX": "Texas", "WI": "Wisconsin"}


def test_frame():
    df = pd.read_parquet(RAW / "foods3_features.parquet",
                         columns=["series_id", "item_id", "store_id", "state_id", "d", "date", "sales",
                                  "sell_price", "split"])
    hist = df[(df.split != "test") & (df.d > 1941 - 112)]
    vel = hist.groupby("series_id", observed=True).sales.mean()
    t = df[df.split == "test"].reset_index(drop=True)
    for c in ("series_id", "item_id", "store_id", "state_id"):
        t[c] = t[c].astype(str)
    v = t.series_id.map(vel.rename(index=str)).astype(float)
    t["velocity"] = np.select([v < 0.5, v <= 2], ["Slow (<0.5/day)", "Medium (0.5-2/day)"], "Fast (>2/day)")
    t["horizon_week"] = "Week " + ((t.d - 1942) // 7 + 1).astype(str)
    return t


def newsvendor_tables(t):
    y, p = t.sales.to_numpy(), t.sell_price.to_numpy()
    sd, isd = [], []
    for key, name in MODELS.items():
        f = pd.read_parquet(OUT / f"fc_{key}.parquet")
        assert (f.series_id.astype(str).values == t.series_id.values).all() and (f.d.values == t.d.values).all()
        Q = f[QCOLS].to_numpy(float)
        for cr in CRS:
            order = np.round(Q[:, QUANTILES.index(cr)])
            sold, lost, waste = np.minimum(order, y), np.maximum(y - order, 0), np.maximum(order - y, 0)
            x = t[["series_id", "item_id", "store_id", "state_id", "date", "d", "horizon_week", "velocity"]].copy()
            x["model"], x["critical_ratio"] = name, cr
            x["units_demanded"], x["units_ordered"], x["units_sold"] = y, order, sold
            x["units_lost"], x["units_wasted"] = lost, waste
            x["stockout_item_days"] = (lost > 0).astype(int)
            x["item_days"] = 1
            x["revenue_demanded"] = p * y
            x["revenue_sold"] = p * sold
            x["lost_revenue"] = p * lost
            x["lost_margin"] = GROSS_MARGIN * p * lost
            x["waste_cost"] = (1 - GROSS_MARGIN) * p * waste
            x["newsvendor_cost_implied"] = cr * p * lost + (1 - cr) * p * waste
            num = [c for c in x.columns if c.startswith(("units_", "stockout", "item_days", "revenue", "lost_",
                                                          "waste_", "newsvendor"))]
            sd.append(x.groupby(["store_id", "state_id", "date", "d", "horizon_week", "model", "critical_ratio"],
                                as_index=False)[num].sum())
            isd.append(x.groupby(["item_id", "store_id", "state_id", "velocity", "model", "critical_ratio"],
                                 as_index=False)[num].sum())
    sd, isd = pd.concat(sd, ignore_index=True), pd.concat(isd, ignore_index=True)
    for d in (sd, isd):
        d.insert(2, "state_name", d.state_id.map(STATE_NAME))
    sd["date"] = pd.to_datetime(sd.date).dt.date
    return sd, isd


def accuracy_tables(t):
    y = t.sales.to_numpy()
    acc, cal = [], []
    for key, name in MODELS.items():
        Q = pd.read_parquet(OUT / f"fc_{key}.parquet")[QCOLS].to_numpy(float)
        pl = np.column_stack([np.maximum(q * (y - Q[:, j]), (q - 1) * (y - Q[:, j])) for j, q in enumerate(QUANTILES)])
        x = t[["store_id", "state_id", "horizon_week"]].copy()
        x["pinball_sum"] = pl.mean(axis=1)
        x["abs_err_median_sum"] = np.abs(y - Q[:, QUANTILES.index(0.5)])
        x["in_stock_q90"] = (y <= np.round(Q[:, QUANTILES.index(0.9)])).astype(int)
        x["item_days"] = 1
        g = x.groupby(["store_id", "state_id", "horizon_week"], as_index=False).sum(numeric_only=True)
        g.insert(0, "model", name)
        acc.append(g)
        for j, q in enumerate(QUANTILES):
            below, at = (y < Q[:, j]).mean(), (y <= Q[:, j]).mean()
            cal.append({"model": name, "quantile_level": q, "share_below": below, "share_at_or_below": at,
                        "coverage_mid": 0.5 * (below + at), "pinball": pl[:, j].mean()})
    return pd.concat(acc, ignore_index=True), pd.DataFrame(cal)


def data_quality():
    """Full history (d_1..d_1969) for FOODS_3. 'Listed' = the item has a shelf price that week."""
    cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    cal["d"] = np.arange(1, len(cal) + 1)
    tr = pd.read_csv(RAW / "sales_train_evaluation.csv")
    te = pd.read_csv(RAW / "sales_test_evaluation.csv")
    keys = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    w = tr[tr.dept_id == "FOODS_3"].merge(te[te.dept_id == "FOODS_3"], on=keys)
    dcols = [f"d_{i}" for i in range(1, 1970)]
    S = w[dcols].to_numpy(np.float32)                          # series x day

    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[prices.item_id.str.startswith("FOODS_3")]
    listed_wk = prices.assign(listed=1).pivot_table(index=["store_id", "item_id"], columns="wm_yr_wk",
                                                    values="listed", fill_value=0)
    L = listed_wk.reindex(pd.MultiIndex.from_frame(w[["store_id", "item_id"]]), fill_value=0)
    L = L.reindex(columns=cal.wm_yr_wk.to_numpy(), fill_value=0).to_numpy(bool)   # series x day
    missing_cells = int(np.isnan(S).sum())

    # store x day
    rows = []
    for s in sorted(w.store_id.unique()):
        m = (w.store_id == s).to_numpy()
        Ss, Ls = S[m], L[m]
        units = Ss.sum(0)
        rows.append(pd.DataFrame({
            "store_id": s, "state_id": s[:2], "date": cal.date.dt.date, "d": cal.d,
            "year": cal.year, "month": cal.date.dt.strftime("%Y-%m"),
            "items_in_catalog": m.sum(), "items_listed": Ls.sum(0),
            "items_not_listed": m.sum() - Ls.sum(0),
            "listed_items_zero_sales": ((Ss == 0) & Ls).sum(0),
            "items_with_sales": (Ss > 0).sum(0),
            "sales_while_unlisted": ((Ss > 0) & ~Ls).sum(0),
            "units_sold": units,
        }))
    dq = pd.concat(rows, ignore_index=True)
    dq["store_closed"] = (dq.units_sold == 0).astype(int)
    med = dq.groupby("store_id").units_sold.transform(lambda v: v.rolling(28, center=True, min_periods=7).median())
    dq["low_sales_anomaly"] = ((dq.units_sold < 0.5 * med) & (dq.store_closed == 0)).astype(int)
    dq["is_christmas"] = pd.to_datetime(dq.date).dt.strftime("%m-%d").eq("12-25").astype(int)
    dq["missing_cells"] = 0
    dq.insert(2, "state_name", dq.state_id.map(STATE_NAME))

    # series profile (listed days only)
    prof = []
    open_days = ~(S.sum(0) == 0)        # exclude days when the department sold nothing anywhere (Christmas)
    for i in range(len(w)):
        lst = L[i] & open_days
        y = S[i][lst]
        n = len(y)
        z = (y == 0)
        run = best = 0
        for v in z:
            run = run + 1 if v else 0
            best = max(best, run)
        nz = y[y > 0]
        if len(nz) >= 2:
            adi = n / len(nz)
            cv2 = (nz.std() / nz.mean()) ** 2
            cls = ("Smooth" if cv2 < 0.49 else "Erratic") if adi < 1.32 else ("Intermittent" if cv2 < 0.49 else "Lumpy")
        else:
            adi, cv2, cls = np.nan, np.nan, "Too sparse"
        first_listed = cal.date.iloc[int(np.argmax(L[i]))].date() if L[i].any() else None
        prof.append({"series_id": f"{w.item_id[i]}_{w.store_id[i]}", "item_id": w.item_id[i],
                     "store_id": w.store_id[i], "state_id": w.state_id[i],
                     "first_listed_date": first_listed, "days_listed": int(n),
                     "days_not_listed": int(open_days.sum() - n), "zero_sales_days": int(z.sum()),
                     "zero_share": float(z.mean()) if n else np.nan,
                     "longest_zero_run_days": int(best), "suspected_stockout_28d": int(best >= 28),
                     "avg_units_per_listed_day": float(y.mean()) if n else 0.0,
                     "adi": adi, "cv2": cv2, "demand_class": cls})
    prof = pd.DataFrame(prof)
    prof.insert(4, "state_name", prof.state_id.map(STATE_NAME))
    meta = {"calendar_days": len(cal), "calendar_gaps": int((cal.date.diff().dropna() != pd.Timedelta("1D")).sum()),
            "missing_cells": missing_cells, "series": len(w)}
    return dq, prof, meta


if __name__ == "__main__":
    TAB.mkdir(exist_ok=True)
    t = test_frame()
    sd, isd = newsvendor_tables(t)
    acc, cal = accuracy_tables(t)
    dq, prof, meta = data_quality()
    tables = {"newsvendor_store_day": sd, "newsvendor_item_store": isd, "accuracy_store_week": acc,
              "calibration": cal, "dq_store_day": dq, "dq_series": prof}
    for n, d in tables.items():
        d.to_csv(TAB / f"{n}.csv", index=False, float_format="%.6g")
    with pd.ExcelWriter(TAB / "m5_tableau_data.xlsx", engine="openpyxl") as xw:
        for n, d in tables.items():
            d.to_excel(xw, sheet_name=n, index=False, float_format="%.6g")
    for n, d in tables.items():
        print(f"{n:24s} {d.shape}")
    print(meta)
