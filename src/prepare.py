"""Step 1: prepare M5 FOODS_3 data in long format with features.

Scope: FOODS_3 department (823 items x 10 stores = 8,230 item-store series), the most
perishable-like slice of M5. Days are numbered d_1 (2011-01-29) ... d_1969 (2016-06-19).

Splits (forecast origin + 28-day horizon, as in the M5 competition):
  train : d_1400 .. d_1913
  valid : d_1914 .. d_1941   (early stopping, calibration checks)
  test  : d_1942 .. d_1969   (official M5 evaluation period, never touched while fitting)

All lag features use lags >= 28 days, so one "direct" model forecasts every day of the
28-day horizon from information available at the forecast origin (no recursion, no leakage).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
OUT = ROOT / "data" / "foods3_features.parquet"

DEPT = "FOODS_3"
FIRST_FEATURE_DAY = 1250          # history needed for the longest rolling window
TRAIN_START, TRAIN_END = 1400, 1913
VALID_END, TEST_END = 1941, 1969
HORIZON = 28


def load_wide() -> pd.DataFrame:
    tr = pd.read_csv(RAW / "sales_train_evaluation.csv")
    te = pd.read_csv(RAW / "sales_test_evaluation.csv")
    tr = tr[tr.dept_id == DEPT].reset_index(drop=True)
    te = te[te.dept_id == DEPT].reset_index(drop=True)
    keys = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    wide = tr.merge(te, on=keys, how="left")
    assert wide[[f"d_{d}" for d in range(1942, 1970)]].notna().all().all()
    return wide


def load_calendar() -> pd.DataFrame:
    cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    cal["d"] = np.arange(1, len(cal) + 1)
    cal["event"] = cal.event_type_1.fillna("none")
    cal["event_tomorrow"] = cal.event.shift(-1).fillna("none").ne("none").astype(int)
    return cal[["d", "date", "wm_yr_wk", "wday", "month", "event", "event_tomorrow",
                "snap_CA", "snap_TX", "snap_WI"]]


def build() -> pd.DataFrame:
    wide = load_wide()
    cal = load_calendar()
    days = [f"d_{d}" for d in range(FIRST_FEATURE_DAY, TEST_END + 1)]
    wide["series_id"] = wide.item_id + "_" + wide.store_id
    long = wide[["series_id", "item_id", "store_id", "state_id"] + days].melt(
        id_vars=["series_id", "item_id", "store_id", "state_id"], var_name="d", value_name="sales")
    long["d"] = long.d.str[2:].astype(np.int16)
    long["sales"] = long.sales.astype(np.float32)
    long = long.merge(cal, on="d", how="left")

    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[prices.item_id.str.startswith(DEPT)]
    g = prices.groupby(["store_id", "item_id"]).sell_price
    prices["price_rel_max"] = prices.sell_price / g.transform("max")
    prices["price_rel_mean"] = prices.sell_price / g.transform("mean")
    prices["price_chg_wk"] = g.pct_change().fillna(0)
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    long["snap"] = np.select([long.state_id == "CA", long.state_id == "TX"],
                             [long.snap_CA, long.snap_TX], long.snap_WI).astype(np.int8)
    long = long.drop(columns=["snap_CA", "snap_TX", "snap_WI", "wm_yr_wk"])
    long = long.sort_values(["series_id", "d"]).reset_index(drop=True)

    # lag / rolling features (all shifted by >= 28 days)
    g = long.groupby("series_id", sort=False).sales
    for L in (28, 35, 42, 49, 56):
        long[f"lag_{L}"] = g.shift(L).astype(np.float32)
    s28 = g.shift(28)
    gs = s28.groupby(long.series_id, sort=False)
    for w in (7, 28, 56, 112):
        long[f"rmean_{w}"] = gs.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean()).astype(np.float32)
    long["rstd_28"] = gs.transform(lambda x: x.rolling(28, min_periods=2).std()).astype(np.float32)
    long["zero_share_28"] = (s28 == 0).astype(float).groupby(long.series_id, sort=False) \
        .transform(lambda x: x.rolling(28, min_periods=1).mean()).astype(np.float32)
    # same-weekday average over the last 8 weeks (from lag 28 onward)
    long["dow_mean_8"] = long[[f"lag_{L}" for L in (28, 35, 42, 49, 56)]].mean(axis=1).astype(np.float32)

    long = long[long.d >= TRAIN_START]
    # an item is "on the shelf" only once it has a price; drop pre-launch rows
    long = long[long.sell_price.notna()].reset_index(drop=True)
    long["split"] = np.where(long.d <= TRAIN_END, "train", np.where(long.d <= VALID_END, "valid", "test"))
    for c in ("series_id", "item_id", "store_id", "state_id", "event", "split"):
        long[c] = long[c].astype("category")
    return long


if __name__ == "__main__":
    df = build()
    df.to_parquet(OUT, index=False)
    print(df.shape, df.split.value_counts().to_dict(), df.series_id.nunique(), "series")
    print(df.memory_usage(deep=True).sum() / 1e9, "GB")
