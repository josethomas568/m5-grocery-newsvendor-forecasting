"""Step 2: three probabilistic forecasts for the 28-day test horizon (d_1942..d_1969).

Every model outputs the same quantile grid for every item-store-day, so they can be
compared on identical terms and fed into the same newsvendor ordering rule.

  1. Baseline      - empirical quantiles of each series' last 56 days of sales
  2. LightGBM      - one global quantile-regression model per quantile level
  3. Hierarchical  - Bayesian Gamma-Poisson model: item-store demand rates are partially
                     pooled toward an item x store prior, with negative-binomial daily noise
                     and weekday / SNAP multipliers; hyperparameters by empirical Bayes
"""
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "foods3_features.parquet"
OUT = ROOT / "output"

QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
QCOLS = [f"q{int(q * 100)}" for q in QUANTILES]
ORIGIN = 1941                      # last day of known data at forecast time
SEED = 7

FEATURES = ["item_id", "store_id", "wday", "month", "event", "event_tomorrow", "snap",
            "sell_price", "price_rel_max", "price_rel_mean", "price_chg_wk",
            "lag_28", "lag_35", "lag_42", "lag_49", "lag_56",
            "rmean_7", "rmean_28", "rmean_56", "rmean_112", "rstd_28", "zero_share_28", "dow_mean_8"]
CATS = ["item_id", "store_id", "event"]


def load():
    df = pd.read_parquet(FEAT)
    test = df[df.split == "test"][["series_id", "item_id", "store_id", "state_id", "d", "date",
                                   "wday", "snap", "sell_price", "sales"]].reset_index(drop=True)
    return df, test


def _sort_quantiles(Q: np.ndarray) -> np.ndarray:
    """Quantile models are fitted independently and can cross; enforce monotonicity."""
    return np.maximum(np.sort(Q, axis=1), 0)


# ------------------------------------------------------------------ 1. baseline
def baseline(df, test, window=56):
    hist = df[(df.d > ORIGIN - window) & (df.d <= ORIGIN)]
    qs = hist.groupby("series_id", observed=True).sales.quantile(QUANTILES, interpolation="higher").unstack()
    qs.columns = QCOLS
    out = test[["series_id", "d"]].merge(qs, left_on="series_id", right_index=True, how="left")
    out[QCOLS] = _sort_quantiles(out[QCOLS].to_numpy())
    return out


# ------------------------------------------------------------------ 2. LightGBM
def lightgbm_quantiles(df, test, refit=True, log=print):
    tr = df[df.split == "train"]
    va = df[df.split == "valid"]
    te = df[df.split == "test"]
    params = dict(learning_rate=0.08, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.6, bagging_freq=1, lambda_l2=1.0, max_bin=127,
                  verbose=-1, seed=SEED, num_threads=2)
    dtr = lgb.Dataset(tr[FEATURES], tr.sales, categorical_feature=CATS, free_raw_data=False)
    dva = lgb.Dataset(va[FEATURES], va.sales, categorical_feature=CATS, reference=dtr)
    full = df[df.split != "test"]
    dfull = lgb.Dataset(full[FEATURES], full.sales, categorical_feature=CATS) if refit else None
    preds, info = {}, []
    for q, col in zip(QUANTILES, QCOLS):
        t0 = time.time()
        p = dict(params, objective="quantile", alpha=q)
        m = lgb.train(p, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        n = m.best_iteration
        if refit:   # refit on train+valid with the early-stopped round count
            m = lgb.train(p, dfull, num_boost_round=n)
        preds[col] = m.predict(te[FEATURES], num_iteration=None if refit else n)
        info.append({"quantile": q, "rounds": n, "seconds": round(time.time() - t0, 1)})
        log(f"  LightGBM q={q}: {n} rounds, {time.time() - t0:.0f}s")
        if q == 0.5:
            imp = pd.Series(m.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
            imp.to_csv(OUT / "lgbm_feature_importance_q50.csv", header=["gain"])
    out = te[["series_id", "d"]].reset_index(drop=True).copy()
    out[QCOLS] = _sort_quantiles(np.column_stack([preds[c] for c in QCOLS]))
    pd.DataFrame(info).to_csv(OUT / "lgbm_training_log.csv", index=False)
    return out


# ------------------------------------------------------------------ 3. hierarchical Bayes
def hierarchical_bayes(df, test, window=112, half_life=56, log=print):
    """Gamma-Poisson hierarchy fitted on the last `window` days with recency weights.

    y_ist | lam_is ~ NegBin(mean = lam_is * x_st, size = k_is)      daily demand
    lam_is         ~ Gamma(shape = a, rate = a / m_is)                item-store rate
    m_is           = r_i * f_s                                        item level x store level
    x_st           = weekday factor (store) * SNAP factor (state)     known multipliers
    a (pooling strength) is chosen by maximising the marginal likelihood (empirical Bayes).
    """
    h = df[(df.d > ORIGIN - window) & (df.d <= ORIGIN)].copy()
    h["w"] = 0.5 ** ((ORIGIN - h.d) / half_life)

    # known multipliers: weekday x store, SNAP x state (pooled over all items)
    lvl = h.groupby(["store_id", "series_id"], observed=True).sales.transform("mean") + 1e-9
    h["rel"] = h.sales / lvl
    dow = h.groupby(["store_id", "wday"], observed=True).rel.mean()
    dow = dow / dow.groupby(level=0, observed=True).transform("mean")
    h = h.merge(dow.rename("f_dow").reset_index(), on=["store_id", "wday"])
    h["rel2"] = h.rel / h.f_dow
    snap = h.groupby(["state_id", "snap"], observed=True).rel2.mean().unstack()
    snap_eff = (snap[1] / snap[0]).rename("f_snap1")
    h = h.merge(snap_eff.reset_index(), on="state_id")
    h["x"] = h.f_dow * np.where(h.snap == 1, h.f_snap1, 1.0)

    # sufficient statistics per series (recency-weighted)
    h["wy"], h["wx"] = h.w * h.sales, h.w * h.x
    s = h.groupby(["series_id", "item_id", "store_id"], observed=True)[["wy", "wx"]].sum().reset_index()
    s.columns = ["series_id", "item_id", "store_id", "Y", "X"]

    # prior mean m_is = item rate x store level (each pooled across the other dimension)
    f_store = s.groupby("store_id", observed=True).Y.sum() / s.groupby("store_id", observed=True).X.sum()
    f_store = f_store / f_store.mean()
    s["f_s"] = s.store_id.map(f_store).astype(float)
    s["fX"] = s.f_s * s.X
    item = s.groupby("item_id", observed=True)[["Y", "fX"]].sum()
    s["r_i"] = s.item_id.map((item.Y + 0.5) / (item.fX + 1.0)).astype(float)
    s["m"] = s.r_i * s.f_s

    # empirical Bayes: choose Gamma shape a maximising the NegBin marginal likelihood of Y
    Y, X, M = s.Y.to_numpy(), s.X.to_numpy(), s.m.to_numpy()

    def neg_marginal(log_a):
        a = np.exp(log_a)
        mu = M * X
        ll = (special.gammaln(Y + a) - special.gammaln(a) - special.gammaln(Y + 1)
              + a * np.log(a / (a + mu)) + Y * np.log(mu / (a + mu) + 1e-300))
        return -ll.sum()

    res = optimize.minimize_scalar(neg_marginal, bounds=(np.log(0.05), np.log(500)), method="bounded")
    a = float(np.exp(res.x))
    s["post_shape"] = a + Y
    s["post_rate"] = a / M + X
    s["lam_mean"] = s.post_shape / s.post_rate
    s["lam_var"] = s.post_shape / s.post_rate ** 2
    s["shrink_wt"] = X / (X + a / M)          # weight on the item-store's own data

    # daily over-dispersion (1/k) per series by moments, shrunk toward the store median
    h = h.merge(s[["series_id", "lam_mean"]], on="series_id")
    h["mu"] = h.lam_mean * h.x
    h["ex"] = (h.sales - h.mu) ** 2 - h.mu
    d = h.groupby("series_id", observed=True).agg(ex=("ex", "sum"), mu2=("mu", lambda v: (v ** 2).sum()),
                                                   n=("sales", "size"), store_id=("store_id", "first"))
    d["inv_k_raw"] = (d.ex / d.mu2.clip(lower=1e-6)).clip(lower=0, upper=20)
    prior_ik = d.groupby("store_id", observed=True).inv_k_raw.transform("median")
    wt = d.n / (d.n + 56)
    d["inv_k"] = wt * d.inv_k_raw + (1 - wt) * prior_ik
    s = s.merge(d[["inv_k"]], left_on="series_id", right_index=True)

    # posterior predictive for each test day: moment-matched negative binomial
    t = test[["series_id", "store_id", "state_id", "d", "wday", "snap"]].merge(
        dow.rename("f_dow").reset_index(), on=["store_id", "wday"], how="left").merge(
        snap_eff.reset_index(), on="state_id", how="left").merge(
        s[["series_id", "lam_mean", "lam_var", "inv_k"]], on="series_id", how="left")
    x = t.f_dow * np.where(t.snap == 1, t.f_snap1, 1.0)
    mean = t.lam_mean * x
    var = mean + x ** 2 * ((t.lam_mean ** 2 + t.lam_var) * t.inv_k + t.lam_var)
    mean, var = mean.to_numpy(), np.maximum(var.to_numpy(), mean.to_numpy() * (1 + 1e-6))
    n_nb = mean ** 2 / (var - mean)
    p_nb = mean / var
    Q = np.column_stack([stats.nbinom.ppf(q, n_nb, p_nb) for q in QUANTILES])
    out = test[["series_id", "d"]].copy()
    out[QCOLS] = _sort_quantiles(np.nan_to_num(Q))
    out["mean"] = mean

    s.to_csv(OUT / "hier_posterior_by_series.csv", index=False)
    meta = {"gamma_shape_a": a, "median_shrink_weight": float(s.shrink_wt.median()),
            "median_inv_k": float(s.inv_k.median()),
            "snap_effect": snap_eff.round(3).to_dict(), "dow_factors": dow.round(3).unstack().to_dict()}
    log(f"  Hierarchical: a={a:.2f}, median weight on own data={meta['median_shrink_weight']:.2f}")
    return out, meta


if __name__ == "__main__":
    import json
    import sys
    OUT.mkdir(exist_ok=True)
    df, test = load()
    which = sys.argv[1:] or ["baseline", "hier", "lgbm"]
    if "baseline" in which:
        t0 = time.time(); b = baseline(df, test); b.to_parquet(OUT / "fc_baseline.parquet")
        print(f"baseline done {time.time() - t0:.0f}s")
    if "hier" in which:
        t0 = time.time(); hb, meta = hierarchical_bayes(df, test); hb.to_parquet(OUT / "fc_hierarchical.parquet")
        (OUT / "hier_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"hierarchical done {time.time() - t0:.0f}s")
    if "lgbm" in which:
        t0 = time.time(); lg = lightgbm_quantiles(df, test); lg.to_parquet(OUT / "fc_lightgbm.parquet")
        print(f"lightgbm done {time.time() - t0:.0f}s")
