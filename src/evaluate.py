"""
evaluate.py: produces every number reported in the paper, in one run.

Design
  * Rolling-origin (expanding-window) evaluation. For each target month from
    January 2022 to December 2025 (48 months), every model is refit on all data
    before that month (from January 2020; 2019 supplies lags) and predicts that
    month for all ten utilities: 480 utility-month forecasts per model.
  * Forecast-time information only. Lag-1 and lag-2 use are treated as available at
    the start of the target month; calendar month and utility are known in advance.
    No temperature is used in the primary models.
  * Hyperparameters are chosen using only data before the evaluation period: fit on
    2020, score on 2021, using a 54-setting grid. (The overall design was settled
    after earlier analyses of 2024-2025 had been seen; see the paper's Methods.)
  * Comparison methods: seasonal naive (same month last year), monthly average
    (utility's mean for that calendar month over the fitting data), and a linear
    regression on the same inputs (month as indicators).
  * Tests: for each comparison, the monthly loss differential is the comparison
    method's MAE minus gradient boosting's MAE across the ten utilities (positive
    favors gradient boosting), giving 48 values. Diebold-Mariano test with the
    Harvey-Leybourne-Newbold small-sample correction (one-step forecasts), and a
    95% circular block-bootstrap interval (block length 3; 2 and 4 as sensitivity).
  * Ablations (same protocol): without lag-1/lag-2; separate model per utility.
  * Temperature is not an input: its forecast-time form (calendar-month climatology)
    is a fixed value per utility and month, which the model already encodes.

Outputs: results/final/*.csv plus results/final/provenance.json
"""
import itertools, json, platform, subprocess, sys
import numpy as np
import pandas as pd
import sklearn, scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from common import PROC, RESULTS, RANDOM_STATE

OUT = RESULTS / "final"
LAGS = ["lag1", "lag2", "lag12"]
GRID = dict(n_estimators=[200, 300, 500], learning_rate=[0.03, 0.05, 0.1],
            max_depth=[2, 3, 4], subsample=[0.8, 1.0])
EVAL_START, EVAL_END = "2022-01-01", "2025-12-01"
N_BOOT = 10000


def load():
    p = pd.read_csv(PROC / "panel.csv", parse_dates=["date"])
    p = p[p.year >= 2020].dropna(subset=["demand_mg"] + LAGS).reset_index(drop=True)
    return p.sort_values(["date", "utility"]).reset_index(drop=True)


def design(fit, target, lags=LAGS, temp=False):
    """Scale by fitting-data utility means; build feature frames for fit and target rows."""
    scale = fit.groupby("utility").demand_mg.mean()
    def frame(d):
        s = d.utility.map(scale)
        X = pd.DataFrame({f"{c}_idx": d[c] / s for c in lags}, index=d.index)
        X["month"] = d.month
        if temp:
            X["temp_F"] = d.temp_F
        for u in sorted(scale.index):
            X[f"u_{u}"] = (d.utility == u).astype(float)
        return X, d.demand_mg / s, s
    return frame(fit), frame(target)


def gbm(params):
    return GradientBoostingRegressor(random_state=RANDOM_STATE, **params)


def lin_X(X):
    """Linear-regression design. Month enters as indicators for February to December
    (January is the reference) and the first utility indicator is dropped, so the
    design is full rank alongside the intercept. With all categories included the
    least-squares problem is rank-deficient and its solution can differ across
    numerical libraries."""
    m = pd.get_dummies(X.month, prefix="m", dtype=float).reindex(
        columns=[f"m_{i}" for i in range(2, 13)], fill_value=0.0)
    u_cols = sorted(c for c in X.columns if c.startswith("u_"))
    return pd.concat([X.drop(columns=["month", u_cols[0]]), m], axis=1)


def tune(p, variant="pooled", lags=LAGS):
    """Choose hyperparameters on 2021 (fit on 2020), before any evaluation month.
    Each model variant is tuned separately."""
    fit, val = p[p.year == 2020], p[p.year == 2021]
    rows, best = [], None
    for vals in itertools.product(*GRID.values()):
        prm = dict(zip(GRID.keys(), vals))
        if variant == "separate":
            errs = []
            for u, gv in val.groupby("utility"):
                (Xa, ya, _), (Xb, yb, _) = design(fit[fit.utility == u], gv, lags=lags)
                cols = [c for c in Xa.columns if not c.startswith("u_")]
                errs.append(np.abs(gbm(prm).fit(Xa[cols], ya).predict(Xb[cols]) - yb))
            mae = np.mean(np.concatenate(errs))
        else:
            (Xf, yf, _), (Xv, yv, _) = design(fit, val, lags=lags)
            mae = np.mean(np.abs(gbm(prm).fit(Xf, yf).predict(Xv) - yv))
        rows.append({**prm, "val_mae_index_2021": mae})
        if best is None or mae < best[1]:
            best = (prm, mae)
    pd.DataFrame(rows).sort_values("val_mae_index_2021").to_csv(OUT / f"grid_search_{variant}.csv", index=False)
    return best[0]


def rolling(p, params, params_nolag, params_sep):
    months = pd.date_range(EVAL_START, EVAL_END, freq="MS")
    out = []
    for t in months:
        fit, tgt = p[p.date < t], p[p.date == t]
        (Xf, yf, _), (Xt, _, st) = design(fit, tgt)
        r = tgt[["utility", "date", "year", "month", "demand_mg", "lag12"]].copy()
        r["gbm"] = gbm(params).fit(Xf, yf).predict(Xt) * st
        r["linear"] = LinearRegression().fit(lin_X(Xf), yf).predict(lin_X(Xt)) * st
        r["seasonal_naive"] = tgt.lag12
        clim = fit.groupby(["utility", "month"]).demand_mg.mean()
        r["monthly_mean"] = [clim.get((u, m), np.nan) for u, m in zip(tgt.utility, tgt.month)]
        # history-length experiment: same baseline, but only the 24 months before t
        recent = fit[fit.date >= t - pd.DateOffset(months=24)]
        clim2 = recent.groupby(["utility", "month"]).demand_mg.mean()
        r["monthly_mean_2yr"] = [clim2.get((u, m), np.nan) for u, m in zip(tgt.utility, tgt.month)]
        # ablation 1: no recent lags
        (Xf2, yf2, _), (Xt2, _, st2) = design(fit, tgt, lags=["lag12"])
        r["gbm_no_recent_lags"] = gbm(params_nolag).fit(Xf2, yf2).predict(Xt2) * st2
        # ablation 2: separate model per utility
        sep = []
        for u, gu in tgt.groupby("utility"):
            fu = fit[fit.utility == u]
            (Xa, ya, _), (Xb, _, sb) = design(fu, gu)
            cols = [c for c in Xa.columns if not c.startswith("u_")]
            sep.append(pd.Series(gbm(params_sep).fit(Xa[cols], ya).predict(Xb[cols]) * sb, index=gu.index))
        r["gbm_separate"] = pd.concat(sep)
        # MASE scale: utility's mean absolute seasonal difference in the fitting data
        r["mase_scale"] = tgt.utility.map(fit.assign(e=(fit.demand_mg - fit.lag12).abs())
                                          .groupby("utility").e.mean())
        out.append(r)
        print(f"  {t:%Y-%m} done", end="\r")
    print()
    return pd.concat(out, ignore_index=True)


METHODS = ["gbm", "linear", "monthly_mean", "seasonal_naive", "gbm_no_recent_lags", "gbm_separate",
           "monthly_mean_2yr"]


def summarize(f):
    rows = []
    for period, g in [("2022-2025", f)] + [(str(y), gy) for y, gy in f.groupby("year")]:
        for mth in METHODS:
            e = (g[mth] - g.demand_mg).abs()
            rows.append(dict(period=period, method=mth, mae_mg=e.mean(),
                             mase=(e / g.mase_scale).mean(),
                             peak_mae_mg=e[g.month.between(6, 9)].mean(), n=len(g)))
    return pd.DataFrame(rows)


HAC_BANDWIDTH = 3   # standard rule floor(4 * (T/100)**(2/9)) = 3 for T = 48; chosen during revision


def dm_test(d, bandwidth=HAC_BANDWIDTH, h=1):
    """Diebold-Mariano test on loss differential d with Harvey-Leybourne-Newbold
    small-sample factor. Long-run variance uses Newey-West (Bartlett) weights up to
    `bandwidth` lags, so serial dependence in d is allowed for; bandwidth=0 gives
    the no-dependence version, reported as a sensitivity check."""
    d = np.asarray(d, float); T = len(d); dbar = d.mean()
    gam = lambda k: np.sum((d[k:] - dbar) * (d[:T - k] - dbar)) / T
    lrv = gam(0) + 2 * sum((1 - k / (bandwidth + 1)) * gam(k) for k in range(1, bandwidth + 1))
    stat = dbar / np.sqrt(lrv / T) * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return stat, 2 * stats.t.sf(abs(stat), df=T - 1)


def ljung_box(d, lags=3):
    """Ljung-Box test for autocorrelation in the loss differential."""
    d = np.asarray(d, float) - np.mean(d); T = len(d); den = np.sum(d * d)
    q = T * (T + 2) * sum((np.sum(d[k:] * d[:T - k]) / den) ** 2 / (T - k) for k in range(1, lags + 1))
    acf = "; ".join(f"{float(np.sum(d[k:] * d[:T - k]) / den):.3f}" for k in range(1, lags + 1))
    return acf, float(stats.chi2.sf(q, lags))


def block_ci(d, L, rng):
    d = np.asarray(d, float); T = len(d); nb = int(np.ceil(T / L))
    starts = rng.integers(0, T, size=(N_BOOT, nb))
    idx = (starts[:, :, None] + np.arange(L)) % T
    bm = d[idx.reshape(N_BOOT, -1)[:, :T]].mean(axis=1)
    return np.percentile(bm, [2.5, 97.5])


def tests(f, rng):
    rows = []
    monthly = f.groupby("date")
    for comp in ["monthly_mean", "seasonal_naive", "linear", "gbm_no_recent_lags", "gbm_separate"]:
        d = monthly.apply(lambda g: (g[comp] - g.demand_mg).abs().mean()
                                    - (g.gbm - g.demand_mg).abs().mean()).values
        stat, p = dm_test(d)
        _, p0 = dm_test(d, bandwidth=0)
        _, p6 = dm_test(d, bandwidth=6)
        acf, lb_p = ljung_box(d)
        row = dict(comparison=f"{comp} minus gbm", primary=(comp == "monthly_mean"),
                   mean_diff_mg=d.mean(), dm_stat=stat, p_hac3=p, p_lag0=p0, p_hac6=p6,
                   acf_lags1to3=acf, ljung_box_p=lb_p,
                   months_gbm_better=int((d > 0).sum()), months=len(d))
        for L in (2, 3, 4):
            lo, hi = block_ci(d, L, rng)
            row[f"ci95_L{L}"] = f"{lo:.1f} to {hi:.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def by_utility(f):
    g = f.groupby("utility")
    t = pd.DataFrame({m: g.apply(lambda x: (x[m] - x.demand_mg).abs().mean()) for m in METHODS[:4]})
    y25 = f[f.year == 2025].groupby("utility")
    t["gbm_annual_err_2025_pct"] = (y25.gbm.sum() - y25.demand_mg.sum()) / y25.demand_mg.sum() * 100
    t["monthly_mean_annual_err_2025_pct"] = (y25.monthly_mean.sum() - y25.demand_mg.sum()) / y25.demand_mg.sum() * 100
    return t


def warming(p, params):
    q = p.dropna(subset=["temp_F"])
    fit, tgt = q[q.year.between(2020, 2024)], q[q.year == 2025]
    (Xf, yf, _), (Xt, _, st) = design(fit, tgt, temp=True)
    m = gbm(params).fit(Xf, yf)
    base = m.predict(Xt) * st
    Xw = Xt.copy(); Xw["temp_F"] += 3
    warm = m.predict(Xw) * st
    r = pd.DataFrame({"utility": tgt.utility, "base": base, "warm": warm})
    s = r.groupby("utility")[["base", "warm"]].sum()
    s["change_pct"] = (s.warm - s.base) / s.base * 100
    return s


def period_tests(f, rng):
    """Main comparison by period, plus the exploratory history-length experiment:
    in 2024-2025, a monthly average limited to the previous 24 months (the history
    available in 2022) versus gradient boosting, at the same forecast origins."""
    rows = []
    for lab, sub, base in [("2022-2023", f[f.year <= 2023], "monthly_mean"),
                           ("2024-2025", f[f.year >= 2024], "monthly_mean"),
                           ("2024-2025, 2-year average (exploratory)", f[f.year >= 2024], "monthly_mean_2yr")]:
        d = sub.groupby("date").apply(lambda g: (g[base] - g.demand_mg).abs().mean()
                                                - (g.gbm - g.demand_mg).abs().mean()).values
        stat, pv = dm_test(d); lo, hi = block_ci(d, 3, rng)
        rows.append(dict(period=lab, baseline=base, mean_diff_mg=d.mean(), ci95_L3=f"{lo:.1f} to {hi:.1f}",
                         p_hac3=pv, months=len(d)))
    return pd.DataFrame(rows)


def figures(f):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for y, g in f.groupby("year"):
        ax.scatter(g.demand_mg, g.gbm, s=12, alpha=0.7, label=str(y))
    hi = max(f.demand_mg.max(), f.gbm.max()) * 1.05
    ax.plot([0, hi], [0, hi], "--", color="gray", lw=1)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("Observed monthly use (MG)"); ax.set_ylabel("Gradient boosting forecast (MG)")
    ax.legend(frameon=False, fontsize=8, title="Year")
    fig.tight_layout(); fig.savefig(OUT / "fig1_observed_vs_forecast.png"); plt.close(fig)

    g25 = f[f.year == 2025]; us = sorted(g25.utility.unique())
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.8), squeeze=False)
    for ax, u in zip(axes.flat, us):
        g = g25[g25.utility == u].sort_values("month")
        ax.plot(g.month, g.demand_mg, "o-", color="black", lw=1.2, ms=3, label="Observed")
        ax.plot(g.month, g.gbm, "s--", color="#0072B2", lw=1.2, ms=3, label="Gradient boosting")
        ax.plot(g.month, g.monthly_mean, "^:", color="#D55E00", lw=1.2, ms=3, label="Monthly average")
        ax.set_title(u, fontsize=9); ax.set_xticks([1, 4, 7, 10])
    for ax in list(axes.flat)[len(us):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.supxlabel("Month (2025)"); fig.supylabel("MG")
    fig.tight_layout(); fig.savefig(OUT / "fig2_2025_by_utility.png"); plt.close(fig)


def provenance(params, params_nolag=None, params_sep=None):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    return dict(git_commit=commit or "not a git repository", python=platform.python_version(),
                numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                random_state=RANDOM_STATE, params=params, params_no_recent_lags=params_nolag,
                params_separate=params_sep, hac_bandwidth=HAC_BANDWIDTH,
                input_sha256={f.name: __import__("hashlib").sha256(f.read_bytes()).hexdigest()
                              for f in [PROC / "panel.csv"] + sorted((PROC.parent / "raw").glob("var_*.csv"))},
                eval_months=f"{EVAL_START[:7]} to {EVAL_END[:7]}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = load()
    params = tune(p)
    params_nolag = tune(p, "no_recent_lags", lags=["lag12"])
    params_sep = tune(p, "separate")
    print("Frozen parameters (chosen on 2021):", params, params_nolag, params_sep, sep="\n  ")
    f = rolling(p, params, params_nolag, params_sep)
    f.to_csv(OUT / "forecasts.csv", index=False)
    s = summarize(f); s.to_csv(OUT / "accuracy.csv", index=False)
    rng = np.random.default_rng(RANDOM_STATE)
    t = tests(f, rng); t.to_csv(OUT / "tests.csv", index=False)
    u = by_utility(f); u.to_csv(OUT / "by_utility.csv")
    pt = period_tests(f, rng); pt.to_csv(OUT / "period_tests.csv", index=False)
    figures(f)
    json.dump(provenance(params, params_nolag, params_sep), open(OUT / "provenance.json", "w"), indent=2)
    pd.set_option("display.width", 160)
    print("\nAccuracy:\n" + s.round(3).to_string(index=False))
    print("\nTests (positive = gradient boosting better):\n" + t.round(3).to_string(index=False))
    print("\nBy utility, MAE 2022-2025 (MG):\n" + u.round(1).to_string())
    print("\nMonthly average minus gradient boosting, by period:\n" + pt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
