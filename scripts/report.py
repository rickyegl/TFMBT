"""Score every saved prediction file and write summary tables.

Derived forecasts (all choices made on VALIDATION, then applied to TEST):
  <timesfm>-mean   TimesFM's mean output channel instead of the median
  timesfm-ens3     average of the 3 TimesFM variants with the best validation RMSE
  <model>+shrink   p' = last + alpha_h * (p - last), alpha_h fit by least squares
                   on validation (alpha=0 is the random walk, 1 the raw model).
                   Quantiles move with the point forecast, so for alpha=0 this
                   is "random-walk centre + model uncertainty".
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

COLS = ["n", "MAE_usd", "RMSE_usd", "MAE_bps", "RMSE_bps", "relMAE_vs_RW", "relRMSE_vs_RW",
        "DM_MSE_stat", "DM_MSE_p", "dir_acc", "dir_n", "pinball_bps", "cov80", "width80_bps"]
H = common.HORIZON


def load_split(split):
    """name -> (origin, point (N,5), quantiles (N,5,10) or None)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(common.RESULTS, f"preds_*_{split}.npz"))):
        name = re.match(rf"preds_(.+)_{split}\.npz", os.path.basename(f)).group(1)
        d = common.load_preds(name, split)
        q = d.get("quantiles")
        out[name] = (d["origin"], d["point"], q)
        if name.startswith("timesfm") and q is not None:
            out[name + "-mean"] = (d["origin"], q[..., 0], q)
    return out


def rel_rmse(df, idx, point, h=H):
    return common.score(df, idx, point, h=h)["relRMSE_vs_RW"]


def fit_alpha(close, idx, point, h):
    """Least-squares alpha for (y - last) ~ alpha * (p - last), in returns."""
    last = close[idx]
    x = (point[:, h - 1] - last) / last
    y = (close[idx + h] - last) / last
    if not np.isfinite(x).any():
        return np.nan
    return float(np.clip((x @ y) / (x @ x), 0, 1.5))


def shrink(close, idx, point, quant, alphas):
    last = close[idx][:, None]
    p = last + np.asarray(alphas)[None, :] * (point - last)
    q = None if quant is None else quant + (p - point)[..., None]
    return p, q


def mc_corrected_rmse(df, name, split, idx, h=H):
    """Kronos only: relRMSE of the infinite-sample mean (removes Monte-Carlo noise)."""
    base = name.split("+")[0]
    f = os.path.join(common.RESULTS, f"samples_{base}_{split}.npz")
    if base.endswith("-median") or "+" in name or not os.path.exists(f):
        return np.nan
    z = np.load(f)
    if not np.array_equal(z["origin"], idx):
        return np.nan
    s = z["samples"][:, :, h - 1].astype(np.float64)
    close = df["close"].to_numpy()
    last, y = close[idx], close[idx + h]
    mse = np.mean(((s.mean(1) - y) / last) ** 2)
    noise = np.mean(s.var(1, ddof=1) / s.shape[1] / last ** 2)
    rw = np.mean(((last - y) / last) ** 2)
    return float(np.sqrt((mse - noise) / rw))


def build(df):
    close = df["close"].to_numpy()
    val, test = load_split("validation"), load_split("test")

    # Ensemble of the best 3 TimesFM variants on validation (that also ran on test).
    tfm = [n for n in val if n.startswith("timesfm") and not np.isnan(val[n][1][:, H - 1]).all()]
    ranked = sorted(tfm, key=lambda n: rel_rmse(df, val[n][0], val[n][1]))
    members = [n for n in ranked if n in test][:3]
    if members:
        for split, d in (("validation", val), ("test", test)):
            idx = d[members[0]][0]
            p = np.mean([d[m][1] for m in members], axis=0)
            q = np.mean([d[m][2] for m in members], axis=0)
            d["timesfm-ens3"] = (idx, p, q)

    alphas = {}
    for name in list(val):
        if name == "rw":
            continue
        idx, p, q = val[name]
        al = [fit_alpha(close, idx, p, h) for h in range(1, H + 1)]
        alphas[name] = al
        val[name + "+shrink"] = (idx, *shrink(close, idx, p, q, al))
        if name in test:
            tidx, tp, tq = test[name]
            test[name + "+shrink"] = (tidx, *shrink(close, tidx, tp, tq, al))
    return val, test, members, alphas


def table(df, split, preds, h=H):
    rows = []
    for name, (idx, point, quant) in preds.items():
        if np.isnan(point[:, h - 1]).all():
            continue
        r = common.score(df, idx, point, None if quant is None else quant[..., 1:], h=h)
        r["model"] = name
        r["relRMSE_MCcorr"] = mc_corrected_rmse(df, name, split, idx, h)
        rows.append(r)
    t = pd.DataFrame(rows).set_index("model")
    return t[COLS + ["relRMSE_MCcorr"]].sort_values("relRMSE_vs_RW")


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)
    df = common.load_bars()
    val, test, members, alphas = build(df)

    for split, preds in (("validation", val), ("test", test)):
        if not preds:
            continue
        for h in (1, H):
            t = table(df, split, preds, h)
            t.to_csv(os.path.join(common.RESULTS, f"summary_{split}_h{h}.csv"), float_format="%.6g")
        print(f"\n=== {split}  (h = {H} min) ===")
        print(t.round(4).to_string())

    # Per-horizon relative RMSE on test.
    rows = {}
    for name, (idx, point, _) in test.items():
        rows[name] = [rel_rmse(df, idx, point, h) if not np.isnan(point[:, h - 1]).all() else np.nan
                      for h in range(1, H + 1)]
    hz = pd.DataFrame(rows, index=[f"h{h}" for h in range(1, H + 1)]).T.sort_values(f"h{H}")
    hz.to_csv(os.path.join(common.RESULTS, "test_relRMSE_by_horizon.csv"), float_format="%.6g")
    print("\n=== test relRMSE vs RW by horizon ===")
    print(hz.round(4).to_string())

    print("\nensemble members (chosen on validation):", members)
    pd.DataFrame(alphas, index=[f"h{h}" for h in range(1, H + 1)]).T.to_csv(
        os.path.join(common.RESULTS, "shrink_alphas.csv"), float_format="%.4g")
    print("shrinkage alphas (fit on validation, h=1..5):")
    for k, v in alphas.items():
        print(f"  {k:40s}", np.round(v, 3))


if __name__ == "__main__":
    main()
