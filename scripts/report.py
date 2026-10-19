"""Score every saved prediction file and write summary tables.

Also fits, on VALIDATION only, a single shrinkage factor per model:
    p' = last + alpha * (p - last)
(alpha=0 is the random walk, alpha=1 the raw model) and applies it to TEST.
"""

import argparse
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


def available(split):
    names = []
    for f in sorted(glob.glob(os.path.join(common.RESULTS, f"preds_*_{split}.npz"))):
        names.append(re.match(rf"preds_(.+)_{split}\.npz", os.path.basename(f)).group(1))
    return names


def fit_alpha(close, idx, point, h):
    """Least-squares alpha for (y - last) ~ alpha * (p - last), in bps."""
    last = close[idx]
    x = (point[:, h - 1] - last) / last
    y = (close[idx + h] - last) / last
    ok = np.isfinite(x)
    return float(np.clip((x[ok] @ y[ok]) / (x[ok] @ x[ok]), 0, 1.5))


def shrink(close, idx, point, quant, alphas):
    last = close[idx][:, None]
    a = np.asarray(alphas)[None, :]
    p = last + a * (point - last)
    q = None if quant is None else quant + (p - point)[..., None]
    return p, q


def mc_corrected_rmse(df, name, split, idx, h):
    """Kronos only: remove the Monte-Carlo noise of a finite-sample mean."""
    f = os.path.join(common.RESULTS, f"samples_{name}_{split}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    assert np.array_equal(z["origin"], idx)
    s = z["samples"][:, :, h - 1].astype(np.float64)
    close = df["close"].to_numpy()
    last, y = close[idx], close[idx + h]
    mse = np.mean(((s.mean(1) - y) / last * 1e4) ** 2)
    noise = np.mean(s.var(1, ddof=1) / s.shape[1] / last ** 2 * 1e8)
    rw = np.mean(((last - y) / last * 1e4) ** 2)
    return float(np.sqrt((mse - noise) / rw))


def table(df, split, names, h, extra=None):
    rows = []
    for name in names:
        d = common.load_preds(name, split)
        idx, point, quant = d["origin"], d["point"], d.get("quantiles")
        if np.isnan(point[:, h - 1]).all():
            continue
        q = None if quant is None else quant[..., 1:]
        r = common.score(df, idx, point, q, h=h)
        r["model"] = name
        mc = mc_corrected_rmse(df, name, split, idx, h)
        r["relRMSE_MCcorr"] = mc if mc is not None else np.nan
        rows.append(r)
    for name, (idx, point, quant) in (extra or {}).items():
        r = common.score(df, idx, point, None if quant is None else quant[..., 1:], h=h)
        r["model"] = name
        rows.append(r)
    t = pd.DataFrame(rows).set_index("model")
    return t[[c for c in COLS + ["relRMSE_MCcorr"] if c in t]].sort_values("relRMSE_vs_RW")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=common.HORIZON)
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    df = common.load_bars()
    close = df["close"].to_numpy()

    val_names = available("validation")
    test_names = available("test")

    # Shrinkage fitted on validation, per horizon, for models present in both.
    extra = {"validation": {}, "test": {}}
    alphas = {}
    for name in val_names:
        if name in ("rw",):
            continue
        v = common.load_preds(name, "validation")
        al = [fit_alpha(close, v["origin"], v["point"], h) if not np.isnan(v["point"][:, h - 1]).all()
              else np.nan for h in range(1, common.HORIZON + 1)]
        alphas[name] = al
        for split in ("validation", "test"):
            if split == "test" and name not in test_names:
                continue
            d = v if split == "validation" else common.load_preds(name, "test")
            p, q = shrink(close, d["origin"], d["point"], d.get("quantiles"), al)
            extra[split][name + "+shrink"] = (d["origin"], p, q)

    os.makedirs(common.RESULTS, exist_ok=True)
    for split, names in (("validation", val_names), ("test", test_names)):
        if not names:
            continue
        t = table(df, split, names, a.h, extra[split])
        t.to_csv(os.path.join(common.RESULTS, f"summary_{split}_h{a.h}.csv"), float_format="%.6g")
        print(f"\n=== {split}  (h = {a.h} min) ===")
        print(t.round(4).to_string())
    print("\nshrinkage alphas (fit on validation, h=1..5):")
    for k, v in alphas.items():
        print(f"  {k:40s}", np.round(v, 3))


if __name__ == "__main__":
    main()
