"""Shared data loading, forecast-origin selection and metrics.

Every model is scored on exactly the same forecast origins. At origin i the
model may see 1-minute bars 0..i (bar i has just closed) and must forecast the
close of bars i+1..i+5, i.e. the BTC price 1..5 minutes ahead. The headline
target is the 5-minute-ahead close.
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "btcusdt_1m.csv.gz")
RESULTS = os.path.join(ROOT, "results")
HORIZON = 5

# Hyper-parameters are chosen on VALIDATION only; TEST is touched once per
# final configuration.
SPLITS = {
    "validation": ("2026-09-21", "2026-10-05"),
    "test": ("2026-10-05", "2026-10-19"),
}
STRIDE = {"validation": 15, "test": 5}  # minutes between forecast origins

QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def load_bars() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def origins(df: pd.DataFrame, split: str) -> np.ndarray:
    """Row indices of forecast origins (last observed bar) for a split."""
    start, end = (pd.Timestamp(s, tz="UTC") for s in SPLITS[split])
    ts = df["timestamp"]
    idx = np.flatnonzero((ts >= start) & (ts < end)).astype(np.int64)
    idx = idx[idx + HORIZON < len(df)]
    # Keep only targets that are fully inside the split window.
    idx = idx[ts.values[idx + HORIZON] < np.datetime64(end.tz_localize(None))]
    return idx[:: STRIDE[split]]


def targets(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    close = df["close"].to_numpy()
    return close[idx[:, None] + np.arange(1, HORIZON + 1)[None, :]]


def save_preds(name: str, split: str, idx, point, quantiles=None, meta=None):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"preds_{name}_{split}.npz")
    kw = dict(origin=np.asarray(idx), point=np.asarray(point, dtype=np.float64))
    if quantiles is not None:
        kw["quantiles"] = np.asarray(quantiles, dtype=np.float64)
    if meta is not None:
        kw["meta"] = np.array(repr(meta))
    np.savez_compressed(path, **kw)
    return path


def load_preds(name: str, split: str) -> dict:
    with np.load(os.path.join(RESULTS, f"preds_{name}_{split}.npz")) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- metrics

def _newey_west_var(d: np.ndarray, lags: int) -> float:
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * (d[k:] @ d[:-k]) / n
    return v


def diebold_mariano(e_model: np.ndarray, e_base: np.ndarray, power: int = 2, lags: int = 0):
    """DM test; negative stat means the model has lower loss than the baseline."""
    from math import erf, sqrt
    d = np.abs(e_model) ** power - np.abs(e_base) ** power
    var = _newey_west_var(d, lags)
    if var <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(var / len(d))
    p = 2 * (1 - 0.5 * (1 + erf(abs(stat) / sqrt(2))))
    return float(stat), float(p)


def pinball(y, q_pred, qs=QUANTILES):
    diff = y[..., None] - q_pred
    return np.mean(np.maximum(qs * diff, (qs - 1) * diff))


def score(df, idx, point, quantiles=None, h=HORIZON):
    """Metrics for the h-step-ahead forecast (h in 1..HORIZON)."""
    close = df["close"].to_numpy()
    last = close[idx]
    y = close[idx + h]
    yhat = point[:, h - 1]
    err = yhat - y
    rw_err = last - y  # random walk / "no change" forecast

    bps = 1e4 / last
    true_move = y - last
    pred_move = yhat - last
    nz = (true_move != 0) & (pred_move != 0)
    out = {
        "n": len(idx),
        "MAE_usd": np.mean(np.abs(err)),
        "RMSE_usd": np.sqrt(np.mean(err ** 2)),
        "MAE_bps": np.mean(np.abs(err) * bps),
        "RMSE_bps": np.sqrt(np.mean((err * bps) ** 2)),
        "MAPE_pct": 100 * np.mean(np.abs(err) / y),
        "relMAE_vs_RW": np.mean(np.abs(err)) / np.mean(np.abs(rw_err)),
        "relRMSE_vs_RW": np.sqrt(np.mean(err ** 2) / np.mean(rw_err ** 2)),
        "dir_acc": np.mean(np.sign(pred_move[nz]) == np.sign(true_move[nz])) if nz.any() else np.nan,
        "dir_n": int(nz.sum()),
    }
    # Overlapping targets when stride < h -> HAC lags.
    lags = max(0, h - 1)
    out["DM_MSE_stat"], out["DM_MSE_p"] = diebold_mariano(err, rw_err, 2, lags)
    out["DM_MAE_stat"], out["DM_MAE_p"] = diebold_mariano(err, rw_err, 1, lags)
    if quantiles is not None:
        q = quantiles[:, h - 1, :]
        out["pinball_bps"] = pinball(y * bps, q * bps[:, None])
        out["cov80"] = np.mean((y >= q[:, 0]) & (y <= q[:, -1]))
        out["width80_bps"] = np.mean((q[:, -1] - q[:, 0]) * bps)
    return out
