"""Simple baselines scored on the same origins as the foundation models.

  rw      : random walk / no-change forecast (price stays at the last close);
            quantiles from a volatility-scaled empirical return distribution
  drift   : last close plus the mean 1-minute return of the past 24h
  linreg  : OLS of the h-minute forward log return on recent returns
            (1, 5, 15, 60, 240 min) and relative volume, fit on bars
            before the validation window only
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

VOL_WIN = 240


def features(df):
    logc = np.log(df["close"].to_numpy())
    vol = df["volume"].to_numpy()
    n = len(logc)
    cols = []
    for k in (1, 5, 15, 60, 240):
        r = np.full(n, np.nan)
        r[k:] = logc[k:] - logc[:-k]
        cols.append(r * 1e4)
    # Relative volume of the last 5 minutes vs. the last day.
    cs = np.concatenate([[0], np.cumsum(vol)])
    v5 = np.full(n, np.nan)
    v1440 = np.full(n, np.nan)
    # Mean volume of bars t-4..t (resp. t-1439..t), known at origin t.
    v5[5:] = (cs[6:] - cs[1:n - 4]) / 5
    v1440[1440:] = (cs[1441:] - cs[1:n - 1439]) / 1440
    cols.append(np.log(v5 / v1440))
    return np.stack(cols, axis=1)


def rolling_std_1m(df, win=VOL_WIN):
    r = np.diff(np.log(df["close"].to_numpy()), prepend=np.nan)
    s = np.full(len(r), np.nan)
    cs = np.nancumsum(r)
    cs2 = np.nancumsum(r ** 2)
    m = (cs[win:] - cs[:-win]) / win
    s[win:] = np.sqrt(np.maximum((cs2[win:] - cs2[:-win]) / win - m ** 2, 1e-12))
    return s


def main():
    df = common.load_bars()
    close = df["close"].to_numpy()
    logc = np.log(close)
    H = common.HORIZON
    hs = np.arange(1, H + 1)
    sig = rolling_std_1m(df)
    X = features(df)

    # Training rows: everything strictly before the validation window,
    # with targets that also end before it.
    val_start = common.origins(df, "validation")[0]
    train = np.arange(1500, val_start - H)
    Xtr = np.column_stack([np.ones(len(train)), X[train]])

    # Standardised forward-return quantiles from the training period.
    zq = {}
    betas = {}
    for h in hs:
        fwd = (logc[train + h] - logc[train]) * 1e4
        z = fwd / (sig[train] * 1e4 * np.sqrt(h))
        zq[h] = np.quantile(z[np.isfinite(z)], common.QUANTILES)
        ok = np.isfinite(Xtr).all(1)
        betas[h], *_ = np.linalg.lstsq(Xtr[ok], fwd[ok], rcond=None)
    print("linreg betas (h=5):", np.round(betas[H], 4))

    for split in common.SPLITS:
        idx = common.origins(df, split)
        last = close[idx]
        rw = np.repeat(last[:, None], H, axis=1)
        q = np.empty((len(idx), H, 10))
        for j, h in enumerate(hs):
            q[:, j, 1:] = last[:, None] * np.exp(zq[h][None, :] * sig[idx, None] * np.sqrt(h))
        q[:, :, 0] = rw
        common.save_preds("rw", split, idx, rw, q)

        mu = (logc[idx] - logc[idx - 1440]) / 1440
        drift = last[:, None] * np.exp(mu[:, None] * hs[None, :])
        common.save_preds("drift", split, idx, drift, q + (drift - rw)[..., None])

        Xi = np.column_stack([np.ones(len(idx)), X[idx]])
        lin = np.stack([last * np.exp(Xi @ betas[h] / 1e4) for h in hs], axis=1)
        common.save_preds("linreg", split, idx, lin, q + (lin - rw)[..., None])

        for name, p in [("rw", rw), ("drift", drift), ("linreg", lin)]:
            r = common.score(df, idx, p, q[..., 1:] + (p - rw)[..., None])
            print(split, name, {k: round(float(v), 4) for k, v in r.items()
                                if k in ("MAE_bps", "RMSE_bps", "relMAE_vs_RW", "relRMSE_vs_RW",
                                         "dir_acc", "pinball_bps", "cov80")})


if __name__ == "__main__":
    main()
