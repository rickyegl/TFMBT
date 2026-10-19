"""Rolling-origin backtest of TimesFM 2.5 (200M, Flax) on BTCUSDT.

Input modes
  price    : last `context` 1-minute closes -> closes 1..5 minutes ahead
  returns  : last `context` 1-minute log returns (bps) -> next 5 returns,
             cumulated back into prices
  price5m  : last `context` closes sampled every 5 minutes (grid ending at the
             origin) -> one step = the close 5 minutes ahead (h=5 only)
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

CKPT = os.path.join(common.ROOT, "checkpoints", "timesfm-2.5-200m-flax")


def build_inputs(close, idx, mode, context):
    if mode == "price":
        return [close[max(0, i - context + 1): i + 1] for i in idx]
    if mode == "returns":
        logc = np.log(close)
        return [np.diff(logc[max(0, i - context): i + 1]) * 1e4 for i in idx]
    if mode == "price5m":
        return [close[i - 5 * (context - 1): i + 1: 5] if i >= 5 * (context - 1)
                else close[i % 5: i + 1: 5] for i in idx]
    raise ValueError(mode)


def to_prices(close, idx, mode, point, quant):
    """Map model outputs back to price space. Returns (N,5) point, (N,5,10) quantiles."""
    last = close[idx]
    n = len(idx)
    if mode == "price":
        return point[:, :5], quant[:, :5, :]
    if mode == "price5m":
        p = np.full((n, 5), np.nan)
        q = np.full((n, 5, quant.shape[-1]), np.nan)
        p[:, 4] = point[:, 0]
        q[:, 4, :] = quant[:, 0, :]
        return p, q
    if mode == "returns":
        med = quant[:, :5, 5]
        cum_med = np.cumsum(med, axis=1)
        cum_mean = np.cumsum(quant[:, :5, 0], axis=1)
        # Quantiles of the cumulative return, assuming independent steps
        # with the model's marginal spreads (approximation).
        spread = quant[:, :5, 1:] - med[..., None]
        cum_spread = np.sign(spread) * np.sqrt(np.cumsum(spread ** 2, axis=1))
        q = np.concatenate([cum_mean[..., None], cum_med[..., None] + cum_spread], axis=-1)
        p = last[:, None] * np.exp(cum_med / 1e4)
        q = last[:, None, None] * np.exp(q / 1e4)
        return p, q
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=list(common.SPLITS))
    ap.add_argument("--mode", default="price", choices=["price", "returns", "price5m"])
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--no-flip", action="store_true", help="disable force_flip_invariance")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N origins")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    import timesfm

    name = a.name or f"timesfm_{a.mode}_L{a.context}" + ("_noflip" if a.no_flip else "")
    df = common.load_bars()
    close = df["close"].to_numpy()
    idx = common.origins(df, a.split)
    if a.limit:
        idx = idx[: a.limit]

    model = timesfm.TimesFM_2p5_200M_flax()
    model.load_checkpoint(CKPT)
    model.compile(timesfm.ForecastConfig(
        max_context=a.context,
        max_horizon=128,
        per_core_batch_size=a.batch,
        normalize_inputs=not a.no_normalize,
        force_flip_invariance=not a.no_flip,
        infer_is_positive=a.mode != "returns",
        fix_quantile_crossing=True,
    ))

    h = 1 if a.mode == "price5m" else 5
    t0 = time.time()
    points, quants = [], []
    chunk = a.batch * 8
    for s in range(0, len(idx), chunk):
        sub = idx[s: s + chunk]
        p, q = model.forecast(h, build_inputs(close, sub, a.mode, a.context))
        points.append(p)
        quants.append(q)
        print(f"{name} {a.split}: {s + len(sub)}/{len(idx)} ({time.time() - t0:.0f}s)", flush=True)
    point, quant = to_prices(close, idx, a.mode, np.concatenate(points), np.concatenate(quants))

    path = common.save_preds(name, a.split, idx, point, quant, meta=vars(a))
    r = common.score(df, idx, point, quant[..., 1:])
    print(path)
    print({k: round(float(v), 4) for k, v in r.items()})


if __name__ == "__main__":
    main()
