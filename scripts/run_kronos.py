"""Rolling-origin backtest of Kronos on BTCUSDT 1-minute OHLCV.

Kronos is a sampling model: every forecast origin gets `--samples` independent
paths of the next 5 candles. All sampled closes are saved (N, S, 5) so the
mean, median and quantiles can be derived later. Supports sharding so the
work can be split across GitHub Actions jobs.

Sampling uses kronos_fast.sample_paths (shared-prefix KV cache), which matches
the reference sampler but needs lookback + 5 - 1 <= max_context, hence the
default lookback of 508 for the 512-token models.

Needs the Kronos repo on PYTHONPATH (--kronos-path) and Hugging Face access.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

MODELS = {
    "mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}
FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def time_features(ts: np.ndarray) -> np.ndarray:
    # Same features as kronos.calc_time_stamps.
    t = pd.DatetimeIndex(ts)
    return np.stack([t.minute, t.hour, t.weekday, t.day, t.month], axis=-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=list(common.SPLITS))
    ap.add_argument("--model", default="small", choices=list(MODELS))
    ap.add_argument("--lookback", type=int, default=0, help="default: max_context - 4")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=8, help="origins per forward pass")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kronos-path", default="Kronos")
    ap.add_argument("--out-dir", default=os.path.join(common.RESULTS, "kronos_shards"))
    a = ap.parse_args()

    import torch
    sys.path.insert(0, a.kronos_path)
    from model import Kronos, KronosTokenizer
    import kronos_fast

    torch.set_num_threads(os.cpu_count())
    torch.manual_seed(a.seed + a.shard)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id, tok_id, max_ctx = MODELS[a.model]
    tokenizer = KronosTokenizer.from_pretrained(tok_id).to(device).eval()
    model = Kronos.from_pretrained(model_id).to(device).eval()
    lookback = a.lookback or max_ctx - common.HORIZON + 1

    df = common.load_bars()
    feats = df[FEATURES].to_numpy(np.float32)
    ts = df["timestamp"].dt.tz_localize(None).to_numpy()
    stamps = time_features(ts)

    idx = common.origins(df, a.split)
    if a.limit:
        idx = idx[: a.limit]
    idx = np.array_split(idx, a.nshards)[a.shard]
    H, S = common.HORIZON, a.samples
    out = np.empty((len(idx), S, H), np.float32)

    t0 = time.time()
    for s in range(0, len(idx), a.batch):
        sub = idx[s: s + a.batch]
        win = np.stack([feats[i - lookback + 1: i + 1] for i in sub])          # (B, L, 6)
        mean, std = win.mean(1, keepdims=True), win.std(1, keepdims=True)
        x = np.clip((win - mean) / (std + 1e-5), -5, 5)
        xs = np.stack([stamps[i - lookback + 1: i + 1] for i in sub])
        ys = np.stack([stamps[i + 1: i + 1 + H] for i in sub])

        t = lambda z: torch.from_numpy(np.ascontiguousarray(z, np.float32)).to(device)  # noqa: E731
        z = kronos_fast.sample_paths(tokenizer, model, t(x), t(xs), t(ys), H, S, max_context=max_ctx,
                                     T=a.T, top_k=0, top_p=a.top_p)
        close = z[..., 3].cpu().numpy()                                       # (B, S, H)
        out[s: s + len(sub)] = close * (std[:, :, 3:4] + 1e-5) + mean[:, :, 3:4]
        done = s + len(sub)
        el = time.time() - t0
        print(f"{a.model} {a.split} shard {a.shard}: {done}/{len(idx)} "
              f"{el:.0f}s ({el / done:.2f}s/origin)", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"kronos_{a.model}_{a.split}_{a.shard:03d}of{a.nshards:03d}.npz")
    np.savez_compressed(path, origin=idx, samples=out, meta=np.array(repr(vars(a))))
    print(path)


if __name__ == "__main__":
    main()
