"""Check scripts/kronos_fast.py against Kronos' reference sampler.

Uses randomly initialised (small) Kronos models and greedy decoding (top_k=1),
so both implementations must produce the same tokens and outputs.

    KRONOS_PATH=/path/to/Kronos python tests/test_kronos_fast.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("KRONOS_PATH", os.path.join(ROOT, "Kronos")))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from model import Kronos, KronosTokenizer  # noqa: E402
from model.kronos import auto_regressive_inference  # noqa: E402
import kronos_fast  # noqa: E402


def stamps(ts):
    t = pd.DatetimeIndex(ts)
    return np.stack([t.minute, t.hour, t.weekday, t.day, t.month], -1).astype(np.float32)


def main():
    torch.manual_seed(0)
    tok = KronosTokenizer(d_in=6, d_model=64, n_heads=4, ff_dim=128, n_enc_layers=3, n_dec_layers=3,
                          ffn_dropout_p=0, attn_dropout_p=0, resid_dropout_p=0, s1_bits=5, s2_bits=5,
                          beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=5).eval()
    model = Kronos(s1_bits=5, s2_bits=5, n_layers=3, d_model=64, n_heads=4, ff_dim=128,
                   ffn_dropout_p=0, attn_dropout_p=0, resid_dropout_p=0, token_dropout_p=0,
                   learn_te=True).eval()
    # Randomly initialised heads are nearly flat; sharpen them so greedy
    # decoding is not decided by float noise.
    with torch.no_grad():
        model.head.proj_s1.weight.mul_(20)
        model.head.proj_s2.weight.mul_(20)

    B, H, max_ctx = 3, 5, 64
    for L in (max_ctx - H + 1, 40):
        ts = pd.date_range("2026-10-01", periods=L + H, freq="1min")
        st = stamps(ts)
        x = torch.randn(B, L, 6).clamp(-5, 5)
        xs = torch.from_numpy(np.repeat(st[None, :L], B, 0))
        ys = torch.from_numpy(np.repeat(st[None, L:], B, 0))

        ref = auto_regressive_inference(tok, model, x, xs, ys, max_ctx, H, clip=5,
                                        T=1.0, top_k=1, top_p=1.0, sample_count=1)[:, -H:, :]
        fast = kronos_fast.sample_paths(tok, model, x, xs, ys, H, samples=4, max_context=max_ctx,
                                        T=1.0, top_k=1, top_p=1.0).numpy()
        for s in range(4):
            err = np.abs(fast[:, s] - ref).max()
            print(f"L={L} sample {s}: max abs diff {err:.2e}")
            assert err < 1e-3, err
    print("OK")


if __name__ == "__main__":
    main()
