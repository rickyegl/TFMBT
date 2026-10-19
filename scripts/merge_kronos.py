"""Merge Kronos shard files into the common prediction format.

Writes, per (model, split):
  results/preds_kronos_<model>[_tag]_<split>.npz        point = sample mean
  results/preds_kronos_<model>[_tag]-median_<split>.npz point = sample median
  results/preds_kronos_<model>[_tag]-anchored_<split>.npz
        sample mean, re-anchored to the true last close: last + (mean - recon),
        where recon is Kronos' own tokenizer reconstruction of the last bar
        (removes the tokenizer's reconstruction offset; not part of stock Kronos)
  results/samples_kronos_<model>[_tag]_<split>.npz      all sampled closes
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default=os.path.join(common.RESULTS, "kronos_shards"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    close = common.load_bars()["close"].to_numpy()
    groups = {}
    for f in sorted(glob.glob(os.path.join(a.shard_dir, "kronos_*.npz"))):
        m = re.match(r"kronos_(\w+?)_(validation|test)_(\d+)of(\d+)\.npz", os.path.basename(f))
        groups.setdefault((m.group(1), m.group(2), int(m.group(4))), []).append(f)

    for (model, split, nshards), files in groups.items():
        if len(files) != nshards:
            print(f"skip {model} {split}: {len(files)}/{nshards} shards")
            continue
        parts = [np.load(f) for f in files]
        idx = np.concatenate([p["origin"] for p in parts])
        samples = np.concatenate([p["samples"] for p in parts]).astype(np.float64)
        order = np.argsort(idx)
        idx, samples = idx[order], samples[order]
        meta = str(parts[0]["meta"])
        recon = (np.concatenate([p["recon_last"] for p in parts])[order].astype(np.float64)
                 if "recon_last" in parts[0].files else None)

        qs = np.quantile(samples, common.QUANTILES, axis=1).transpose(1, 2, 0)  # (N,5,9)
        mean = samples.mean(1)
        median = np.median(samples, 1)
        quant = np.concatenate([mean[..., None], qs], axis=-1)                  # [mean, q10..q90]
        name = f"kronos_{model}" + (f"_{a.tag}" if a.tag else "")
        common.save_preds(name, split, idx, mean, quant, meta=meta)
        common.save_preds(name + "-median", split, idx, median, quant, meta=meta)
        if recon is not None:
            shift = (close[idx] - recon)[:, None]
            common.save_preds(name + "-anchored", split, idx, mean + shift, quant + shift[..., None], meta=meta)
        np.savez_compressed(os.path.join(common.RESULTS, f"samples_{name}_{split}.npz"),
                            origin=idx, samples=samples.astype(np.float32), meta=np.array(meta))
        print(f"{name} {split}: {len(idx)} origins, {samples.shape[1]} samples")


if __name__ == "__main__":
    main()
