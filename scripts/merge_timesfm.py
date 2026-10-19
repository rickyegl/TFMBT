"""Merge sharded TimesFM prediction files into results/preds_<name>_<split>.npz."""

import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    shard_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(common.RESULTS, "timesfm_shards")
    groups = {}
    for f in sorted(glob.glob(os.path.join(shard_dir, "*.npz"))):
        m = re.match(r"(.+)__(validation|test)__(\d+)of(\d+)\.npz", os.path.basename(f))
        groups.setdefault((m.group(1), m.group(2), int(m.group(4))), []).append(f)
    for (name, split, n), files in groups.items():
        if len(files) != n:
            print(f"skip {name} {split}: {len(files)}/{n} shards")
            continue
        parts = [np.load(f) for f in files]
        idx = np.concatenate([p["origin"] for p in parts])
        o = np.argsort(idx)
        point = np.concatenate([p["point"] for p in parts])[o]
        quant = np.concatenate([p["quantiles"] for p in parts])[o]
        common.save_preds(name, split, idx[o], point, quant, meta=str(parts[0]["meta"]))
        print(f"{name} {split}: {len(idx)} origins")


if __name__ == "__main__":
    main()
