"""Figures for the README (test split, models chosen on validation)."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import report  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FIG = os.path.join(common.ROOT, "figures")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def family(name):
    if name.startswith("timesfm"):
        return BLUE
    if name.startswith("kronos"):
        return ORANGE
    return AQUA


def pretty(name):
    n = name.replace("timesfm_", "TimesFM ").replace("kronos_", "Kronos-").replace("timesfm-", "TimesFM ")
    n = n.replace("returns_L", "returns ctx ").replace("price_L", "price ctx ").replace("_noflip", " no-flip")
    n = n.replace("-mean", " (mean)").replace("-anchored", " anchored").replace("-median", " (median)")
    n = n.replace("+shrink", " + shrink").replace("ens3", "ensemble of 3")
    if n.endswith(" + shrink") and name.startswith(("timesfm_price", "timesfm_returns", "kronos")):
        n = n.replace(" + shrink", ", shrunk to last price")
    return {"rw": "Random walk", "drift": "Drift (24h)", "linreg": "Linear regression"}.get(n, n)


def pinball(df, test, name, h):
    close = df["close"].to_numpy()
    idx, _, q = test[name]
    y, bps = close[idx + h], 1e4 / close[idx]
    return common.pinball(y * bps, q[:, h - 1, 1:] * bps[:, None])


def dot_plot(df, test, names, h, path, metric="rmse"):
    if metric == "rmse":
        vals = [(n, report.rel_rmse(df, test[n][0], test[n][1], h)) for n in names if n in test]
        xlabel = "RMSE relative to random walk\n(lower is better; below 1 beats \"price stays the same\")"
        title = f"Point forecast error, {h} min ahead"
    else:
        ref = pinball(df, test, "rw", h)
        vals = [(n, pinball(df, test, n, h) / ref) for n in names if n in test and test[n][2] is not None]
        xlabel = "Pinball loss over 10-90% quantiles, relative to random walk\nwith volatility-scaled bands (lower is better)"
        title = f"Uncertainty band quality, {h} min ahead"
    vals.sort(key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(vals) + 2.0))
    y = np.arange(len(vals))
    ax.axvline(1.0, color=MUTED, lw=1.2, ls="--", zorder=1)
    for yi, (n, v) in zip(y, vals):
        ax.hlines(yi, 1.0, v, color=family(n), lw=2, alpha=0.45, zorder=2)
        ax.scatter(v, yi, s=64, color=family(n), edgecolor=SURFACE, linewidth=2, zorder=3)
        right = v >= 1 - 1e-4
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(7 if right else -7, 0), textcoords="offset points",
                    va="center", ha="left" if right else "right", color=INK2, fontsize=9)
    labels = [pretty(n) for n, _ in vals]
    if metric != "rmse":
        labels = ["Random walk, volatility-scaled bands" if n == "rw" else l for (n, _), l in zip(vals, labels)]
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    lo = min(v for _, v in vals)
    hi = max(v for _, v in vals)
    pad = 0.07 * (hi - min(lo, 1.0)) + 0.012
    ax.set_xlim(min(lo, 1.0) - pad, hi + pad)
    ax.set_xlabel(xlabel)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, fontweight="bold")
    ax.set_title(f"BTC/USDT, test window 5-18 Oct 2026, {len(test['rw'][0])} forecasts",
                 loc="left", fontsize=9, fontweight="normal", color=INK2)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=8, label=l)
               for c, l in ((BLUE, "TimesFM 2.5"), (ORANGE, "Kronos"), (AQUA, "Simple baselines"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def horizon_plot(df, test, series, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    hs = np.arange(1, common.HORIZON + 1)
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.text(hs[0], 1.0015, "Random walk", va="bottom", color=INK2, fontsize=9)
    for name, color in series:
        if name not in test:
            continue
        v = [report.rel_rmse(df, test[name][0], test[name][1], h) for h in hs]
        ax.plot(hs, v, color=color, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, label=pretty(name))
    ax.set_xticks(hs, [f"{h} min" for h in hs])
    ax.set_xlim(0.8, hs[-1] + 0.2)
    ax.set_ylabel("RMSE relative to random walk")
    ax.set_title("Point forecast error by horizon (test, lower is better)", loc="left")
    ax.legend(frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def example_plot(df, test, panels, path, start="2026-10-14 13:00", hours=3):
    close = df["close"].to_numpy()
    ts = df["timestamp"].dt.tz_localize(None).to_numpy()
    t0 = np.datetime64(start)
    mask_bars = (ts >= t0) & (ts < t0 + np.timedelta64(hours * 60 + 5, "m"))
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.9 * len(panels)), sharex=True, sharey=True)
    for ax, (name, color) in zip(np.atleast_1d(axes), panels):
        idx, point, quant = test[name]
        sel = (ts[idx] >= t0) & (ts[idx] < t0 + np.timedelta64(hours * 60, "m"))
        tgt = ts[idx[sel] + common.HORIZON]
        ax.plot(ts[mask_bars], close[mask_bars], color=INK, lw=1.2, label="Actual price (1-min close)")
        if quant is not None:
            ax.vlines(tgt, quant[sel, -1, 1], quant[sel, -1, -1], color=color, alpha=0.35, lw=3,
                      label="80% interval")
        ax.scatter(tgt, point[sel, -1], s=22, color=color, edgecolor=SURFACE, linewidth=1, zorder=3,
                   label="Forecast made 5 min earlier")
        ax.set_title(pretty(name), loc="left", fontsize=11)
        ax.set_ylabel("USDT")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel(f"{str(t0)[:10]} (UTC)")
    handles = [plt.Line2D([], [], color=INK, lw=1.2, label="Actual price (1-min close)"),
               plt.Line2D([], [], color=MUTED, lw=3, alpha=0.6, label="80% interval"),
               plt.Line2D([], [], marker="o", ls="", color=MUTED, label="Forecast made 5 min earlier")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    df = common.load_bars()
    _, test, _, _ = report.build(df)
    key = [n for n in [
        "rw", "drift", "linreg",
        "timesfm_price_L2048", "timesfm_price_L16000", "timesfm_price_L2048_noflip",
        "timesfm_returns_L8192-mean", "timesfm_returns_L16000-mean", "timesfm-ens3",
        "timesfm_returns_L16000-mean+shrink",
        "kronos_mini", "kronos_small", "kronos_base",
        "kronos_mini-anchored", "kronos_small-anchored", "kronos_base-anchored",
    ] if n in test]
    for h in (1, common.HORIZON):
        dot_plot(df, test, key, h, os.path.join(FIG, f"rel_rmse_h{h}.png"))
    prob = [n for n in [
        "rw", "timesfm_price_L2048", "timesfm_returns_L16000", "timesfm_price_L16000",
        "timesfm_price_L2048+shrink", "timesfm_returns_L16000+shrink", "timesfm_price_L16000+shrink",
        "kronos_small", "kronos_base", "kronos_small-anchored+shrink", "kronos_base-anchored+shrink",
    ] if n in test]
    dot_plot(df, test, prob, common.HORIZON, os.path.join(FIG, "rel_pinball_h5.png"), metric="pinball")
    horizon_plot(df, test, [("timesfm_returns_L16000-mean", BLUE), ("timesfm_price_L2048", "#6da7ec"),
                            ("kronos_small-anchored", ORANGE)],
                 os.path.join(FIG, "rel_rmse_by_horizon.png"))
    example_plot(df, test, [("timesfm_price_L2048", BLUE), ("kronos_small", ORANGE)],
                 os.path.join(FIG, "example_window.png"))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
