"""Download BTCUSDT 1-minute klines from Binance's public data archive.

Uses the S3 bucket behind data.binance.vision (monthly archives for complete
months, daily archives for the current month) and writes one CSV with UTC
timestamps.
"""

import argparse
import io
import os
import urllib.request
import zipfile

import pandas as pd

BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def _get_zip_csv(key: str) -> pd.DataFrame:
    with urllib.request.urlopen(f"{BUCKET}/{key}", timeout=120) as r:
        raw = r.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, names=COLS)
    # Some archives ship a header row.
    if not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:]
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default="2026-07-01")
    p.add_argument("--end", default="2026-10-18", help="inclusive, UTC day")
    p.add_argument("--out", default="data/btcusdt_1m.csv.gz")
    a = p.parse_args()

    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    frames = []
    day = start
    while day <= end:
        month_end = day + pd.offsets.MonthEnd(0)
        if day.day == 1 and month_end <= end:
            key = f"data/spot/monthly/klines/{a.symbol}/1m/{a.symbol}-1m-{day:%Y-%m}.zip"
            print("monthly", key)
            frames.append(_get_zip_csv(key))
            day = month_end + pd.Timedelta(days=1)
        else:
            key = f"data/spot/daily/klines/{a.symbol}/1m/{a.symbol}-1m-{day:%Y-%m-%d}.zip"
            print("daily  ", key)
            frames.append(_get_zip_csv(key))
            day += pd.Timedelta(days=1)

    df = pd.concat(frames, ignore_index=True)
    ot = df["open_time"].astype("int64")
    # Binance switched spot archives from ms to us timestamps in 2025.
    unit = "us" if ot.max() > 10**14 else "ms"
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(ot, unit=unit, utc=True),
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
        "amount": df["quote_volume"].astype(float),
    })
    out = out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    full = pd.date_range(out.timestamp.iloc[0], out.timestamp.iloc[-1], freq="1min")
    missing = len(full) - len(out)
    print(f"{len(out)} rows, {out.timestamp.iloc[0]} -> {out.timestamp.iloc[-1]}, {missing} missing minutes")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False, compression="gzip")


if __name__ == "__main__":
    main()
