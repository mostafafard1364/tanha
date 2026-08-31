"""
binance_feed.py — pulls live/rolling OHLCV candles from Binance's PUBLIC REST API.
No API key needed for market data (klines are public). Requires: pip install requests
"""
import time
import requests
import pandas as pd

BINANCE_BASE = "https://api.binance.com"

def fetch_klines(symbol="BTCUSDT", interval="15m", limit=1000, start_time_ms=None):
    """One page of up to 1000 candles. interval: 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d ..."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time_ms:
        params["startTime"] = start_time_ms
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw, columns=[
        "open_time","Open","High","Low","Close","Volume","close_time",
        "quote_volume","Trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    for c in ["Open","High","Low","Close","Volume","quote_volume"]:
        df[c] = df[c].astype(float)
    df["Trades"] = df["Trades"].astype(int)
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["total_volume"] = df["Volume"]
    return df[["dt","Open","High","Low","Close","total_volume","Trades"]]


class LiveCandleBuffer:
    """Keeps a rolling window of the base timeframe (15m) candles, refreshed on demand.
    Only CLOSED candles are used for signals — the currently-forming candle is dropped,
    to respect the 'no lookahead / closed bars only' rule from the original engine."""

    def __init__(self, symbol="BTCUSDT", base_interval="15m", max_bars=3000):
        self.symbol = symbol
        self.base_interval = base_interval
        self.max_bars = max_bars
        self.df = pd.DataFrame()

    def refresh(self):
        new = fetch_klines(self.symbol, self.base_interval, limit=1000)
        new = new.iloc[:-1]  # drop the still-forming (unclosed) last candle
        if self.df.empty:
            self.df = new
        else:
            self.df = pd.concat([self.df, new]).drop_duplicates(subset="dt").sort_values("dt")
        if len(self.df) > self.max_bars:
            self.df = self.df.iloc[-self.max_bars:]
        return self.df.reset_index(drop=True)


if __name__ == "__main__":
    buf = LiveCandleBuffer()
    df = buf.refresh()
    print(df.tail())
    print("rows:", len(df))
