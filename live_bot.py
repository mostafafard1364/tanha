"""
live_bot.py — main loop. Polls Binance, updates the analysis engine, and
(only if DRY_RUN=False) sends orders through the Iran exchange adapter.

Run:  DRY_RUN=1 python3 live_bot.py        # safe: prints signals only
      DRY_RUN=0 python3 live_bot.py        # LIVE: places real orders. Be sure.
"""
import os
import time
import traceback
import pandas as pd

from binance_feed import LiveCandleBuffer
from engine import latest_zone, LivePosition
from iran_exchange_adapter import IranExchangeAdapter

SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"          # which cluster-detection timeframe to trade off of
POLL_SECONDS = 60         # how often to check for new closed candles
POSITION_SIZE = 0.0       # <-- set your real position size before going live

DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"


def main():
    print(f"Starting live engine | symbol={SYMBOL} tf={TIMEFRAME} dry_run={DRY_RUN}")
    buf = LiveCandleBuffer(symbol=SYMBOL, base_interval="15m")
    exch = IranExchangeAdapter(headless=True, dry_run=DRY_RUN)
    if not DRY_RUN:
        exch.login()

    live_pos = None
    last_zone_end = None

    while True:
        try:
            candles = buf.refresh()
            if len(candles) < 800:
                print("warming up history buffer..."); time.sleep(POLL_SECONDS); continue

            zone = latest_zone(candles, TIMEFRAME)
            if zone and zone['zone_end'] != last_zone_end and live_pos is None:
                last_zone_end = zone['zone_end']
                live_pos = LivePosition(zone['zone_high'], zone['zone_low'], zone['risk'])
                print(f"[{pd.Timestamp.utcnow()}] new zone resolved -> starting LONG plan "
                      f"at {live_pos.entry:.2f}, stop {live_pos.stop:.2f}")
                exch.place_market_order("buy", SYMBOL, POSITION_SIZE)

            if live_pos is not None:
                last_row = candles.iloc[-1]
                action = live_pos.update(last_row.High, last_row.Low, last_row.Close)
                if action == 'EXIT_AND_FLIP_SHORT':
                    print(f"[{last_row.dt}] fake breakout -> flip SHORT at {live_pos.entry:.2f}")
                    exch.place_market_order("sell", SYMBOL, POSITION_SIZE * 2)  # close long + open short
                elif action == 'EXIT_AND_FLIP_LONG':
                    print(f"[{last_row.dt}] fake breakout -> flip LONG at {live_pos.entry:.2f}")
                    exch.place_market_order("buy", SYMBOL, POSITION_SIZE * 2)
                elif action == 'EXIT_FINAL':
                    print(f"[{last_row.dt}] real breakout profit-take -> flat")
                    exch.place_market_order("sell" if live_pos.pos_dir == 'UP' else "buy",
                                             SYMBOL, POSITION_SIZE)
                    live_pos = None

        except Exception as e:
            print("ERROR:", e)
            traceback.print_exc()

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
