"""
engine.py — the analysis core (color-bars + intensity + loose-cluster range
detection + smart-flip trade logic), reused as-is from the validated backtest.
This module is pure analysis: given historical closed candles, it tells you
the CURRENT open position/plan, if any. It never touches money.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

W = 720
GAP_TOL = 2
MIN_WEAK = 4
TRAIL_MULT = 0.15
OUTSIDE_STOP_MULT = 0.25


def build_engine_frame(base_15m: pd.DataFrame, freq: str) -> pd.DataFrame:
    base = base_15m.set_index('dt').copy()
    base['body'] = (base.Close - base.Open).abs()
    z = base.resample(freq, label='right', closed='right').agg(
        Open=('Open', 'first'), High=('High', 'max'), Low=('Low', 'min'),
        Close=('Close', 'last'), total_volume=('total_volume', 'sum'),
        Trades=('Trades', 'sum'), system_bar=('body', 'sum')
    ).dropna().reset_index()
    z['chart_bar'] = z.High - z.Low
    z['net_movement'] = z.Close - z.Open

    def addq(col):
        s = z[col].shift(1)
        for q in (.2, .4, .6, .8):
            z[f'{col}_q{int(q*100)}'] = s.rolling(W, min_periods=100).quantile(q)
    addq('chart_bar'); addq('system_bar')

    def lvl(v, p):
        a = z[v].to_numpy()
        q = z[[f'{p}_q{x}' for x in (20, 40, 60, 80)]].to_numpy()
        r = np.full(len(z), 'UNKNOWN', object)
        ok = np.isfinite(a) & np.all(np.isfinite(q), axis=1)
        r[ok] = np.array(['VERY_WEAK','WEAK','NORMAL','STRONG','VERY_STRONG'])[
            np.sum(a[ok, None] >= q[ok], axis=1)]
        return r
    z['chart_level5'] = lvl('chart_bar', 'chart_bar')
    z['system_level5'] = lvl('system_bar', 'system_bar')

    colors = []
    for a, b, n in zip(z.chart_level5, z.system_level5, z.net_movement):
        if 'UNKNOWN' in (a, b): x = 'INVALID'
        elif a == 'VERY_WEAK' and b == 'VERY_WEAK': x = 'YELLOW'
        elif a == 'NORMAL' and b == 'NORMAL': x = 'GREEN'
        elif a == 'VERY_STRONG' and b == 'VERY_STRONG': x = 'BLUE' if n > 0 else ('RED' if n < 0 else 'PINK')
        elif b in ('STRONG', 'VERY_STRONG') and a in ('VERY_WEAK', 'WEAK'): x = 'WHITE'
        elif a in ('STRONG', 'VERY_STRONG') and b in ('VERY_WEAK', 'WEAK'): x = 'CYAN'
        elif a == 'NORMAL' and b in ('STRONG', 'VERY_STRONG'): x = 'ORANGE'
        elif a in ('VERY_WEAK', 'WEAK') and b == 'NORMAL': x = 'PURPLE'
        elif a == 'NORMAL' and b in ('VERY_WEAK', 'WEAK'): x = 'GRAY'
        else: x = 'PINK'
        colors.append(x)
    z['color_code'] = colors

    dt_h = z.dt.diff().dt.total_seconds().div(3600)
    z['time_delta_hours'] = dt_h.fillna(dt_h.iloc[1] if len(z) > 1 else 1)
    z['volume_time'] = z.total_volume * z.time_delta_hours
    z['abs_price_change'] = z.net_movement.abs()
    z['intensity_avg_causal'] = (z.abs_price_change.where(z.volume_time > 0, 0).cumsum().shift(1) /
                                  z.volume_time.where(z.volume_time > 0, 0).cumsum().shift(1))
    z['bar_intensity'] = z.abs_price_change / z.volume_time.replace(0, np.nan)
    z['intensity_ratio'] = z.bar_intensity / z.intensity_avg_causal.replace(0, np.nan)
    for q in (.2, .4, .6, .8):
        z[f'ir_q{int(q*100)}'] = z.intensity_ratio.shift(1).rolling(W, min_periods=100).quantile(q)
    z['pressure_level5'] = lvl('intensity_ratio', 'ir')
    return z.drop(columns=[c for c in z.columns if '_q' in c])


def find_loose_clusters(color: np.ndarray):
    is_weak = np.isin(color, ['YELLOW', 'WHITE'])
    n = len(color); clusters, i = [], 0
    while i < n:
        if is_weak[i]:
            start = i; weak_count = 1; last_weak = i; j = i + 1
            while j < n:
                if is_weak[j]: weak_count += 1; last_weak = j; j += 1
                else:
                    gap = 1; k = j
                    while k < n and not is_weak[k] and gap <= GAP_TOL: k += 1; gap += 1
                    if k < n and is_weak[k] and gap <= GAP_TOL: j = k
                    else: break
            end = last_weak
            if weak_count >= MIN_WEAK: clusters.append((start, end))
            i = end + 1
        else: i += 1
    return clusters


class LivePosition:
    """Holds the state of the ONE currently-live plan for a given (symbol, timeframe)."""
    def __init__(self, zone_high, zone_low, risk):
        self.pos_dir = 'UP'
        self.entry = zone_low
        self.stop = self.entry - risk * OUTSIDE_STOP_MULT
        self.far_edge = zone_high
        self.phase = 'approach'
        self.extreme = self.entry
        self.risk = risk
        self.zone_low = zone_low
        self.zone_high = zone_high
        self.closed = False
        self.flips = 0

    def update(self, high, low, close):
        """Feed the latest CLOSED candle's H/L/C. Returns an action string:
        None | 'ENTER_LONG' | 'ENTER_SHORT' | 'EXIT_AND_FLIP_SHORT' | 'EXIT_AND_FLIP_LONG' | 'EXIT_FINAL'"""
        risk = self.risk
        if self.pos_dir == 'UP':
            if self.phase == 'approach':
                if low <= self.stop:
                    self.pos_dir, self.entry, self.extreme = 'DOWN', self.stop, self.stop
                    self.stop = self.entry + risk * OUTSIDE_STOP_MULT
                    self.phase = 'trailing'; self.flips += 1
                    return 'EXIT_AND_FLIP_SHORT'
                if high >= self.far_edge:
                    self.extreme = high; self.phase = 'trailing'
                    self.stop = self.extreme - risk * TRAIL_MULT
                return None
            else:
                self.extreme = max(self.extreme, high)
                ns = self.extreme - risk * TRAIL_MULT
                if ns > self.stop: self.stop = ns
                if low <= self.stop:
                    if self.stop < self.far_edge:
                        exit_price = self.stop
                        self.pos_dir, self.entry, self.extreme = 'DOWN', exit_price, exit_price
                        self.far_edge = self.zone_low
                        self.stop = self.entry + risk * OUTSIDE_STOP_MULT
                        self.phase = 'approach'; self.flips += 1
                        return 'EXIT_AND_FLIP_SHORT'
                    else:
                        self.closed = True
                        return 'EXIT_FINAL'
                return None
        else:
            if self.phase == 'approach':
                if high >= self.stop:
                    self.pos_dir, self.entry, self.extreme = 'UP', self.stop, self.stop
                    self.stop = self.entry - risk * OUTSIDE_STOP_MULT
                    self.phase = 'trailing'; self.flips += 1
                    return 'EXIT_AND_FLIP_LONG'
                if low <= self.far_edge:
                    self.extreme = low; self.phase = 'trailing'
                    self.stop = self.extreme + risk * TRAIL_MULT
                return None
            else:
                self.extreme = min(self.extreme, low)
                ns = self.extreme + risk * TRAIL_MULT
                if ns < self.stop: self.stop = ns
                if high >= self.stop:
                    if self.stop > self.far_edge:
                        exit_price = self.stop
                        self.pos_dir, self.entry, self.extreme = 'UP', exit_price, exit_price
                        self.far_edge = self.zone_high
                        self.stop = self.entry - risk * OUTSIDE_STOP_MULT
                        self.phase = 'approach'; self.flips += 1
                        return 'EXIT_AND_FLIP_LONG'
                    else:
                        self.closed = True
                        return 'EXIT_FINAL'
                return None


def latest_zone(closed_candles: pd.DataFrame, freq: str):
    """Given closed base-timeframe candles, resample to `freq`, find the most
    recent loose cluster, and return (zone_high, zone_low, zone_end_time) or None."""
    z = build_engine_frame(closed_candles, freq)
    color = z.color_code.to_numpy(dtype=object)
    clusters = find_loose_clusters(color)
    if not clusters:
        return None
    s, e = clusters[-1]
    if e < len(z) - 1:
        # a newer bar already exists after the cluster ended -> zone is "resolved", ready to trade
        zh = z.High.iloc[s:e+1].max(); zl = z.Low.iloc[s:e+1].min()
        return dict(zone_high=zh, zone_low=zl, zone_end=z.dt.iloc[e], risk=zh - zl)
    return None  # cluster still forming, not yet actionable
