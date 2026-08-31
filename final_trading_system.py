#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
 FULL SYSTEM — "Weak-Bar Range / Smart-Flip Breakout Engine"
==========================================================================
Pipeline (everything built and validated across this whole project):

  STAGE 1 — Bar classification ("رنگ میله‌ها")
     For each timeframe: chart_bar (High-Low), system_bar (sum of |Close-Open|
     of the underlying 15m bars), rolling 720-bar quantiles -> 5-level
     strength labels -> color_code (YELLOW/WHITE/PINK/GREEN/RED/BLUE/...)

  STAGE 2 — Volume-weighted intensity ("میانگین حرکت به حجم")
     volume_time = volume * bar_duration_hours
     bar_intensity = |price_change| / volume_time
     intensity_avg_causal = running (look-back only) average intensity
     intensity_ratio = bar_intensity / intensity_avg_causal  -> pressure_level5

  STAGE 3 — Range-zone detection ("محدوده‌های رنج")
     Loose clusters of >=4 weak bars (YELLOW/WHITE), bridging gaps of <=2
     non-weak bars. zone_high/zone_low = full span High/Low.

  STAGE 4 — Trading logic ("سیستم اسمارت-فلیپ")
     Enter LONG at zone_low. Protective stop OUTSIDE the range (0.25R).
     Once price reaches the far edge (zone_high), switch to a tight
     trailing stop (0.15R). If trail exits BACK inside/through the range
     -> confirmed fake breakout -> flip to SHORT with a fresh outside
     stop. If trail exits beyond the range -> genuine breakout -> ride it,
     no flip. Execution/path resolution always uses 15m closed candles
     for precision, regardless of the zone-detection timeframe.

  STAGE 5 — Backtest & report
==========================================================================
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# STAGE 1+2: bar classification + intensity engine (any timeframe)
# ---------------------------------------------------------------------
W = 720  # rolling quantile window, in bars of the TARGET timeframe

def build_engine_frame(base_15m: pd.DataFrame, freq: str) -> pd.DataFrame:
    base = base_15m.set_index('dt')
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
        r[ok] = np.array(['VERY_WEAK', 'WEAK', 'NORMAL', 'STRONG', 'VERY_STRONG'])[
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
    z['time_delta_hours'] = dt_h.fillna(dt_h.iloc[1])
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


# ---------------------------------------------------------------------
# STAGE 3: loose weak-bar cluster (range zone) detection
# ---------------------------------------------------------------------
GAP_TOL = 2
MIN_WEAK = 4

def find_loose_clusters(color: np.ndarray):
    is_weak = np.isin(color, ['YELLOW', 'WHITE'])
    n = len(color)
    clusters, i = [], 0
    while i < n:
        if is_weak[i]:
            start = i; weak_count = 1; last_weak = i; j = i + 1
            while j < n:
                if is_weak[j]:
                    weak_count += 1; last_weak = j; j += 1
                else:
                    gap = 1; k = j
                    while k < n and not is_weak[k] and gap <= GAP_TOL:
                        k += 1; gap += 1
                    if k < n and is_weak[k] and gap <= GAP_TOL: j = k
                    else: break
            end = last_weak
            if weak_count >= MIN_WEAK: clusters.append((start, end))
            i = end + 1
        else:
            i += 1
    return clusters


# ---------------------------------------------------------------------
# STAGE 4: smart-flip trade simulation (executed on 15m closed candles)
# ---------------------------------------------------------------------
TRAIL_MULT = 0.15          # trailing distance once the far edge is reached
OUTSIDE_STOP_MULT = 0.25   # protective stop, placed OUTSIDE the range
HOLD_HOURS = 240           # 10-day resolution horizon per zone
MAX_FLIPS = 8
FEE_PCT_ROUNDTRIP = 0.10   # realistic exchange fee + slippage, % round-trip, charged per leg

def _idx_at_or_after(t15_int, ts):
    return np.searchsorted(t15_int, pd.Timestamp(ts).value, side='left')

def simulate_smart_flip(rh, rl, risk, start_idx, end_idx, H15, L15, C15):
    """Returns (total_r_gross, total_r_net_of_fees, flips)."""
    pos_dir, entry = 'UP', rl
    stop = entry - risk * OUTSIDE_STOP_MULT
    far_edge, phase, extreme = rh, 'approach', entry
    total_r, total_r_net, flips, k = 0.0, 0.0, 0, start_idx

    def close_leg(exit_price, leg_entry, leg_dir):
        r = (exit_price - leg_entry) / risk if leg_dir == 'UP' else (leg_entry - exit_price) / risk
        risk_pct = risk / leg_entry * 100
        fee_r = FEE_PCT_ROUNDTRIP / risk_pct
        return r, r - fee_r

    while k <= end_idx and flips <= MAX_FLIPS:
        if pos_dir == 'UP':
            if phase == 'approach':
                if L15[k] <= stop:
                    r, rn = close_leg(stop, entry, 'UP'); total_r += r; total_r_net += rn
                    pos_dir, entry, extreme = 'DOWN', stop, stop
                    stop = entry + risk * OUTSIDE_STOP_MULT
                    phase, flips = 'trailing', flips + 1
                elif H15[k] >= far_edge:
                    extreme = H15[k]; phase = 'trailing'; stop = extreme - risk * TRAIL_MULT
            else:
                extreme = max(extreme, H15[k]); ns = extreme - risk * TRAIL_MULT
                if ns > stop: stop = ns
                if L15[k] <= stop:
                    r, rn = close_leg(stop, entry, 'UP'); total_r += r; total_r_net += rn
                    if stop < far_edge:
                        pos_dir, entry, extreme = 'DOWN', stop, stop
                        far_edge = rl; stop = entry + risk * OUTSIDE_STOP_MULT
                        phase, flips = 'approach', flips + 1
                    else:
                        return total_r, total_r_net, flips
        else:
            if phase == 'approach':
                if H15[k] >= stop:
                    r, rn = close_leg(stop, entry, 'DOWN'); total_r += r; total_r_net += rn
                    pos_dir, entry, extreme = 'UP', stop, stop
                    stop = entry - risk * OUTSIDE_STOP_MULT
                    phase, flips = 'trailing', flips + 1
                elif L15[k] <= far_edge:
                    extreme = L15[k]; phase = 'trailing'; stop = extreme + risk * TRAIL_MULT
            else:
                extreme = min(extreme, L15[k]); ns = extreme + risk * TRAIL_MULT
                if ns < stop: stop = ns
                if H15[k] >= stop:
                    r, rn = close_leg(stop, entry, 'DOWN'); total_r += r; total_r_net += rn
                    if stop > far_edge:
                        pos_dir, entry, extreme = 'UP', stop, stop
                        far_edge = rh; stop = entry - risk * OUTSIDE_STOP_MULT
                        phase, flips = 'approach', flips + 1
                    else:
                        return total_r, total_r_net, flips
        k += 1
    final = C15[end_idx]
    r, rn = close_leg(final, entry, pos_dir)
    total_r += r; total_r_net += rn
    return total_r, total_r_net, flips


# ---------------------------------------------------------------------
# STAGE 5: full backtest across timeframes
# ---------------------------------------------------------------------
def run_full_system(source_csv: str, timeframes=('30min', '1h', '2h', '4h', '6h', '1D')):
    base = pd.read_csv(source_csv, usecols=['dt', 'Open', 'High', 'Low', 'Close', 'total_volume', 'Trades'])
    base['dt'] = pd.to_datetime(base['dt'], utc=True)
    base = base.sort_values('dt').reset_index(drop=True)

    exec_df = base[['dt', 'High', 'Low', 'Close']].copy()
    t15 = exec_df.dt.astype('datetime64[ns, UTC]').astype('int64').to_numpy()
    H15, L15, C15 = exec_df.High.to_numpy(), exec_df.Low.to_numpy(), exec_df.Close.to_numpy()

    all_rows = []
    for tf in timeframes:
        z = build_engine_frame(base[['dt','Open','High','Low','Close','total_volume','Trades']].copy(), tf)
        color = z.color_code.to_numpy(dtype=object)
        High, Low, dts = z.High.to_numpy(), z.Low.to_numpy(), z.dt.to_numpy()
        clusters = find_loose_clusters(color)

        for (s, e) in clusters:
            zone_high, zone_low = High[s:e+1].max(), Low[s:e+1].min()
            risk = zone_high - zone_low
            if risk <= 0: continue
            start15 = _idx_at_or_after(t15, dts[e])
            if start15 >= len(t15): continue
            end15 = min(len(t15) - 1, start15 + int(HOLD_HOURS * 4))
            r, rn, flips = simulate_smart_flip(zone_high, zone_low, risk, start15, end15, H15, L15, C15)
            all_rows.append(dict(tf=tf, zone_end=dts[e], r_gross=r, r_net=rn, flips=flips))

    res = pd.DataFrame(all_rows)
    return res


def report(res: pd.DataFrame):
    print(f"\n{'='*70}\nFINAL SYSTEM BACKTEST — all timeframes combined (fee={FEE_PCT_ROUNDTRIP}% RT/leg)\n{'='*70}")
    print(f"total zone events traded: {len(res)}")
    for col, label in [('r_gross','GROSS (no fees)'), ('r_net','NET (after fees)')]:
        print(f"\n--- {label} ---")
        print(f"win rate (R>0):           {(res[col]>0).mean():.1%}")
        print(f"mean R per event:         {res[col].mean():.3f}")
        print(f"median R per event:       {res[col].median():.3f}")
        print(f"sum R (total):            {res[col].sum():.1f}")
    top10 = res.r_net.sort_values(ascending=False).head(10).sum()
    print(f"\ntop-10 trades share of NET sum: {top10/res.r_net.sum():.1%}")
    print(f"avg flips/event:          {res.flips.mean():.2f}")
    print(f"\n--- by timeframe (NET of fees) ---")
    print(res.groupby('tf').r_net.agg(n='size', win=lambda x: (x>0).mean(), mean='mean', sum='sum').round(3))

if __name__ == '__main__':
    res = run_full_system('/mnt/user-data/uploads/15m.csv')
    res.to_csv('/home/claude/final_system_results.csv', index=False)
    report(res)
