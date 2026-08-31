import pandas as pd, numpy as np

res = pd.read_csv('/home/claude/final_results_with_risk.csv')
res['entry_time'] = pd.to_datetime(res['entry_time'])
res = res.sort_values('entry_time').reset_index(drop=True)

START_BALANCE = 1500.0
RISK_FRACTION = 0.03      # 3% of CURRENT equity risked per trade
MAX_LEVERAGE = 100.0

def simulate(res, risk_col):
    balance = START_BALANCE
    rows = []
    liquidation_events = 0
    capped_events = 0
    for row in res.itertuples():
        risk_pct = row.risk_pct / 100.0  # fraction
        desired_notional = (balance * RISK_FRACTION) / risk_pct
        max_notional = balance * MAX_LEVERAGE
        notional = min(desired_notional, max_notional)
        effective_risk_fraction = notional * risk_pct / balance
        capped = notional < desired_notional
        if capped: capped_events += 1

        leverage_used = notional / balance
        liq_distance = 1.0 / leverage_used  # rough, ignoring maintenance margin
        liquidated = liq_distance <= risk_pct  # stop is farther than liquidation -> wipeout first
        if liquidated:
            liquidation_events += 1
            pnl_fraction = -effective_risk_fraction  # lose the margin allocated to this trade
        else:
            r = getattr(row, risk_col)
            pnl_fraction = effective_risk_fraction * r

        balance_before = balance
        balance = balance * (1 + pnl_fraction)
        balance = max(balance, 0.01)  # floor, avoid negative
        rows.append(dict(tf=row.tf, entry_time=row.entry_time, risk_pct=row.risk_pct,
                          effective_risk_fraction=effective_risk_fraction, capped=capped,
                          liquidated=liquidated, pnl_fraction=pnl_fraction,
                          balance_before=balance_before, balance_after=balance))
    out = pd.DataFrame(rows)
    return out, liquidation_events, capped_events

print("="*70)
print("HONEST EQUITY SIMULATION — $1,500 start, 3% risk/trade, 100x leverage cap")
print("="*70)

for label, col in [('GROSS (no fees)', 'r_gross'), ('NET (0.10% RT fee per leg)', 'r_net')]:
    sim, liq, capped = simulate(res, col)
    final_bal = sim.balance_after.iloc[-1]
    print(f"\n--- {label} ---")
    print(f"final balance: ${final_bal:,.2f}   (return: {(final_bal/START_BALANCE-1)*100:,.1f}%)")
    print(f"liquidation events: {liq} / {len(sim)}   (leverage-capped trades: {capped})")
    print(f"max drawdown: {(1 - (sim.balance_after/sim.balance_after.cummax())).max()*100:.1f}%")
    print("\nper-timeframe contribution (avg pnl_fraction per trade, count):")
    print(sim.groupby('tf').pnl_fraction.agg(n='size', mean_pnl_frac='mean', total_pnl_frac='sum').round(4))
    sim.to_csv(f'/home/claude/equity_sim_{"gross" if col=="r_gross" else "net"}.csv', index=False)

# also show a per-timeframe ISOLATED simulation (each tf trading with its OWN separate $1500, for comparison)
print("\n" + "="*70)
print("For comparison: EACH TIMEFRAME TRADED SEPARATELY (own $1,500 each, net of fees)")
print("="*70)
for tf in res.tf.unique():
    sub = res[res.tf==tf].sort_values('entry_time').reset_index(drop=True)
    sim, liq, capped = simulate(sub, 'r_net')
    final_bal = sim.balance_after.iloc[-1]
    print(f"{tf:6s} n={len(sub):4d}  final=${final_bal:>12,.2f}  return={((final_bal/START_BALANCE-1)*100):>8.1f}%  liq={liq}")
