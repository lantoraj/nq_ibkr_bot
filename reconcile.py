"""
Live-vs-Backtest Reconciliation CLI (`reconcile.py`).

Replays recorded live bars (or an official IBKR historical CSV) through `Backtester`
and compares every live signal, fill, slippage, and trade outcome against the backtest.

Usage:
  python reconcile.py                       # Reconcile all recorded journal dates
  python reconcile.py --date 2026-11-09     # Reconcile a specific ET trading date
  python reconcile.py --date 2026-11-09 --ibkr-csv data/ibkr_check.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from backtest import Backtester, SimulatedTrade
from config import BotConfig
from trade_journal import DEFAULT_JOURNAL_DIR, load_journal


def list_journal_days(base_dir: str = DEFAULT_JOURNAL_DIR) -> List[str]:
    events_dir = Path(base_dir) / "events"
    if not events_dir.exists():
        return []
    return sorted(p.stem for p in events_dir.glob("*.jsonl"))


def reconstruct_bars_df(events: List[dict]) -> pd.DataFrame:
    rows = []
    seen = set()
    for ev in events:
        if ev.get("event") == "bar" and "bar_ts" in ev:
            ts = ev["bar_ts"]
            if ts in seen:
                continue
            seen.add(ts)
            rows.append(
                {
                    "timestamp": pd.Timestamp(ts),
                    "open": float(ev["open"]),
                    "high": float(ev["high"]),
                    "low": float(ev["low"]),
                    "close": float(ev["close"]),
                    "volume": float(ev["volume"]),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def reconcile_day(
    day: str,
    base_dir: str = DEFAULT_JOURNAL_DIR,
    ibkr_csv: Optional[str] = None,
    warmup_days: int = 2,
) -> dict:
    """Reconcile a single ET trading day (`YYYY-MM-DD`) and return structured summary metrics."""
    all_days = list_journal_days(base_dir)
    prior_days = [d for d in all_days if d < day][-warmup_days:]
    warmup_events: List[dict] = []
    for d in prior_days:
        warmup_events.extend(load_journal(d, base_dir=base_dir))
    day_events = load_journal(day, base_dir=base_dir)

    combined_bars = reconstruct_bars_df(warmup_events + day_events)
    cfg = BotConfig()

    if ibkr_csv and Path(ibkr_csv).exists():
        ref_df = pd.read_csv(ibkr_csv)
        ref_df["timestamp"] = pd.to_datetime(ref_df["timestamp"], utc=True)
        bt_input_df = ref_df
    else:
        bt_input_df = combined_bars

    bt_trades_day: List[SimulatedTrade] = []
    if not bt_input_df.empty:
        bt = Backtester(cfg)
        res = bt.run(bt_input_df)
        bt_trades_day = [
            t for t in res["trades"] if t.entry_time.tz_convert(cfg.risk.timezone).strftime("%Y-%m-%d") == day
        ]

    live_signals = [e for e in day_events if e.get("event") == "signal"]
    blocked_signals = [e for e in day_events if e.get("event") == "signal_blocked"]
    cancelled_entries = [e for e in day_events if e.get("event") == "entry_cancelled"]
    live_fills = [e for e in day_events if e.get("event") == "entry_filled"]
    live_closed = [e for e in day_events if e.get("event") == "trade_closed"]
    day_bars = [e for e in day_events if e.get("event") == "bar"]

    # Match signals by bar timestamp (within 2 minutes tolerance)
    bt_by_ts = {
        pd.Timestamp(t.entry_time).tz_convert("UTC"): t
        for t in bt_trades_day
    }
    matched = []
    live_only = []
    matched_bt_keys = set()

    for sig in live_signals:
        sig_ts = pd.Timestamp(sig["bar_ts"]).tz_convert("UTC")
        match_t = bt_by_ts.get(sig_ts)
        if match_t is None:
            # Check +/- 2 min tolerance
            for b_ts, b_t in bt_by_ts.items():
                if abs((b_ts - sig_ts).total_seconds()) <= 120 and b_t.direction == sig["direction"]:
                    match_t = b_t
                    matched_bt_keys.add(b_ts)
                    break
        else:
            matched_bt_keys.add(sig_ts)

        if match_t is not None:
            matched.append((sig, match_t))
        else:
            live_only.append(sig)

    bt_only = [t for k, t in bt_by_ts.items() if k not in matched_bt_keys]

    slippages = [float(f["slippage_points"]) for f in live_fills if f.get("slippage_points") is not None]
    delays = [float(f["fill_delay_sec"]) for f in live_fills if f.get("fill_delay_sec") is not None]
    live_net_pnl = sum(float(c.get("net_pnl_usd", 0.0)) for c in live_closed)
    bt_net_pnl = sum(float(t.net_pnl_usd) for t in bt_trades_day)

    # Feed accuracy comparison if reference IBKR CSV is supplied
    feed_mae_pts = None
    if ibkr_csv and Path(ibkr_csv).exists() and not combined_bars.empty:
        ref_df = pd.read_csv(ibkr_csv)
        ref_df["ts_utc"] = pd.to_datetime(ref_df["timestamp"], utc=True)
        live_df = reconstruct_bars_df(day_events)
        if not live_df.empty:
            live_df["ts_utc"] = pd.to_datetime(live_df["timestamp"], utc=True)
            merged = pd.merge(live_df, ref_df, on="ts_utc", suffixes=("_live", "_ibkr"))
            if not merged.empty:
                feed_mae_pts = float((merged["close_live"] - merged["close_ibkr"]).abs().mean())

    return {
        "day": day,
        "bars_logged": len(day_bars),
        "live_signals": len(live_signals),
        "blocked_signals": len(blocked_signals),
        "cancelled_entries": len(cancelled_entries),
        "live_fills": len(live_fills),
        "live_closed": len(live_closed),
        "bt_trades": len(bt_trades_day),
        "matched_signals": len(matched),
        "live_only_signals": len(live_only),
        "bt_only_signals": len(bt_only),
        "avg_slippage_pts": round(float(np.mean(slippages)), 3) if slippages else 0.0,
        "avg_fill_delay_sec": round(float(np.mean(delays)), 2) if delays else 0.0,
        "live_net_pnl_usd": round(live_net_pnl, 2),
        "bt_net_pnl_usd": round(bt_net_pnl, 2),
        "pnl_diff_usd": round(live_net_pnl - bt_net_pnl, 2),
        "feed_mae_pts": round(feed_mae_pts, 3) if feed_mae_pts is not None else None,
        "matched_details": matched,
        "live_closed_details": live_closed,
        "bt_trades_details": bt_trades_day,
    }


def print_report(res: dict) -> None:
    print(f"\n======================================================================")
    print(f" LIVE vs. BACKTEST RECONCILIATION — {res['day']} (ET)")
    print(f"======================================================================")
    print(f"  2-Min Bars Logged Live : {res['bars_logged']}")
    if res["feed_mae_pts"] is not None:
        print(f"  Feed Close MAE vs IBKR : {res['feed_mae_pts']:.2f} pts")
    print(f"  Live Signals Fired     : {res['live_signals']} (Blocked by Risk: {res['blocked_signals']}, Unfilled/Cancelled: {res['cancelled_entries']})")
    print(f"  Backtest Trades Fired  : {res['bt_trades']}")
    print(f"  Signal Alignment       : {res['matched_signals']} matched | {res['live_only_signals']} live-only | {res['bt_only_signals']} backtest-only")
    print(f"  Live Fills / Closed    : {res['live_fills']} filled / {res['live_closed']} closed")
    print(f"  Avg Entry Slippage     : {res['avg_slippage_pts']:+.2f} pts ({res['avg_slippage_pts']/0.25:+.1f} ticks) | Avg Fill Delay: {res['avg_fill_delay_sec']:.1f}s")
    print(f"  Live Net PnL           : ${res['live_net_pnl_usd']:+.2f}")
    print(f"  Backtest Net PnL       : ${res['bt_net_pnl_usd']:+.2f} (Assumes 1-tick = 0.25 pt slippage)")
    print(f"  Live - Backtest Delta  : ${res['pnl_diff_usd']:+.2f}")
    if res["live_closed_details"]:
        print("\n  --- Live Closed Trades ---")
        for c in res["live_closed_details"]:
            print(
                f"    [{c.get('bar_ts', '')[:16]}] {c.get('direction'):5s} | Sig={c.get('signal_price')} -> Fill={c.get('entry_price')} -> Exit={c.get('exit_price')} | Net=${c.get('net_pnl_usd', 0.0):+.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile live journal events against the Backtester.")
    parser.add_argument("--date", default="", help="Specific ET date YYYY-MM-DD (default: all dates)")
    parser.add_argument("--journal-dir", default=DEFAULT_JOURNAL_DIR, help="Journal directory path")
    parser.add_argument("--ibkr-csv", default="", help="Optional IBKR historical CSV for feed error comparison")
    args = parser.parse_args()

    days = [args.date] if args.date else list_journal_days(args.journal_dir)
    if not days:
        print(f"No journal files found in '{args.journal_dir}/events/'.")
        return
    for d in days:
        res = reconcile_day(d, base_dir=args.journal_dir, ibkr_csv=args.ibkr_csv or None)
        print_report(res)


if __name__ == "__main__":
    main()
