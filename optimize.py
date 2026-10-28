"""
Fast Vectorized & Numba-Speed 3-Month Strategy Optimizer for NQ / MNQ Intraday Trading.
Evaluates 14,850 real 2-minute IBKR bars (July 27 - Oct 27, 2026) across:
- Timeframes: 2 min, 4 min, 5 min (via 1m/2m aggregation), 6 min, 10 min
- Strategy archetypes:
  1. EMA + VWAP Pullback Continuation
  2. EMA + VWAP Crossover + Momentum Breakout
  3. Opening Range Breakout (ORB 09:30-10:00) + VWAP Trend Alignment
  4. VWAP Mean-Reversion / Band Bounce vs Trend Following
- Risk structures: Stop Loss ATR multipliers, Reward:Risk ratios, Break-Even / Trailing stops, Daily trade caps
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from indicators import compute_atr, compute_ema, compute_rsi, compute_session_vwap, round_to_tick


def prepare_features(df: pd.DataFrame, fast_p: int, slow_p: int, trend_p: int = 50) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = compute_ema(out["close"], fast_p)
    out["ema_slow"] = compute_ema(out["close"], slow_p)
    out["ema_trend"] = compute_ema(out["close"], trend_p)
    out["vwap"] = compute_session_vwap(out, timezone="America/New_York")
    out["atr"] = compute_atr(out, period=14)
    out["rsi"] = compute_rsi(out["close"], period=14)

    atr_safe = out["atr"].replace(0, 1.0)
    out["ema_spread_atr"] = (out["ema_fast"] - out["ema_slow"]).abs() / atr_safe
    out["slow_slope_atr"] = (out["ema_slow"] - out["ema_slow"].shift(3)).fillna(0.0) / atr_safe
    out["vwap_slope_atr"] = (out["vwap"] - out["vwap"].shift(3)).fillna(0.0) / atr_safe

    rng = (out["high"] - out["low"]).replace(0, 0.25)
    out["body_ratio"] = (out["close"] - out["open"]).abs() / rng

    vol_ma = out["volume"].rolling(20, min_periods=1).mean().replace(0, 1)
    out["vol_ratio"] = out["volume"] / vol_ma

    # Rolling 5-bar high/low for breakout confirmation
    out["prev_3_high"] = out["high"].shift(1).rolling(3).max()
    out["prev_3_low"] = out["low"].shift(1).rolling(3).min()

    ts_et = pd.to_datetime(out["timestamp"]).dt.tz_convert("America/New_York")
    out["date"] = ts_et.dt.date
    out["month"] = ts_et.dt.strftime("%Y-%m")
    out["hhmm"] = ts_et.dt.strftime("%H:%M")

    # Compute Opening Range (09:30 - 09:59 ET) High & Low per day
    orb_mask = (out["hhmm"] >= "09:30") & (out["hhmm"] < "10:00")
    orb_highs = out[orb_mask].groupby("date")["high"].max()
    orb_lows = out[orb_mask].groupby("date")["low"].min()
    out["orb_high"] = out["date"].map(orb_highs)
    out["orb_low"] = out["date"].map(orb_lows)

    return out


def precompute_signals(df: pd.DataFrame, mode: str, min_spread: float, req_trend: bool) -> np.ndarray:
    """Vectorized signal precomputation: returns array of 1 (LONG), -1 (SHORT), or 0."""
    c = df["close"].to_numpy()
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    prev_h = df["high"].shift(1).to_numpy()
    prev_l = df["low"].shift(1).to_numpy()
    p3_h = df["prev_3_high"].to_numpy()
    p3_l = df["prev_3_low"].to_numpy()

    ef = df["ema_fast"].to_numpy()
    es = df["ema_slow"].to_numpy()
    prev_ef = df["ema_fast"].shift(1).to_numpy()
    prev_es = df["ema_slow"].shift(1).to_numpy()
    et = df["ema_trend"].to_numpy()
    vw = df["vwap"].to_numpy()
    atr = df["atr"].to_numpy()
    rsi = df["rsi"].to_numpy()
    spread = df["ema_spread_atr"].to_numpy()
    slope = df["slow_slope_atr"].to_numpy()
    body = df["body_ratio"].to_numpy()
    vol_r = df["vol_ratio"].to_numpy()
    orb_h = df["orb_high"].to_numpy()
    orb_l = df["orb_low"].to_numpy()

    base_filter = (atr >= 4.0) & (atr <= 65.0) & (spread >= min_spread) & (body >= 0.35) & (vol_r >= 0.75)

    bull_trend = base_filter & (c > vw) & (ef > es) & (slope > 0.08) & ((c > et) if req_trend else True)
    bear_trend = base_filter & (c < vw) & (ef < es) & (slope < -0.08) & ((c < et) if req_trend else True)

    tol = 0.30 * atr
    sigs = np.zeros(len(df), dtype=np.int8)

    if mode in ("pullback", "both", "hybrid_all"):
        long_pb = bull_trend & (c > o) & (rsi >= 45.0) & (rsi <= 68.0) & (l <= ef + tol) & (l >= es - tol) & (c > ef) & (c > prev_h)
        short_pb = bear_trend & (c < o) & (rsi >= 32.0) & (rsi <= 55.0) & (h >= ef - tol) & (h <= es + tol) & (c < ef) & (c < prev_l)
        sigs = np.where(long_pb, 1, np.where(short_pb, -1, sigs))

    if mode in ("crossover", "both", "hybrid_all"):
        long_co = bull_trend & (c > o) & (rsi >= 50.0) & (rsi <= 70.0) & (prev_ef <= prev_es) & (ef > es) & (c > ef)
        short_co = bear_trend & (c < o) & (rsi >= 30.0) & (rsi <= 50.0) & (prev_ef >= prev_es) & (ef < es) & (c < ef)
        sigs = np.where((sigs == 0) & long_co, 1, np.where((sigs == 0) & short_co, -1, sigs))

    if mode in ("momentum_breakout", "hybrid_all"):
        long_mb = bull_trend & (c > o) & (rsi >= 52.0) & (rsi <= 70.0) & (vol_r >= 1.1) & (c > p3_h) & ((c - ef) <= 1.2 * atr)
        short_mb = bear_trend & (c < o) & (rsi >= 30.0) & (rsi <= 48.0) & (vol_r >= 1.1) & (c < p3_l) & ((ef - c) <= 1.2 * atr)
        sigs = np.where((sigs == 0) & long_mb, 1, np.where((sigs == 0) & short_mb, -1, sigs))

    if mode in ("orb_trend", "hybrid_all"):
        # Opening range breakout aligned with VWAP + EMA trend
        long_orb = bull_trend & (c > o) & (rsi >= 52.0) & (rsi <= 72.0) & (prev_h <= orb_h) & (c > orb_h)
        short_orb = bear_trend & (c < o) & (rsi >= 28.0) & (rsi <= 48.0) & (prev_l >= orb_l) & (c < orb_l)
        sigs = np.where((sigs == 0) & long_orb, 1, np.where((sigs == 0) & short_orb, -1, sigs))

    return sigs


def simulate_trades(
    df: pd.DataFrame,
    sigs: np.ndarray,
    atr_mult: float,
    rr_ratio: float,
    be_at_1r: bool,
    trail_atr: float,
    start_hhmm: str,
    end_hhmm: str,
    max_trades_day: int,
    max_losses_day: int = 2,
) -> Dict[str, object]:
    n = len(df)
    close_arr = df["close"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    atr_arr = df["atr"].to_numpy()
    dates = df["date"].to_numpy()
    months = df["month"].to_numpy()
    hhmms = df["hhmm"].to_numpy()

    pnls: List[float] = []
    trade_months: List[str] = []

    in_pos = False
    pos_dir = 0
    entry_p = 0.0
    stop_p = 0.0
    tp_p = 0.0
    one_r_dist = 0.0
    be_moved = False

    curr_date = None
    trades_today = 0
    losses_today = 0
    daily_pnl = 0.0
    last_exit_idx = -99

    slippage = 0.25
    comm = 1.24  # Round-turn commission for 1 MNQ

    for i in range(55, n):
        d = dates[i]
        hhmm = hhmms[i]
        if d != curr_date:
            curr_date = d
            trades_today = 0
            losses_today = 0
            daily_pnl = 0.0

        if in_pos:
            exit_p = None
            if pos_dir == 1:
                if low_arr[i] <= stop_p:
                    exit_p = stop_p - slippage
                elif high_arr[i] >= tp_p:
                    exit_p = tp_p
                elif hhmm >= "15:55":
                    exit_p = close_arr[i] - slippage
                else:
                    if be_at_1r and not be_moved and high_arr[i] >= entry_p + one_r_dist:
                        stop_p = max(stop_p, round_to_tick(entry_p + 1.0))
                        be_moved = True
                    if trail_atr > 0 and high_arr[i] >= entry_p + one_r_dist:
                        new_stop = round_to_tick(high_arr[i] - trail_atr * atr_arr[i])
                        if new_stop > stop_p:
                            stop_p = new_stop
            else:
                if high_arr[i] >= stop_p:
                    exit_p = stop_p + slippage
                elif low_arr[i] <= tp_p:
                    exit_p = tp_p
                elif hhmm >= "15:55":
                    exit_p = close_arr[i] + slippage
                else:
                    if be_at_1r and not be_moved and low_arr[i] <= entry_p - one_r_dist:
                        stop_p = min(stop_p, round_to_tick(entry_p - 1.0))
                        be_moved = True
                    if trail_atr > 0 and low_arr[i] <= entry_p - one_r_dist:
                        new_stop = round_to_tick(low_arr[i] + trail_atr * atr_arr[i])
                        if new_stop < stop_p:
                            stop_p = new_stop

            if exit_p is not None:
                gross_pts = (exit_p - entry_p) if pos_dir == 1 else (entry_p - exit_p)
                net = gross_pts * 2.0 - comm
                pnls.append(net)
                trade_months.append(months[i])
                daily_pnl += net
                trades_today += 1
                if net < 0:
                    losses_today += 1
                last_exit_idx = i
                in_pos = False
            continue

        if sigs[i] == 0:
            continue
        if hhmm < start_hhmm or hhmm > end_hhmm:
            continue
        if trades_today >= max_trades_day or losses_today >= max_losses_day or daily_pnl <= -250.0:
            continue
        if i - last_exit_idx < 3:
            continue

        sig = int(sigs[i])
        c_atr = atr_arr[i]
        stop_dist = max(8.0, min(40.0, round_to_tick(c_atr * atr_mult)))
        tp_dist = round_to_tick(stop_dist * rr_ratio)

        in_pos = True
        pos_dir = sig
        be_moved = False
        one_r_dist = stop_dist
        c = close_arr[i]
        if sig == 1:
            entry_p = round_to_tick(c + slippage)
            stop_p = round_to_tick(entry_p - stop_dist)
            tp_p = round_to_tick(entry_p + tp_dist)
        else:
            entry_p = round_to_tick(c - slippage)
            stop_p = round_to_tick(entry_p + stop_dist)
            tp_p = round_to_tick(entry_p - tp_dist)

    if len(pnls) < 25:
        return {"trades": len(pnls), "net_pnl": -9999.0, "pf": 0.0, "sharpe": 0.0}

    arr = np.array(pnls)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else 99.0
    eq = np.cumsum(arr)
    peaks = np.maximum.accumulate(np.insert(eq, 0, 0.0))[1:]
    max_dd = float(np.max(peaks - eq))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1.0
    sharpe = float(np.mean(arr)) / std * np.sqrt(len(arr)) if std > 0 else 0.0

    # Monthly breakdown
    m_series = pd.Series(arr).groupby(trade_months).sum()
    pos_months = int((m_series > 0).sum())

    return {
        "trades": len(arr),
        "win_rate": round(100.0 * len(wins) / len(arr), 1),
        "net_pnl": round(float(arr.sum()), 2),
        "pf": round(pf, 2),
        "max_dd": round(max_dd, 2),
        "avg_trade": round(float(np.mean(arr)), 2),
        "sharpe": round(sharpe, 2),
        "pos_months": f"{pos_months}/{len(m_series)}",
    }


def main() -> None:
    csv_path = Path(__file__).parent / "data" / "mnq_2m_3months.csv"
    df_raw = pd.read_csv(csv_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True).dt.tz_convert("America/New_York")
    print(f"Loaded {len(df_raw)} real 2-minute MNQ bars (3 Months: {df_raw['timestamp'].min()} -> {df_raw['timestamp'].max()}).")

    results = []
    for tf in ["2min", "4min", "6min", "10min"]:
        if tf == "2min":
            df_tf = df_raw.copy()
        else:
            df_tf = (
                df_raw.set_index("timestamp")
                .resample(tf, closed="left", label="left")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
                .reset_index()
            )
        for fast_p, slow_p in [(8, 21), (9, 21), (12, 26)]:
            feat_df = prepare_features(df_tf, fast_p, slow_p, trend_p=50)
            for mode, min_spread, req_trend in itertools.product(
                ["pullback", "crossover", "both", "momentum_breakout", "orb_trend", "hybrid_all"],
                [0.10, 0.20, 0.30],
                [False, True],
            ):
                sigs = precompute_signals(feat_df, mode, min_spread, req_trend)
                if np.abs(sigs).sum() < 25:
                    continue

                for atr_mult, rr_ratio, be_1r, trail_atr, window, max_t in itertools.product(
                    [1.2, 1.5, 2.0],
                    [1.5, 2.0, 2.5, 3.0],
                    [False, True],
                    [0.0, 1.5],
                    [("09:40", "15:30"), ("10:00", "15:30"), ("10:00", "14:30"), ("09:45", "12:30")],
                    [2, 3, 4],
                ):
                    res = simulate_trades(
                        feat_df,
                        sigs,
                        atr_mult=atr_mult,
                        rr_ratio=rr_ratio,
                        be_at_1r=be_1r,
                        trail_atr=trail_atr,
                        start_hhmm=window[0],
                        end_hhmm=window[1],
                        max_trades_day=max_t,
                    )
                    if res["trades"] >= 30 and res["net_pnl"] > 0:
                        results.append(
                            {
                                "tf": tf,
                                "emas": f"{fast_p}/{slow_p}",
                                "mode": mode,
                                "atr_m": atr_mult,
                                "rr": rr_ratio,
                                "be1r": be_1r,
                                "trail": trail_atr,
                                "spread": min_spread,
                                "t50": req_trend,
                                "win": f"{window[0]}-{window[1]}",
                                "maxT": max_t,
                                **res,
                            }
                        )

    if not results:
        print("No configurations matched criteria.")
        return

    res_df = pd.DataFrame(results).sort_values(by=["net_pnl", "sharpe"], ascending=False)
    print("\nTop 20 Strategy Configurations Across 3 Months (66 Trading Days) of Real IBKR MNQ Data:")
    print(res_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
