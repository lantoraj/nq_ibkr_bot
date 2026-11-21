"""
Event-driven Backtester & Synthetic NQ Intraday Data Generator.
Uses the exact same `NQIntradayStrategy` and `RiskManager` classes as the live IBKR runner.
Simulates:
- Bracket Order fills (Stop Loss, Take Profit, and 15:55 ET End-of-Day flatten)
- Commission & slippage modeling for NQ / MNQ futures
- Detailed performance report (Win Rate, Profit Factor, Max Drawdown, Sharpe, Expectancy)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import BotConfig
from indicators import round_to_tick
from risk_manager import RiskManager
from strategy import NQIntradayStrategy, TradeSignal


@dataclass
class SimulatedTrade:
    """Record of a completed backtest trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    reason: str
    contracts: int
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    exit_reason: str
    gross_points: float
    net_pnl_usd: float


def generate_synthetic_nq_data(
    num_days: int = 5,
    seed: int = 42,
    start_price: float = 20250.0,
) -> pd.DataFrame:
    """
    Generate realistic 1-minute RTH (09:30 - 16:00 ET) NQ futures bars with
    intraday trend regimes, pullbacks, and volume profiles for offline verification.
    """
    rng = np.random.default_rng(seed)
    rows = []
    base_dates = pd.bdate_range("2026-10-19", periods=num_days)
    price = start_price

    for day_idx, bdate in enumerate(base_dates):
        day_timestamps = pd.date_range(
            f"{bdate.strftime('%Y-%m-%d')} 09:30:00",
            f"{bdate.strftime('%Y-%m-%d')} 15:59:00",
            freq="1min",
            tz="America/New_York",
        )
        # Alternate bullish and bearish intraday trend regimes with periodic 5-8 bar pullbacks
        regime_dir = 1.0 if day_idx % 2 == 0 else -1.0

        for m_idx, ts in enumerate(day_timestamps):
            # Create structured trend + periodic pullback waves every ~25 minutes
            cycle_pos = m_idx % 28
            if cycle_pos < 20:
                drift = regime_dir * 0.85
            elif cycle_pos < 25:
                # Controlled pullback toward fast/slow EMA
                drift = -regime_dir * 1.15
            else:
                # Strong resumption bar reclaiming Fast EMA
                drift = regime_dir * 2.10

            noise = rng.normal(0, 2.2)
            delta = drift + noise

            open_p = round_to_tick(price)
            close_p = round_to_tick(open_p + delta)
            wick_up = abs(rng.normal(2.5, 1.2))
            wick_dn = abs(rng.normal(2.5, 1.2))

            high_p = round_to_tick(max(open_p, close_p) + wick_up)
            low_p = round_to_tick(min(open_p, close_p) - wick_dn)

            # Higher volume at open, close, and trend resumption bars
            u_shape = 1.6 if (m_idx < 30 or m_idx > 350) else 1.0
            resumption_boost = 1.35 if cycle_pos >= 25 else 1.0
            volume = int(max(150, rng.normal(950, 220) * u_shape * resumption_boost))

            rows.append(
                {
                    "timestamp": ts,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": volume,
                }
            )
            price = close_p

    return pd.DataFrame(rows)


class Backtester:
    """Bar-by-bar backtester for the NQ/MNQ Intraday Strategy."""

    def __init__(
        self,
        config: BotConfig,
        commission_per_side_usd: float = 0.62,  # Typical IBKR MNQ commission ($2.10 for NQ)
        slippage_ticks: int = 1,
        use_limit_execution: bool = False,
    ) -> None:
        self.cfg = config
        self.strategy = NQIntradayStrategy(config.strategy, config.risk, config.contract)
        self.risk = RiskManager(config.risk, config.contract)
        self.commission_per_side = commission_per_side_usd
        self.slippage_points = slippage_ticks * config.contract.tick_size
        self.use_limit_execution = use_limit_execution

    def run(self, df: pd.DataFrame) -> Dict[str, object]:
        """Run bar-by-bar simulation and return trade log and summary metrics."""
        work_df = df.copy()
        work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True).dt.tz_convert(self.cfg.risk.timezone)

        # Automatically resample 1-minute CSV data if strategy is configured for 2 mins or 5 mins
        if len(work_df) > 2:
            median_diff_sec = work_df["timestamp"].diff().dt.total_seconds().median()
            target_map = {"1 min": ("1min", 60), "2 mins": ("2min", 120), "5 mins": ("5min", 300)}
            rule, target_sec = target_map.get(self.cfg.strategy.bar_size, ("1min", 60))
            if median_diff_sec == 60 and target_sec > 60:
                work_df = (
                    work_df.set_index("timestamp")
                    .resample(rule, closed="left", label="left")
                    .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                    .dropna()
                    .reset_index()
                )

        enriched = self.strategy.prepare_dataframe(work_df)
        trades: List[SimulatedTrade] = []

        active_signal: Optional[TradeSignal] = None
        active_contracts: int = 0
        rem_contracts: int = 0
        active_entry_price: float = 0.0
        active_stop_loss: float = 0.0
        active_take_profit: float = 0.0
        active_tp1_price: float = 0.0
        active_tp1_r: float = 0.0
        active_is_pm: bool = False
        use_partial_tp: bool = False
        tp1_done: bool = False
        realized_leg1_pnl: float = 0.0
        active_entry_idx: int = 0
        be_moved: bool = False
        decay_moved: bool = False

        # Pending StopLimit order state when use_limit_execution=True
        pending_signal: Optional[TradeSignal] = None
        pending_sig_idx: int = -1
        pending_lmt: float = 0.0
        pending_stp: float = 0.0
        pending_best_fav: float = 0.0
        prev_session_date = None

        point_val = self.cfg.contract.point_value
        tick = self.cfg.contract.tick_size

        def _activate_position(sig: TradeSignal, fill_price: float, bar_idx: int, reanchor_sl_tp: bool) -> None:
            nonlocal active_signal, active_contracts, rem_contracts, active_entry_price
            nonlocal active_stop_loss, active_take_profit, active_tp1_price, active_tp1_r
            nonlocal active_is_pm, use_partial_tp, tp1_done, realized_leg1_pnl
            nonlocal active_entry_idx, be_moved, decay_moved

            active_signal = sig
            active_contracts = self.risk.calculate_contracts(sig.stop_distance_points)
            rem_contracts = active_contracts
            active_entry_price = fill_price
            one_r_pts = sig.stop_distance_points

            sig_et = pd.Timestamp(sig.timestamp)
            sig_et = (
                sig_et.tz_localize(self.cfg.risk.timezone)
                if sig_et.tzinfo is None
                else sig_et.tz_convert(self.cfg.risk.timezone)
            )
            active_is_pm = sig_et.strftime("%H:%M") < "09:30"
            use_partial_tp = (
                active_contracts >= 2
                and getattr(self.cfg.risk, "enable_partial_tp", True)
            )

            if reanchor_sl_tp:
                active_stop_loss = (
                    round_to_tick(fill_price - one_r_pts, tick)
                    if sig.direction == "LONG"
                    else round_to_tick(fill_price + one_r_pts, tick)
                )
            else:
                active_stop_loss = sig.stop_loss

            if use_partial_tp:
                active_tp1_r = (
                    getattr(self.cfg.risk, "premarket_partial_tp1_r", 1.4)
                    if active_is_pm
                    else getattr(self.cfg.risk, "partial_tp1_r", 2.0)
                )
                runner_r = (
                    getattr(self.cfg.risk, "premarket_reward_risk_ratio", 2.0)
                    if active_is_pm
                    else getattr(self.cfg.risk, "runner_reward_risk_ratio", 5.0)
                )
                active_tp1_price = (
                    round_to_tick(active_entry_price + active_tp1_r * one_r_pts, tick)
                    if sig.direction == "LONG"
                    else round_to_tick(active_entry_price - active_tp1_r * one_r_pts, tick)
                )
                runner_dist = round_to_tick(one_r_pts * runner_r, tick)
                tp2_base = fill_price if reanchor_sl_tp else sig.entry_price
                active_take_profit = (
                    round_to_tick(tp2_base + runner_dist, tick)
                    if sig.direction == "LONG"
                    else round_to_tick(tp2_base - runner_dist, tick)
                )
            else:
                active_tp1_r = 0.0
                active_tp1_price = 0.0
                if reanchor_sl_tp:
                    rr = (
                        getattr(self.cfg.risk, "premarket_reward_risk_ratio", self.cfg.risk.reward_risk_ratio)
                        if active_is_pm
                        else self.cfg.risk.reward_risk_ratio
                    )
                    tp_dist = round_to_tick(one_r_pts * rr, tick)
                    active_take_profit = (
                        round_to_tick(fill_price + tp_dist, tick)
                        if sig.direction == "LONG"
                        else round_to_tick(fill_price - tp_dist, tick)
                    )
                else:
                    active_take_profit = sig.take_profit

            active_entry_idx = bar_idx
            be_moved = False
            decay_moved = False
            tp1_done = False
            realized_leg1_pnl = 0.0

        for i in range(len(enriched)):
            bar = enriched.iloc[i]
            ts = pd.Timestamp(bar["timestamp"])
            open_p = float(bar["open"])
            high_p = float(bar["high"])
            low_p = float(bar["low"])
            close_p = float(bar["close"])

            self.risk.check_and_reset_session(ts)
            bar_date = self.risk.state.session_date
            if bar_date != prev_session_date:
                prev_session_date = bar_date
                pending_signal = None

            # 0. Evaluate resting StopLimit entry order if one is pending fill
            if self.use_limit_execution and pending_signal is not None and active_signal is None:
                bars_waited = i - pending_sig_idx
                if self.risk.should_flatten_eod(ts):
                    pending_signal = None
                else:
                    cancel_bars = int(getattr(self.cfg.risk, "cancel_unfilled_entry_bars", 1) or 1)
                    runaway_pts = float(getattr(self.cfg.risk, "runaway_cancel_points", 0.0) or 0.0)
                    filled_price: Optional[float] = None
                    cancelled = False

                    if pending_signal.direction == "LONG":
                        if pending_stp <= open_p <= pending_lmt:
                            filled_price = min(pending_lmt, round_to_tick(open_p + self.slippage_points, tick))
                        elif runaway_pts > 0.0 and pending_best_fav >= runaway_pts:
                            cancelled = True
                        elif open_p > pending_lmt and low_p <= pending_lmt:
                            if runaway_pts > 0.0 and (high_p - pending_signal.entry_price) >= runaway_pts:
                                cancelled = True
                            else:
                                filled_price = pending_lmt
                        elif open_p < pending_stp and high_p >= pending_stp:
                            filled_price = min(pending_lmt, round_to_tick(pending_stp + self.slippage_points, tick))
                        if filled_price is None and not cancelled:
                            pending_best_fav = max(pending_best_fav, high_p - pending_signal.entry_price)
                    else:  # SHORT
                        if pending_lmt <= open_p <= pending_stp:
                            filled_price = max(pending_lmt, round_to_tick(open_p - self.slippage_points, tick))
                        elif runaway_pts > 0.0 and pending_best_fav >= runaway_pts:
                            cancelled = True
                        elif open_p < pending_lmt and high_p >= pending_lmt:
                            if runaway_pts > 0.0 and (pending_signal.entry_price - low_p) >= runaway_pts:
                                cancelled = True
                            else:
                                filled_price = pending_lmt
                        elif open_p > pending_stp and low_p <= pending_stp:
                            filled_price = max(pending_lmt, round_to_tick(pending_stp - self.slippage_points, tick))
                        if filled_price is None and not cancelled:
                            pending_best_fav = max(pending_best_fav, pending_signal.entry_price - low_p)

                    if filled_price is not None:
                        _activate_position(pending_signal, filled_price, bar_idx=i, reanchor_sl_tp=True)
                        pending_signal = None
                    else:
                        if cancelled or bars_waited >= cancel_bars or (
                            runaway_pts > 0.0 and pending_best_fav >= runaway_pts
                        ):
                            pending_signal = None
                        continue

            # 1. Manage open position if one is active
            if active_signal is not None:
                exit_price: Optional[float] = None
                exit_reason = ""
                one_r = active_signal.stop_distance_points
                bars_held = i - active_entry_idx
                be_r = (
                    getattr(self.cfg.risk, "premarket_breakeven_r", 1.0)
                    if active_is_pm
                    else 1.0
                )

                if active_signal.direction == "LONG":
                    if low_p <= active_stop_loss:
                        exit_price = active_stop_loss - self.slippage_points
                        exit_reason = (
                            "Break-Even Stop"
                            if be_moved
                            else ("Time-Decay Stop (-0.4R)" if decay_moved else "Stop Loss")
                        )
                    else:
                        if use_partial_tp and not tp1_done and rem_contracts >= 2 and high_p >= active_tp1_price:
                            leg1_qty = rem_contracts // 2
                            leg1_gross = active_tp1_price - active_entry_price
                            realized_leg1_pnl += (leg1_gross * point_val * leg1_qty) - (
                                2.0 * self.commission_per_side * leg1_qty
                            )
                            rem_contracts -= leg1_qty
                            tp1_done = True
                            if self.cfg.risk.breakeven_at_1r and not be_moved:
                                be_sl = round_to_tick(active_entry_price + self.cfg.risk.breakeven_offset_points)
                                if close_p > be_sl:
                                    active_stop_loss = max(active_stop_loss, be_sl)
                                    be_moved = True

                        if high_p >= active_take_profit:
                            exit_price = active_take_profit
                            exit_reason = "Take Profit"
                        elif self.risk.should_flatten_eod(ts):
                            exit_price = close_p - self.slippage_points
                            exit_reason = "EOD Flatten"
                        elif (
                            self.cfg.risk.breakeven_at_1r
                            and not be_moved
                            and high_p >= active_entry_price + be_r * one_r
                        ):
                            new_sl = round_to_tick(active_entry_price + self.cfg.risk.breakeven_offset_points)
                            if close_p > new_sl:
                                active_stop_loss = max(active_stop_loss, new_sl)
                                be_moved = True
                        elif (
                            self.cfg.risk.time_decay_bars > 0
                            and not be_moved
                            and not decay_moved
                            and bars_held >= self.cfg.risk.time_decay_bars
                        ):
                            decayed_stop = round_to_tick(
                                active_entry_price - self.cfg.risk.time_decay_stop_r * one_r
                            )
                            if low_p > decayed_stop and close_p > decayed_stop and decayed_stop > active_stop_loss:
                                active_stop_loss = decayed_stop
                                decay_moved = True
                else:  # SHORT
                    if high_p >= active_stop_loss:
                        exit_price = active_stop_loss + self.slippage_points
                        exit_reason = (
                            "Break-Even Stop"
                            if be_moved
                            else ("Time-Decay Stop (-0.4R)" if decay_moved else "Stop Loss")
                        )
                    else:
                        if use_partial_tp and not tp1_done and rem_contracts >= 2 and low_p <= active_tp1_price:
                            leg1_qty = rem_contracts // 2
                            leg1_gross = active_entry_price - active_tp1_price
                            realized_leg1_pnl += (leg1_gross * point_val * leg1_qty) - (
                                2.0 * self.commission_per_side * leg1_qty
                            )
                            rem_contracts -= leg1_qty
                            tp1_done = True
                            if self.cfg.risk.breakeven_at_1r and not be_moved:
                                be_sl = round_to_tick(active_entry_price - self.cfg.risk.breakeven_offset_points)
                                if close_p < be_sl:
                                    active_stop_loss = min(active_stop_loss, be_sl)
                                    be_moved = True

                        if low_p <= active_take_profit:
                            exit_price = active_take_profit
                            exit_reason = "Take Profit"
                        elif self.risk.should_flatten_eod(ts):
                            exit_price = close_p + self.slippage_points
                            exit_reason = "EOD Flatten"
                        elif (
                            self.cfg.risk.breakeven_at_1r
                            and not be_moved
                            and low_p <= active_entry_price - be_r * one_r
                        ):
                            new_sl = round_to_tick(active_entry_price - self.cfg.risk.breakeven_offset_points)
                            if close_p < new_sl:
                                active_stop_loss = min(active_stop_loss, new_sl)
                                be_moved = True
                        elif (
                            self.cfg.risk.time_decay_bars > 0
                            and not be_moved
                            and not decay_moved
                            and bars_held >= self.cfg.risk.time_decay_bars
                        ):
                            decayed_stop = round_to_tick(
                                active_entry_price + self.cfg.risk.time_decay_stop_r * one_r
                            )
                            if high_p < decayed_stop and close_p < decayed_stop and decayed_stop < active_stop_loss:
                                active_stop_loss = decayed_stop
                                decay_moved = True

                if exit_price is not None:
                    gross_pts = (
                        (exit_price - active_entry_price)
                        if active_signal.direction == "LONG"
                        else (active_entry_price - exit_price)
                    )
                    round_turn_comm = 2.0 * self.commission_per_side * rem_contracts
                    leg2_pnl = (gross_pts * point_val * rem_contracts) - round_turn_comm
                    net_pnl = realized_leg1_pnl + leg2_pnl
                    full_reason = f"TP1 (+{active_tp1_r:.1f}R) + {exit_reason}" if tp1_done else exit_reason

                    trades.append(
                        SimulatedTrade(
                            entry_time=active_signal.timestamp,
                            exit_time=ts,
                            direction=active_signal.direction,
                            reason=active_signal.reason,
                            contracts=active_contracts,
                            entry_price=round(active_entry_price, 2),
                            exit_price=round(exit_price, 2),
                            stop_loss=active_stop_loss,
                            take_profit=active_take_profit,
                            exit_reason=full_reason,
                            gross_points=round(gross_pts, 2),
                            net_pnl_usd=round(net_pnl, 2),
                        )
                    )
                    self.risk.record_trade_closed(net_pnl, exit_bar_index=i)
                    active_signal = None
                    active_contracts = 0
                    rem_contracts = 0
                    realized_leg1_pnl = 0.0
                    tp1_done = False
                continue

            # 2. Check if new trade is allowed when flat
            allowed, _ = self.risk.can_open_trade(ts, current_bar_index=i)
            if not allowed:
                continue

            signal = self.strategy.evaluate_bar(enriched, idx=i)
            if signal is not None:
                if self.use_limit_execution:
                    slip_lmt = self.cfg.risk.entry_limit_slippage_ticks * tick
                    collar = float(getattr(self.cfg.risk, "entry_collar_points", 0.0) or 0.0)
                    pending_signal = signal
                    pending_sig_idx = i
                    pending_best_fav = 0.0
                    if signal.direction == "LONG":
                        pending_lmt = round_to_tick(signal.entry_price + slip_lmt, tick)
                        pending_stp = round_to_tick(signal.entry_price - collar, tick) if collar > 0.0 else -1e12
                    else:
                        pending_lmt = round_to_tick(signal.entry_price - slip_lmt, tick)
                        pending_stp = round_to_tick(signal.entry_price + collar, tick) if collar > 0.0 else 1e12
                else:
                    instant_fill = (
                        signal.entry_price + self.slippage_points
                        if signal.direction == "LONG"
                        else signal.entry_price - self.slippage_points
                    )
                    _activate_position(signal, instant_fill, bar_idx=i, reanchor_sl_tp=False)

        metrics = self._compute_metrics(trades)
        return {
            "trades": trades,
            "metrics": metrics,
            "trades_df": pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame(),
        }

    @staticmethod
    def _compute_metrics(trades: List[SimulatedTrade]) -> Dict[str, float]:
        if not trades:
            return {
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "net_pnl_usd": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_usd": 0.0,
                "avg_trade_usd": 0.0,
                "sharpe_ratio": 0.0,
            }

        pnls = np.array([t.net_pnl_usd for t in trades], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        equity_curve = np.cumsum(pnls)
        peaks = np.maximum.accumulate(np.insert(equity_curve, 0, 0.0))[1:]
        drawdowns = peaks - equity_curve
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        std_pnl = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
        sharpe = (float(np.mean(pnls)) / std_pnl * np.sqrt(len(pnls))) if std_pnl > 0 else 0.0

        return {
            "total_trades": len(trades),
            "win_rate_pct": round(100.0 * len(wins) / len(trades), 2),
            "net_pnl_usd": round(float(pnls.sum()), 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_usd": round(max_dd, 2),
            "avg_trade_usd": round(float(np.mean(pnls)), 2),
            "sharpe_ratio": round(sharpe, 2),
        }
