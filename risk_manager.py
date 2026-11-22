"""
Intraday Risk & Session Guardrails for NQ / MNQ Futures Trading.
Enforces:
- RTH entry time window & mandatory End-of-Day (EOD) flatten check
- Daily max loss circuit breaker
- Max daily trades & max consecutive losses circuit breakers
- Post-exit bar cooldown
- Dynamic or fixed contract sizing
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import Optional, Tuple

import pandas as pd
import pytz

from config import ContractConfig, RiskConfig
from economic_calendar import is_fomc_decision_date


@dataclass
class DailyRiskState:
    """Tracks live or simulated intraday risk metrics for a single trading session (Pre-Market + RTH)."""
    session_date: Optional[date] = None
    realized_pnl_usd: float = 0.0
    premarket_realized_pnl_usd: float = 0.0
    rth_realized_pnl_usd: float = 0.0
    trades_taken: int = 0
    premarket_trades_taken: int = 0
    premarket_vrev_trades_taken: int = 0
    rth_trades_taken: int = 0
    rth_reset_done: bool = False
    in_premarket_phase: bool = False
    in_vrev_phase: bool = False
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    last_exit_bar_index: int = -999
    halted: bool = False
    halt_reason: str = ""
    account_halted: bool = False
    account_halt_reason: str = ""
    mr_b_trades_taken: int = 0
    mr_b_wins: int = 0
    mr_b_losses: int = 0
    mr_b_last_exit_bar_index: int = -999
    mr_b_realized_pnl_usd: float = 0.0


class RiskManager:
    """Validates whether new signals are allowed and calculates contract size."""

    def __init__(self, risk_cfg: RiskConfig, contract_cfg: ContractConfig, calendar=None) -> None:
        self.r_cfg = risk_cfg
        self.c_cfg = contract_cfg
        self.calendar = calendar
        self.tz = pytz.timezone(self.r_cfg.timezone)
        self.state = DailyRiskState()

        self._pm_start = self._parse_hhmm(getattr(self.r_cfg, "premarket_start_time", "04:00"))
        self._pm_trend_end = self._parse_hhmm(getattr(self.r_cfg, "premarket_trend_end_time", "06:55"))
        self._pm_end = self._parse_hhmm(getattr(self.r_cfg, "premarket_end_time", "08:20"))
        self._entry_start = self._parse_hhmm(self.r_cfg.entry_start_time)
        self._entry_end = self._parse_hhmm(self.r_cfg.entry_end_time)
        self._flatten_time = self._parse_hhmm(self.r_cfg.flatten_eod_time)
        blackouts = getattr(self.r_cfg, "rth_blackout_windows", ())
        self._rth_blackouts = [(self._parse_hhmm(s), self._parse_hhmm(e)) for s, e in blackouts]

        self._mr_b_start = self._parse_hhmm(getattr(self.r_cfg, "mr_b_entry_start_time", "10:15"))
        self._mr_b_end = self._parse_hhmm(getattr(self.r_cfg, "mr_b_entry_end_time", "14:30"))
        self._mr_b_fomc_b_start = self._parse_hhmm(getattr(self.r_cfg, "mr_b_fomc_blackout_start", "13:50"))
        self._mr_b_fomc_b_end = self._parse_hhmm(getattr(self.r_cfg, "mr_b_fomc_blackout_end", "14:45"))
        self._mr_b_fomc_f_start = self._parse_hhmm(getattr(self.r_cfg, "mr_b_fomc_flatten_start", "13:56"))
        self._mr_b_fomc_f_end = self._parse_hhmm(getattr(self.r_cfg, "mr_b_fomc_flatten_end", "14:00"))

    @staticmethod
    def _parse_hhmm(hhmm: str) -> time:
        parts = hhmm.strip().split(":")
        return time(int(parts[0]), int(parts[1]))

    def _to_session_dt(self, ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            return ts.tz_localize(self.tz)
        return ts.tz_convert(self.tz)

    def check_and_reset_session(self, ts: pd.Timestamp) -> None:
        """Reset daily counters on a new trading date and refresh RTH trade budget at 09:30 ET."""
        local_dt = self._to_session_dt(ts)
        curr_date = local_dt.date()
        t = local_dt.time()
        in_pm = t < time(9, 30)
        in_vrev = in_pm and (t > self._pm_trend_end)
        if self.state.session_date != curr_date:
            self.state = DailyRiskState(
                session_date=curr_date,
                rth_reset_done=(t >= time(9, 30)),
                in_premarket_phase=in_pm,
                in_vrev_phase=in_vrev,
            )
        else:
            self.state.in_premarket_phase = in_pm
            self.state.in_vrev_phase = in_vrev
            if not self.state.rth_reset_done and t >= time(9, 30):
                self.state.rth_reset_done = True
                self.state.consecutive_losses = 0
                # Unconditionally clear any Pre-Market halt so RTH starts with its own isolated daily loss budget
                self.state.halted = False
                self.state.halt_reason = ""
                self.state.account_halted = False
                self.state.account_halt_reason = ""

    def should_flatten_eod(self, ts: pd.Timestamp) -> bool:
        """Return True if current time is at Pre-Market close (09:24-09:29 ET) or EOD flatten time (15:55-17:00 ET)."""
        local_dt = self._to_session_dt(ts)
        t = local_dt.time()
        if getattr(self.r_cfg, "use_premarket_session", False) and (time(9, 24) <= t < time(9, 30)):
            return True
        return self._flatten_time <= t <= time(17, 0)

    def is_within_entry_window(self, ts: pd.Timestamp) -> bool:
        """Return True if current timestamp falls within allowed Pre-Market (when enabled) or RTH entry hours."""
        local_dt = self._to_session_dt(ts)
        t = local_dt.time()
        in_rth = self._entry_start <= t <= self._entry_end
        in_pm = getattr(self.r_cfg, "use_premarket_session", False) and (self._pm_start <= t <= self._pm_end)
        return in_rth or in_pm

    def can_open_trade(
        self,
        ts: pd.Timestamp,
        current_bar_index: int = 0,
    ) -> Tuple[bool, str]:
        """
        Check all risk rules before submitting a new bracket order.
        Returns (allowed: bool, reason: str).
        """
        self.check_and_reset_session(ts)

        if self.state.halted or self.state.account_halted:
            reason = self.state.halt_reason or self.state.account_halt_reason
            return False, f"Session halted: {reason}"

        if not self.is_within_entry_window(ts):
            return False, "Outside allowed RTH / Premarket entry time window"

        if not self.state.in_premarket_phase:
            local_t = self._to_session_dt(ts).time()
            for b_start, b_end in self._rth_blackouts:
                if b_start <= local_t <= b_end:
                    return (
                        False,
                        f"Inside RTH volatility blackout window ({b_start.strftime('%H:%M')}–{b_end.strftime('%H:%M')} ET)",
                    )

        if self.calendar is not None and getattr(self.r_cfg, "use_news_calendar_filter", True):
            in_news, news_reason = self.calendar.is_in_news_blackout(ts)
            if in_news:
                return False, news_reason

        active_session_pnl = (
            self.state.premarket_realized_pnl_usd
            if self.state.in_premarket_phase
            else self.state.rth_realized_pnl_usd
        )
        if active_session_pnl <= -abs(self.r_cfg.max_daily_loss_usd):
            session_label = "Pre-Market" if self.state.in_premarket_phase else "RTH"
            self.state.halted = True
            self.state.halt_reason = (
                f"Max {session_label} daily loss reached (${active_session_pnl:.2f} <= -${self.r_cfg.max_daily_loss_usd:.2f})"
            )
            self.state.account_halted = True
            self.state.account_halt_reason = self.state.halt_reason
            return False, self.state.halt_reason

        if self.state.in_premarket_phase:
            pm_max = getattr(self.r_cfg, "premarket_max_trades", 2)
            total_pm = self.state.premarket_trades_taken + self.state.premarket_vrev_trades_taken
            if total_pm >= pm_max:
                return False, f"Max pre-market trades reached ({total_pm}/{pm_max})"
            if self.state.in_vrev_phase and self.state.premarket_vrev_trades_taken >= 1:
                return False, "Max pre-market V-Reversal trades reached (1/1)"
        else:
            if self.state.rth_trades_taken >= self.r_cfg.max_daily_trades:
                return False, f"Max daily trades reached ({self.state.rth_trades_taken}/{self.r_cfg.max_daily_trades})"

        if self.state.consecutive_losses >= self.r_cfg.max_consecutive_losses:
            self.state.halted = True
            self.state.halt_reason = (
                f"Max consecutive losses reached ({self.state.consecutive_losses})"
            )
            return False, self.state.halt_reason

        bars_since_exit = current_bar_index - self.state.last_exit_bar_index
        if bars_since_exit < self.r_cfg.cooldown_bars_after_exit:
            return False, f"Cooldown active ({bars_since_exit}/{self.r_cfg.cooldown_bars_after_exit} bars)"

        return True, "OK"

    def is_fomc_day(self, ts: pd.Timestamp) -> bool:
        """Return True if `ts` falls on an FOMC decision date."""
        return is_fomc_decision_date(ts, timezone_name=self.r_cfg.timezone, calendar=self.calendar)

    def should_flatten_mr_b_fomc(self, ts: pd.Timestamp) -> bool:
        """Return True during the 13:56-14:00 ET pre-FOMC flatten window on FOMC decision days."""
        if not self.is_fomc_day(ts):
            return False
        local_t = self._to_session_dt(ts).time()
        return self._mr_b_fomc_f_start <= local_t <= self._mr_b_fomc_f_end

    def can_open_mr_b_trade(
        self,
        ts: pd.Timestamp,
        current_bar_index: int = 0,
    ) -> Tuple[bool, str]:
        """
        Check isolated MR-B daily guardrails plus the shared account RTH daily loss kill-switch.
        Deliberately ignores v4.2 consecutive-loss halts, cooldowns, and EU Close blackouts.
        """
        self.check_and_reset_session(ts)

        if self.state.account_halted:
            return False, f"Session account halted: {self.state.account_halt_reason}"

        local_dt = self._to_session_dt(ts)
        local_t = local_dt.time()
        if self.state.in_premarket_phase or not (self._mr_b_start <= local_t <= self._mr_b_end):
            return False, "Outside allowed MR-B entry window (10:15-14:30 ET)"

        if self.is_fomc_day(ts) and (self._mr_b_fomc_b_start <= local_t <= self._mr_b_fomc_b_end):
            return False, "Inside MR-B FOMC blackout window (13:50-14:45 ET)"

        if self.calendar is not None and getattr(self.r_cfg, "use_news_calendar_filter", True):
            in_news, news_reason = self.calendar.is_in_news_blackout(ts)
            if in_news:
                return False, news_reason

        if self.state.rth_realized_pnl_usd <= -abs(self.r_cfg.max_daily_loss_usd):
            reason = (
                f"Max RTH daily loss reached (${self.state.rth_realized_pnl_usd:.2f} <= "
                f"-${self.r_cfg.max_daily_loss_usd:.2f})"
            )
            self.state.account_halted = True
            self.state.account_halt_reason = reason
            self.state.halted = True
            self.state.halt_reason = reason
            return False, reason

        max_mr_losses = getattr(self.r_cfg, "mr_b_max_daily_losses", 1)
        if self.state.mr_b_losses >= max_mr_losses:
            return False, f"MR-B max daily losses reached ({self.state.mr_b_losses}/{max_mr_losses})"

        max_mr_trades = getattr(self.r_cfg, "mr_b_max_daily_trades", 3)
        if self.state.mr_b_trades_taken >= max_mr_trades:
            return False, f"MR-B max daily trades reached ({self.state.mr_b_trades_taken}/{max_mr_trades})"

        mr_cooldown = getattr(self.r_cfg, "mr_b_cooldown_bars", 3)
        bars_since_exit = current_bar_index - self.state.mr_b_last_exit_bar_index
        if bars_since_exit < mr_cooldown:
            return False, f"MR-B cooldown active ({bars_since_exit}/{mr_cooldown} bars)"

        return True, "OK"

    def calculate_mr_b_contracts(self) -> int:
        """Determine fixed contract quantity for MR-B trades."""
        mr_c = int(getattr(self.r_cfg, "mr_b_contracts", 0) or 0)
        base = mr_c if mr_c > 0 else self.r_cfg.fixed_contracts
        return max(1, min(base, self.r_cfg.max_contracts))

    def calculate_contracts(self, stop_distance_points: float) -> int:
        """
        Determine contract quantity based on fixed_contracts or dynamic dollar risk per trade.
        """
        if self.r_cfg.risk_per_trade_usd <= 0 or stop_distance_points <= 0:
            return min(self.r_cfg.fixed_contracts, self.r_cfg.max_contracts)

        dollar_risk_per_contract = stop_distance_points * self.c_cfg.point_value
        if dollar_risk_per_contract <= 0:
            return 1

        raw_qty = int(self.r_cfg.risk_per_trade_usd // dollar_risk_per_contract)
        return max(1, min(raw_qty, self.r_cfg.max_contracts))

    def record_trade_closed(
        self,
        net_pnl_usd: float,
        exit_bar_index: int = 0,
        strategy_id: str = "v4.2",
    ) -> None:
        """Update session risk state after a position is closed (isolating MR-B counters from v4.2)."""
        if strategy_id == "MR-B":
            self.state.mr_b_trades_taken += 1
            self.state.mr_b_realized_pnl_usd += net_pnl_usd
            self.state.rth_realized_pnl_usd += net_pnl_usd
            self.state.realized_pnl_usd += net_pnl_usd
            self.state.mr_b_last_exit_bar_index = exit_bar_index
            if net_pnl_usd < 0:
                self.state.mr_b_losses += 1
            else:
                self.state.mr_b_wins += 1

            if self.state.rth_realized_pnl_usd <= -abs(self.r_cfg.max_daily_loss_usd):
                reason = f"Max RTH daily loss breached (${self.state.rth_realized_pnl_usd:.2f})"
                self.state.account_halted = True
                self.state.account_halt_reason = reason
                self.state.halted = True
                self.state.halt_reason = reason
            return

        self.state.trades_taken += 1
        if self.state.in_premarket_phase:
            if self.state.in_vrev_phase:
                self.state.premarket_vrev_trades_taken += 1
            else:
                self.state.premarket_trades_taken += 1
            self.state.premarket_realized_pnl_usd += net_pnl_usd
        else:
            self.state.rth_trades_taken += 1
            self.state.rth_realized_pnl_usd += net_pnl_usd
        self.state.realized_pnl_usd += net_pnl_usd
        self.state.last_exit_bar_index = exit_bar_index

        if net_pnl_usd < 0:
            self.state.losses += 1
            self.state.consecutive_losses += 1
        else:
            self.state.wins += 1
            self.state.consecutive_losses = 0

        active_session_pnl = (
            self.state.premarket_realized_pnl_usd
            if self.state.in_premarket_phase
            else self.state.rth_realized_pnl_usd
        )
        if active_session_pnl <= -abs(self.r_cfg.max_daily_loss_usd):
            session_label = "Pre-Market" if self.state.in_premarket_phase else "RTH"
            self.state.halted = True
            self.state.halt_reason = (
                f"Max {session_label} daily loss breached (${active_session_pnl:.2f})"
            )
            self.state.account_halted = True
            self.state.account_halt_reason = self.state.halt_reason
        elif self.state.consecutive_losses >= self.r_cfg.max_consecutive_losses:
            self.state.halted = True
            self.state.halt_reason = (
                f"Max consecutive losses breached ({self.state.consecutive_losses})"
            )
