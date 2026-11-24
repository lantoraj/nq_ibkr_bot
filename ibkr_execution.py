"""
Interactive Brokers (TWS / IB Gateway) Execution & Live Streaming Engine using `ib_async`.
Handles:
- Connecting to TWS / IB Gateway (Paper or Live port)
- Resolving the active front-month NQ or MNQ futures contract on CME
- Streaming real-time historical + keepUpToDate bars
- Atomic placement of Server-Side Bracket Orders (Parent Limit/Market + Stop Loss + Take Profit)
- EOD position flattening & realized PnL tracking
"""
from __future__ import annotations

import io
import json
import logging
import threading
import urllib.request
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from ib_async import (
    IB,
    BarDataList,
    ContFuture,
    Contract,
    Future,
    LimitOrder,
    MarketOrder,
    Order,
    StopLimitOrder,
    StopOrder,
    Trade,
    util,
)
from ib_async.objects import UNSET_DOUBLE

from config import BotConfig
from economic_calendar import EconomicCalendar
from risk_manager import RiskManager
from strategy import NQIntradayStrategy, TradeSignal
from trade_journal import TradeJournal, load_journal
from tv_data_feed import TradingViewDataFeed

logger = logging.getLogger("nq_ibkr_bot")


def compute_pre_news_tightened_stop(
    direction: str,
    entry_ref: float,
    one_r: float,
    current_sl: float,
    close_p: float,
    be_offset_pts: float,
    decay_stop_r: float,
    round_fn=None,
) -> tuple[Optional[float], str]:
    """
    Compute tighter pre-news Stop Loss ahead of imminent USD High-Impact releases:
      - If current close profit >= +0.30R: move SL to Break-Even (+be_offset_pts).
      - Otherwise: tighten SL to -decay_stop_r (default -0.40R).
    Includes the +/-0.50 pt close-price safety guard against marketable fills.
    """
    if one_r <= 0:
        return None, ""
    r_round = round_fn if round_fn is not None else (lambda x: round(x * 4.0) / 4.0)

    if direction == "LONG":
        curr_r = (close_p - entry_ref) / one_r
        if curr_r >= 0.30:
            target = r_round(entry_ref + be_offset_pts)
            kind = "pre_news_be"
        else:
            target = r_round(entry_ref - decay_stop_r * one_r)
            kind = "pre_news_tighten"
        if close_p > target + 0.50 and target > current_sl:
            return float(target), kind
        return None, ""

    curr_r = (entry_ref - close_p) / one_r
    if curr_r >= 0.30:
        target = r_round(entry_ref - be_offset_pts)
        kind = "pre_news_be"
    else:
        target = r_round(entry_ref + decay_stop_r * one_r)
        kind = "pre_news_tighten"
    if close_p < target - 0.50 and target < current_sl:
        return float(target), kind
    return None, ""


class IBKRNQBotRunner:
    """Live / Paper execution engine for NQ & MNQ futures on Interactive Brokers."""

    def __init__(self, config: BotConfig, mode: str = "paper") -> None:
        self.cfg = config
        self.mode = mode.lower()
        self.ib = IB()
        self.strategy = NQIntradayStrategy(config.strategy, config.risk, config.contract)
        journal_root = Path("/app/journal") if Path("/app/journal").exists() else Path("journal")
        report_json = "/app/report/ff_calendar.json" if Path("/app/report").is_dir() else None
        self.calendar = EconomicCalendar(
            cache_path=str(journal_root / "ff_calendar_cache.json"),
            report_path=report_json,
            timezone_name=config.risk.timezone,
            tier1_pre_mins=getattr(config.risk, "news_mega_pre_blackout_mins", 10),
            tier1_post_mins=getattr(config.risk, "news_mega_post_blackout_mins", 10),
            tier2_pre_mins=getattr(config.risk, "news_pre_blackout_mins", 5),
            tier2_post_mins=getattr(config.risk, "news_post_blackout_mins", 5),
        )
        self.calendar.load_from_cache()
        self._calendar_brief_state_file = journal_root / "last_calendar_brief_date.txt"
        self._last_calendar_brief_date: Optional[str] = None
        if self._calendar_brief_state_file.exists():
            try:
                self._last_calendar_brief_date = self._calendar_brief_state_file.read_text(encoding="utf-8").strip() or None
            except Exception:
                self._last_calendar_brief_date = None
        self.risk = RiskManager(config.risk, config.contract, calendar=self.calendar)

        self.contract: Optional[Contract] = None
        self.bars: Optional[BarDataList | pd.DataFrame] = None
        self.bar_count: int = 0
        self._last_processed_bar_time: Optional[datetime] = None
        self.tv_feed: Optional[TradingViewDataFeed] = (
            TradingViewDataFeed(
                auth_token=getattr(config.ibkr, "tradingview_auth_token", ""),
                realtime_proxy_symbol=getattr(config.ibkr, "tradingview_realtime_proxy", "CAPITALCOM:US100"),
                tick_size=config.contract.tick_size,
            )
            if getattr(config.ibkr, "use_tradingview_data", True)
            else None
        )

        # Active bracket order tracking for +1R Break-Even, Time-Decay Stop, 2-Ct Scale-Out & Unfilled Timeout
        self._active_signal: Optional[TradeSignal] = None
        self._active_parent_order: Optional[Order] = None
        self._active_tp_order: Optional[Order] = None
        self._active_stop_order: Optional[Order] = None
        self._active_parent_orders: List[Order] = []
        self._active_tp_orders: List[Order] = []
        self._active_stop_orders: List[Order] = []
        self._active_tp1_order: Optional[Order] = None
        self._trade_initial_tp1: Optional[float] = None
        self._tp1_filled: bool = False
        self._entry_fill_handled: bool = False
        self._current_trade_filled_qty: int = 0
        self._current_trade_closed_qty: int = 0
        self._closing_exec_qty: int = 0
        self._last_exit_exec_utc: Optional[pd.Timestamp] = None
        self._current_trade_realized_pnl: float = 0.0
        self._current_trade_commissions: float = 0.0
        self._entry_submitted_bar: int = -1
        self._trade_filled_bar: int = -1
        self._post_fill_bars: int = 0
        self._trade_filled_ts_utc: Optional[pd.Timestamp] = None
        self._trade_closed_ts_utc: Optional[pd.Timestamp] = None
        self._be_moved: bool = False
        self._decay_moved: bool = False
        self._last_summary_date: Optional[date] = None
        self._trade_entry_price: Optional[float] = None
        self._trade_direction: Optional[str] = None
        self._trade_initial_sl: Optional[float] = None
        self._trade_initial_tp: Optional[float] = None
        self._pending_post_trade_chart: Optional[dict] = None
        # Chronological history of the live server-side stop level (initial / re-anchor / breakeven / time_decay)
        self._stop_history: list[dict] = []
        self._last_error_alert_ts: dict[int, float] = {}

        # Live-vs-Backtest reconciliation journal (append-only JSONL; never interrupts trading)
        self.journal = TradeJournal(timezone=config.risk.timezone)
        self._signal_bar_ts: Optional[pd.Timestamp] = None
        self._signal_price_orig: Optional[float] = None
        self._signal_submitted_utc: Optional[pd.Timestamp] = None

        # Dual-strategy MR-B regime & single-account PREEMPT_V42 coordination state
        self._active_strategy_id: str = "v4.2"
        self._daily_regime_date: Optional[date] = None
        self._daily_regime_last_attempt_utc: Optional[pd.Timestamp] = None
        self._daily_bull_50: Optional[bool] = None
        self._daily_prev_close: Optional[float] = None
        self._daily_ema50: Optional[float] = None
        self._deferred_closed_trades: dict[Any, dict] = {}
        self._order_strategy_map: dict[int, str] = {}

    def _sync_discord_webhook_profile(self) -> None:
        """Upload custom bot_icon.png and v4.1 name to the Discord webhook profile on startup."""
        import base64
        from pathlib import Path

        url = (self.cfg.ibkr.discord_webhook_url or "").strip()
        if not url:
            return
        icon_path = Path(__file__).resolve().parent / "assets" / "bot_icon.png"
        if not icon_path.exists():
            return
        try:
            b64_icon = base64.b64encode(icon_path.read_bytes()).decode("ascii")
            bot_name = getattr(self.cfg.ibkr, "discord_bot_name", "NQ Quant Bot (v4.1)")
            patch_body = json.dumps({"name": bot_name, "avatar": f"data:image/png;base64,{b64_icon}"}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=patch_body,
                headers={"Content-Type": "application/json", "User-Agent": "NQ-IBKR-Quant-Bot/4.1"},
                method="PATCH",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                wh_id = data.get("id")
                wh_av = data.get("avatar")
                if wh_id and wh_av:
                    self.cfg.ibkr.discord_avatar_url = f"https://cdn.discordapp.com/avatars/{wh_id}/{wh_av}.png"
                    logger.info("Synced Discord webhook profile: %s (%s)", bot_name, self.cfg.ibkr.discord_avatar_url)
        except Exception as exc:
            logger.debug("Discord webhook profile sync skipped: %s", exc)

    def _send_discord_alert(
        self,
        title: str,
        description: str,
        color: int = 0x3498DB,
        image_bytes: Optional[bytes] = None,
        blocking: bool = False,
    ) -> None:
        """Send a rich embed notification (with optional chart image) to Discord."""
        url = (self.cfg.ibkr.discord_webhook_url or "").strip()
        if not url:
            return

        bot_name = getattr(self.cfg.ibkr, "discord_bot_name", "NQ Quant Bot (v4.1)")
        avatar_url = getattr(self.cfg.ibkr, "discord_avatar_url", "https://raw.githubusercontent.com/lantoraj/nq_ibkr_bot/main/assets/bot_icon.png")

        embed_obj = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "footer": {
                "text": f"{bot_name} • {self.mode.upper()}",
                "icon_url": avatar_url,
            },
        }
        payload = {
            "username": bot_name,
            "avatar_url": avatar_url,
            "embeds": [embed_obj],
        }
        if image_bytes is not None:
            payload["embeds"][0]["image"] = {"url": "attachment://trade_chart.png"}

        def _post() -> None:
            try:
                if image_bytes is not None:
                    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
                    body = io.BytesIO()
                    body.write(f"--{boundary}\r\n".encode("utf-8"))
                    body.write(b'Content-Disposition: form-data; name="payload_json"\r\n')
                    body.write(b"Content-Type: application/json\r\n\r\n")
                    body.write(json.dumps(payload).encode("utf-8"))
                    body.write(b"\r\n")

                    body.write(f"--{boundary}\r\n".encode("utf-8"))
                    body.write(b'Content-Disposition: form-data; name="files[0]"; filename="trade_chart.png"\r\n')
                    body.write(b"Content-Type: image/png\r\n\r\n")
                    body.write(image_bytes)
                    body.write(b"\r\n")
                    body.write(f"--{boundary}--\r\n".encode("utf-8"))

                    post_data = body.getvalue()
                    content_type = f"multipart/form-data; boundary={boundary}"
                else:
                    post_data = json.dumps(payload).encode("utf-8")
                    content_type = "application/json"

                req = urllib.request.Request(
                    url,
                    data=post_data,
                    headers={
                        "Content-Type": content_type,
                        "User-Agent": "NQ-IBKR-Quant-Bot/4.1",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15):
                    pass
            except Exception as exc:
                logger.warning("Failed to send Discord alert: %s", exc)

        if blocking:
            _post()
        else:
            threading.Thread(target=_post, daemon=True).start()

    def _maybe_send_daily_calendar_briefing(self, now_utc: Optional[datetime] = None) -> bool:
        """
        Send the daily 09:00 AM Prague (Europe/Prague) ForexFactory economic calendar
        briefing with table PNG once per trading day (Mon–Fri) from primary client_id=101.
        """
        if getattr(self.cfg.ibkr, "client_id", 101) != 101:
            return False
        if not getattr(self.cfg.risk, "use_news_calendar_filter", True):
            return False

        try:
            ref_utc = now_utc or datetime.now(timezone.utc)
            prague_now = ref_utc.astimezone(self.calendar.cet_tz)
            if prague_now.weekday() >= 5:
                return False

            target_hhmm = getattr(self.cfg.risk, "daily_calendar_alert_time_cet", "09:00")
            if prague_now.strftime("%H:%M") < target_hhmm:
                return False

            today_str = prague_now.strftime("%Y-%m-%d")
            if self._last_calendar_brief_date == today_str:
                return False

            self.calendar.fetch_and_cache(force=True)
            events = self.calendar.get_events_for_cet_date(prague_now.date())
            title, desc = self.calendar.format_daily_discord_message(
                prague_now.date(),
                events,
                pre_news_tighten_mins=int(getattr(self.cfg.risk, "pre_news_tighten_stop_mins", 4)),
            )
            img_bytes = self.calendar.render_daily_calendar_image(prague_now.date(), events)
            color = 0xE74C3C if any(e.tier in (1, 2) for e in events) else 0x3498DB
            self._send_discord_alert(
                title=title,
                description=desc,
                color=color,
                image_bytes=img_bytes,
            )

            self._last_calendar_brief_date = today_str
            try:
                self._calendar_brief_state_file.parent.mkdir(parents=True, exist_ok=True)
                self._calendar_brief_state_file.write_text(today_str, encoding="utf-8")
            except Exception as exc:
                logger.debug("Could not persist daily calendar brief date: %s", exc)
            logger.info("Sent daily 09:00 CET ForexFactory Discord briefing for %s (%d events).", today_str, len(events))
            return True
        except Exception as exc:
            logger.warning("Failed sending daily calendar briefing: %s", exc)
            return False

    def _record_stop_level(self, price: Optional[float], kind: str, ts_utc: Optional[pd.Timestamp] = None) -> None:
        """Append a live stop level change to the per-trade history (used for chart rendering)."""
        if price is None:
            return
        try:
            self._stop_history.append({
                "ts": ts_utc if ts_utc is not None else pd.Timestamp.now(tz="UTC"),
                "price": float(price),
                "kind": kind,
            })
        except Exception:  # never interrupt trading for chart bookkeeping
            pass

    def _generate_trade_chart(
        self,
        entry_price: float,
        exit_price: float,
        sl_price: Optional[float],
        tp_price: Optional[float],
        direction: str,
        pnl: float,
        entry_bar_offset: int = 6,
        post_exit_bars: int = 0,
        pre_entry_bars: int = 15,
        title_prefix: Optional[str] = None,
        entry_ts_utc: Optional[pd.Timestamp] = None,
        exit_ts_utc: Optional[pd.Timestamp] = None,
        stop_history: Optional[list] = None,
        tp1_price: Optional[float] = None,
    ) -> Optional[bytes]:
        """Generate a clean dark-theme candlestick chart of the trade with surrounding context for Discord."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            if self.bars is None or len(self.bars) == 0:
                return None

            df = self._bars_to_dataframe(self.bars)
            if df.empty or len(df) < 5:
                return None

            enriched = self.strategy.prepare_dataframe(df)
            total_needed = max(40, pre_entry_bars + entry_bar_offset + post_exit_bars + 6)
            chart_df = enriched.tail(total_needed).copy().reset_index(drop=True)
            if len(chart_df) < 5:
                return None

            # Compute New York ET time string for each candle
            ts_utc_series = pd.to_datetime(chart_df["timestamp"], utc=True)
            ts_et = ts_utc_series.dt.tz_convert(self.cfg.risk.timezone)
            time_strs = ts_et.dt.strftime("%H:%M").tolist()
            chart_df["time_str"] = time_strs

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(11, 5.2), dpi=130)
            fig.patch.set_facecolor("#131722")
            ax.set_facecolor("#131722")

            # Candlesticks
            width = 0.6
            for i, row in chart_df.iterrows():
                o, h, l, c = row["open"], row["high"], row["low"], row["close"]
                color = "#26a69a" if c >= o else "#ef5350"
                ax.plot([i, i], [l, h], color=color, linewidth=1.1, zorder=2)
                b_bottom = min(o, c)
                b_height = max(abs(c - o), 0.25)
                rect = patches.Rectangle(
                    (i - width / 2, b_bottom), width, b_height,
                    facecolor=color, edgecolor=color, zorder=3
                )
                ax.add_patch(rect)

            # Technical indicator overlays
            x = np.arange(len(chart_df))
            if "vwap" in chart_df.columns:
                ax.plot(x, chart_df["vwap"], color="#f39c12", linestyle="--", linewidth=1.2, label="VWAP", alpha=0.9)
            if "ema_fast" in chart_df.columns:
                ax.plot(x, chart_df["ema_fast"], color="#3498db", linewidth=1.2, label="EMA(9)", alpha=0.9)
            if "ema_slow" in chart_df.columns:
                ax.plot(x, chart_df["ema_slow"], color="#9b59b6", linewidth=1.2, label="EMA(21)", alpha=0.9)
            if "ema_trend" in chart_df.columns:
                ax.plot(x, chart_df["ema_trend"], color="#00d2d3", linestyle="-.", linewidth=1.2, label="EMA(50)", alpha=0.9)

            # Determine entry and exit x indices: the 2-min bar that CONTAINS the fill timestamp
            def _containing_idx(ts_val) -> int:
                t = pd.Timestamp(ts_val)
                t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
                return int(max(0, min(len(chart_df) - 1, int(ts_utc_series.searchsorted(t, side="right")) - 1)))

            if exit_ts_utc is not None and ts_utc_series.iloc[-1] >= pd.Timestamp(exit_ts_utc) - pd.Timedelta(minutes=2):
                exit_idx = _containing_idx(exit_ts_utc)
            else:
                exit_idx = max(0, len(chart_df) - 1 - post_exit_bars)

            if entry_ts_utc is not None and ts_utc_series.iloc[0] <= entry_ts_utc <= ts_utc_series.iloc[-1] + pd.Timedelta(minutes=2):
                entry_idx = _containing_idx(entry_ts_utc)
            else:
                entry_idx = max(0, exit_idx - max(1, entry_bar_offset))

            if entry_idx >= exit_idx and exit_idx > 0:
                entry_idx = max(0, exit_idx - 1)

            entry_time = time_strs[entry_idx]
            exit_time = time_strs[exit_idx]

            # ------------------------------------------------------------------
            # Trade level rendering: Entry, TP, Initial SL, live SL moves, Exit
            # ------------------------------------------------------------------
            n_bars = len(chart_df)
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            is_long = direction != "SHORT"

            def _ts_to_idx(ts) -> int:
                """Index of the 2-min bar that contains timestamp ts (clamped to chart)."""
                try:
                    t = pd.Timestamp(ts)
                    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
                    pos = int(ts_utc_series.searchsorted(t, side="right")) - 1
                    return int(min(max(pos, entry_idx), exit_idx))
                except Exception:
                    return entry_idx

            # Build stop segments from the live stop history (fallback: single initial SL)
            hist = sorted(
                [h for h in (stop_history or []) if h.get("price") is not None],
                key=lambda h: pd.Timestamp(h["ts"]),
            )
            if not hist and sl_price is not None:
                hist = [{"ts": entry_ts_utc, "price": float(sl_price), "kind": "initial"}]
            initial_sl = float(hist[0]["price"]) if hist else sl_price

            stop_style = {
                "initial": ("#ff5252", "--", "Initial SL"),
                "re_anchor": ("#ff5252", "--", "Re-anchored SL"),
                "breakeven": ("#ffd600", "-", "BE SL"),
                "time_decay": ("#ff9800", "-", "Time-Decay SL"),
            }
            seg_end_x = exit_idx + 0.45
            for k, h in enumerate(hist):
                x0 = entry_idx if k == 0 else _ts_to_idx(h["ts"]) + 0.5
                x1 = (_ts_to_idx(hist[k + 1]["ts"]) + 0.5) if k + 1 < len(hist) else seg_end_x
                color, ls, _ = stop_style.get(h["kind"], stop_style["initial"])
                ax.hlines(h["price"], xmin=x0, xmax=max(x0 + 0.2, x1), colors=color, linestyles=ls,
                          linewidth=2.0 if k > 0 else 1.5, zorder=4)
                if k > 0:
                    prev = hist[k - 1]["price"]
                    ax.vlines(x0, min(prev, h["price"]), max(prev, h["price"]), colors=color,
                              linestyles=":", linewidth=1.3, zorder=4)
                    move_et = pd.Timestamp(h["ts"]).tz_convert(self.cfg.risk.timezone).strftime("%H:%M")
                    lbl = "BE +lock" if h["kind"] == "breakeven" else (
                        f"-{self.cfg.risk.time_decay_stop_r:.2f}R decay" if h["kind"] == "time_decay" else h["kind"])
                    ax.annotate(
                        f"SL moved {move_et}\n{prev:.2f} → {h['price']:.2f} ({lbl})",
                        xy=(x0, h["price"]),
                        xytext=(0, -26 if is_long else 26), textcoords="offset points",
                        ha="center", va="top" if is_long else "bottom", fontsize=7.5, fontweight="bold",
                        color="#131722", zorder=7, annotation_clip=False,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=color, alpha=0.95, edgecolor="none"),
                        arrowprops=dict(arrowstyle="-", color=color, lw=1.0),
                    )

            # Take-profit line(s)
            if tp1_price is not None:
                ax.hlines(tp1_price, xmin=entry_idx, xmax=seg_end_x, colors="#26c6da", linestyles="--",
                          linewidth=1.35, zorder=4)
            if tp_price is not None:
                ax.hlines(tp_price, xmin=entry_idx, xmax=seg_end_x, colors="#00e676", linestyles="--",
                          linewidth=1.5, zorder=4)

            # Entry level: bold solid line during the trade + faint extension to the price axis
            entry_line_color = "#2979ff"
            ax.hlines(entry_price, xmin=entry_idx - 0.45, xmax=seg_end_x, colors=entry_line_color,
                      linestyles="-", linewidth=2.4, zorder=5)
            ax.hlines(entry_price, xmin=seg_end_x, xmax=n_bars - 0.5, colors=entry_line_color,
                      linestyles=":", linewidth=1.0, alpha=0.7, zorder=4)
            ax.scatter([entry_idx], [entry_price], marker="^" if is_long else "v", s=170,
                       color=entry_line_color, edgecolors="white", linewidths=1.3, zorder=8)

            # Exit marker
            exit_color = "#00e676" if pnl >= 0 else "#ff5252"
            ax.scatter([exit_idx], [exit_price], marker="X", s=150, color=exit_color,
                       edgecolors="white", linewidths=1.2, zorder=8)

            # Trade window shading
            ax.axvspan(entry_idx - 0.5, exit_idx + 0.5, color="#2979ff", alpha=0.06, zorder=1)

            # Y limits (include every level so nothing is clipped)
            levels = [p for p in (tp_price, tp1_price, entry_price, exit_price, initial_sl) if p is not None]
            levels += [h["price"] for h in hist]
            y_lo = min(float(chart_df["low"].min()), min(levels))
            y_hi = max(float(chart_df["high"].max()), max(levels))
            pad = max(8.0, (y_hi - y_lo) * 0.08)
            y_min, y_max = y_lo - pad, y_hi + pad
            ax.set_ylim(y_min, y_max)

            # Right-hand price tags (TradingView style) with collision avoidance
            tags = []
            if tp1_price is not None:
                tags.append((tp1_price, f"TP1  {tp1_price:.2f}", "#26c6da", "#131722"))
            if tp_price is not None:
                tp_lbl = "TP2  " if tp1_price is not None else "TP  "
                tags.append((tp_price, f"{tp_lbl}{tp_price:.2f}", "#00e676", "#131722"))
            tags.append((entry_price, f"ENTRY {direction}  {entry_price:.2f}  @{entry_time}", entry_line_color, "#ffffff"))
            if initial_sl is not None:
                tags.append((initial_sl, f"INITIAL SL  {initial_sl:.2f}", "#ff5252", "#ffffff"))
            if len(hist) > 1:
                last = hist[-1]
                c, _, name = stop_style.get(last["kind"], stop_style["initial"])
                tags.append((last["price"], f"{name.upper()}  {last['price']:.2f}", c, "#131722"))
            tags.append((exit_price, f"EXIT  {exit_price:.2f}  @{exit_time}  ({pnl_str})", exit_color,
                         "#131722" if pnl >= 0 else "#ffffff"))

            min_gap = (y_max - y_min) * 0.058
            tags.sort(key=lambda t: t[0])
            placed = []
            for price, *_ in tags:
                y = price if not placed else max(price, placed[-1] + min_gap)
                placed.append(y)
            overflow = placed[-1] - (y_max - min_gap * 0.5) if placed else 0
            if overflow > 0:  # shift block down if it runs off the top
                placed = [max(y - overflow, y_min + min_gap * 0.5) for y in placed]
                for i in range(1, len(placed)):
                    placed[i] = max(placed[i], placed[i - 1] + min_gap)

            tag_x = n_bars + 0.2
            for (price, text, bg, fg), y in zip(tags, placed):
                ax.annotate(
                    text, xy=(n_bars - 0.3, price), xytext=(tag_x, y), textcoords="data",
                    ha="left", va="center", fontsize=8.3, fontweight="bold", color=fg, zorder=9,
                    annotation_clip=False,
                    bbox=dict(boxstyle="round,pad=0.28", facecolor=bg, alpha=0.97, edgecolor="none"),
                    arrowprops=dict(arrowstyle="-", color=bg, lw=1.0, shrinkA=0, shrinkB=0),
                )

            # Horizontal Axis: Exact time labels
            tick_step = max(1, len(chart_df) // 8)
            tick_indices = list(range(0, len(chart_df), tick_step))
            if (len(chart_df) - 1) not in tick_indices:
                tick_indices.append(len(chart_df) - 1)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([time_strs[i] for i in tick_indices], rotation=0, fontsize=8, color="#d1d4dc")
            ax.set_xlabel("Time (ET)", color="#787b86", fontsize=8.5)

            ax.grid(True, linestyle=":", alpha=0.22, color="#787b86")
            ax.set_ylabel("Price (USD)", color="#d1d4dc", fontsize=9)
            ax.tick_params(colors="#787b86", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#2a2e39")

            sym = self.contract.localSymbol if self.contract else self.cfg.contract.symbol
            title_color = "#2ecc71" if pnl >= 0 else "#e74c3c"
            if title_prefix:
                full_title = f"{sym} (2 min) — {title_prefix} [{entry_time} → {exit_time} ET] ({pnl_str} USD)"
            elif post_exit_bars > 0:
                full_title = f"{sym} (2 min) — {direction} Trade [{entry_time} → {exit_time} ET] ({pnl_str} USD) (+{post_exit_bars} Bars Context)"
            else:
                full_title = f"{sym} (2 min) — {direction} Trade Closed [{entry_time} → {exit_time} ET] ({pnl_str} USD)"
            ax.set_title(full_title, color=title_color, fontsize=11, fontweight="bold", pad=10)
            ax.legend(loc="upper left", framealpha=0.4, facecolor="#1e222d", edgecolor="none", fontsize=8)

            ax.set_xlim(-1, n_bars - 0.3)

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=130)
            plt.close(fig)
            return buf.getvalue()
        except Exception as exc:
            logger.warning("Could not generate trade chart image: %s", exc)
            return None

    def _dispatch_post_trade_review_chart(self, task: dict) -> None:
        """Generate and post the 10-bar delayed post-trade review chart to Discord."""
        try:
            chart_bytes = self._generate_trade_chart(
                entry_price=task["entry_price"],
                exit_price=task["exit_price"],
                sl_price=task["sl_price"],
                tp_price=task["tp_price"],
                direction=task["direction"],
                pnl=task["pnl"],
                entry_bar_offset=task.get("trade_duration_bars", 6),
                post_exit_bars=10,
                pre_entry_bars=15,
                title_prefix=f"{task['direction']} Post-Trade Review (+10 Bars)",
                entry_ts_utc=task.get("entry_ts_utc"),
                exit_ts_utc=task.get("exit_ts_utc"),
                stop_history=task.get("stop_history"),
                tp1_price=task.get("tp1_price"),
            )
            sym = self.contract.localSymbol if self.contract else self.cfg.contract.symbol
            pnl = task["pnl"]
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            color = 0x2ECC71 if pnl >= 0 else 0xE74C3C
            self._send_discord_alert(
                title=f"📊 POST-TRADE REVIEW (10 BARS POST-EXIT) — {sym} ({pnl_str} USD)",
                description=(
                    f"**Trade Realized PnL:** **`{pnl_str} USD`**\n"
                    f"**Execution:** `{task['direction']}` @ `{task['entry_price']:.2f}` → Exit @ `{task['exit_price']:.2f}`\n"
                    f"**Surrounding Context:** `15 bars prior to entry` | `10 bars following exit`\n"
                    f"_Review market structure, liquidity reaction, and continuation after trade close._"
                ),
                color=color,
                image_bytes=chart_bytes,
            )
            logger.info("Dispatched 10-bar post-trade review chart to Discord for trade PnL $%.2f", pnl)
        except Exception as exc:
            logger.warning("Error generating/dispatching post-trade review chart: %s", exc)

    def connect(self, max_retries: int = 12, retry_wait_sec: int = 10) -> None:
        """Connect to IBKR TWS or IB Gateway (with automatic retries for Docker IB Gateway bootup)."""
        import time

        port = self.cfg.ibkr.paper_port if self.mode == "paper" else self.cfg.ibkr.live_port
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Connecting to IBKR (%s mode) at %s:%d (clientId=%d, attempt %d/%d)...",
                    self.mode.upper(),
                    self.cfg.ibkr.host,
                    port,
                    self.cfg.ibkr.client_id,
                    attempt,
                    max_retries,
                )
                self.ib.connect(
                    host=self.cfg.ibkr.host,
                    port=port,
                    clientId=self.cfg.ibkr.client_id,
                    timeout=self.cfg.ibkr.timeout,
                    account=self.cfg.ibkr.account or "",
                )
                logger.info("Connected to IBKR. Managed accounts: %s", self.ib.managedAccounts())
                return
            except Exception as exc:
                if attempt >= max_retries:
                    self._send_discord_alert(
                        title=f"🚨 BOT CONNECTION ERROR — IBKR API Unreachable ({self.mode.upper()})",
                        description=(
                            f"**Host / Port:** `{self.cfg.ibkr.host}:{port}` (`clientId={self.cfg.ibkr.client_id}`)\n"
                            f"**Failed Attempts:** `{attempt}/{max_retries}`\n"
                            f"**Error Details:** `{exc}`\n"
                            f"_Bot process is stopping; Docker & IB Gateway Watchdog will attempt automatic recovery._"
                        ),
                        color=0xE74C3C,
                        blocking=True,
                    )
                    raise
                logger.warning(
                    "IBKR connection attempt %d/%d failed (%s). Retrying in %ds...",
                    attempt,
                    max_retries,
                    exc,
                    retry_wait_sec,
                )
                time.sleep(retry_wait_sec)

    def resolve_front_month_contract(self) -> Contract:
        """
        Qualify the NQ or MNQ futures contract.
        If `contract_month` is empty, automatically fetches all non-expired contracts
        from IBKR and selects the nearest upcoming quarterly expiry.
        """
        c_cfg = self.cfg.contract
        if c_cfg.contract_month:
            fut = Future(
                symbol=c_cfg.symbol,
                lastTradeDateOrContractMonth=c_cfg.contract_month,
                exchange=c_cfg.exchange,
                currency=c_cfg.currency,
            )
            qualified = self.ib.qualifyContracts(fut)
            if not qualified:
                raise RuntimeError(f"Could not qualify contract {c_cfg.symbol} {c_cfg.contract_month}")
            self.contract = qualified[0]
        else:
            probe = Future(symbol=c_cfg.symbol, exchange=c_cfg.exchange, currency=c_cfg.currency)
            details = self.ib.reqContractDetails(probe)
            if not details:
                raise RuntimeError(f"No contract details returned by IBKR for {c_cfg.symbol} on {c_cfg.exchange}")

            today_str = datetime.now().strftime("%Y%m%d")
            valid_contracts = [
                d.contract
                for d in details
                if d.contract.lastTradeDateOrContractMonth >= today_str
            ]
            valid_contracts.sort(key=lambda c: c.lastTradeDateOrContractMonth)
            if not valid_contracts:
                raise RuntimeError(f"No active non-expired contracts found for {c_cfg.symbol}")

            self.contract = valid_contracts[0]

        logger.info(
            "Qualified Active Contract: %s (localSymbol=%s, expiry=%s, conId=%s)",
            self.contract.symbol,
            self.contract.localSymbol,
            self.contract.lastTradeDateOrContractMonth,
            self.contract.conId,
        )
        return self.contract

    def has_open_position_or_orders(self) -> bool:
        """Check if there is an active position or open parent order on the target contract."""
        if self.contract is None:
            return False

        for pos in self.ib.positions():
            if pos.contract.conId == self.contract.conId and pos.position != 0:
                return True

        for trade in self.ib.openTrades():
            if trade.contract.conId == self.contract.conId:
                return True

        return False

    def flatten_all(
        self,
        reason: str = "Manual/EOD Flatten",
        order_ref: Optional[str] = None,
    ) -> List[Order]:
        """Cancel all open orders for the contract and close any open position with a MarketOrder."""
        submitted: List[Order] = []
        if self.contract is None:
            return submitted

        logger.warning("Flattening all orders and positions for %s (%s)", self.contract.localSymbol, reason)

        # 1. Cancel open orders on this contract
        for trade in self.ib.openTrades():
            if (
                trade.contract.conId == self.contract.conId
                and getattr(getattr(trade, "orderStatus", None), "status", "")
                not in ("Cancelled", "ApiCancelled", "Filled", "Inactive")
            ):
                self.ib.cancelOrder(trade.order)

        # 2. Close net position if non-zero
        for pos in self.ib.positions():
            if pos.contract.conId == self.contract.conId and pos.position != 0:
                action = "SELL" if pos.position > 0 else "BUY"
                qty = abs(int(pos.position))
                close_order = MarketOrder(
                    action=action,
                    totalQuantity=qty,
                    tif="DAY",
                    outsideRth=not self.cfg.risk.rth_only,
                )
                if order_ref:
                    close_order.orderRef = order_ref
                get_req = getattr(getattr(self.ib, "client", None), "getReqId", None)
                if callable(get_req) and getattr(close_order, "orderId", 0) == 0:
                    try:
                        close_order.orderId = int(get_req())
                    except Exception:
                        pass
                tr = self.ib.placeOrder(self.contract, close_order)
                oid = int(
                    getattr(close_order, "orderId", 0)
                    or getattr(getattr(tr, "order", None), "orderId", 0)
                    or 0
                )
                if oid and order_ref:
                    self._order_strategy_map[oid] = order_ref.split(":", 1)[0]
                submitted.append(close_order)
                logger.warning("Submitted %s %d %s MarketOrder to flatten position.", action, qty, self.contract.localSymbol)
        return submitted

    def _wait_until_flat(self, timeout_sec: float = 8.0, poll_sec: float = 0.25) -> bool:
        """Poll IBKR positions & open orders until contract is completely flat or timeout expires."""
        import time

        if self.contract is None:
            return True

        terminal = ("Cancelled", "ApiCancelled", "Filled", "Inactive")

        def _is_flat_now() -> bool:
            live_pos = any(
                p.contract.conId == self.contract.conId and p.position != 0
                for p in self.ib.positions()
            )
            if live_pos:
                return False
            active_orders = [
                t for t in self.ib.openTrades()
                if t.contract.conId == self.contract.conId
                and getattr(getattr(t, "orderStatus", None), "status", "") not in terminal
            ]
            return len(active_orders) == 0

        if _is_flat_now():
            return True

        try:
            is_conn = bool(self.ib.isConnected())
        except Exception:
            is_conn = False
        if not is_conn:
            return _is_flat_now()

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self.ib.sleep(poll_sec)
            if _is_flat_now():
                return True
        return _is_flat_now()

    def _reset_active_trade_state(self) -> None:
        """Clear in-memory active trade & order references cleanly."""
        self._current_trade_realized_pnl = 0.0
        self._current_trade_commissions = 0.0
        self._current_trade_closed_qty = 0
        self._current_trade_filled_qty = 0
        self._closing_exec_qty = 0
        self._last_exit_exec_utc = None
        self._active_signal = None
        self._active_strategy_id = "v4.2"
        self._active_parent_order = None
        self._active_parent_orders = []
        self._active_tp_orders = []
        self._active_stop_orders = []
        self._active_tp1_order = None
        self._active_tp_order = None
        self._active_stop_order = None
        self._trade_entry_price = None
        self._trade_direction = None
        self._trade_initial_sl = None
        self._trade_initial_tp1 = None
        self._trade_initial_tp = None
        self._entry_submitted_bar = -1
        self._trade_filled_bar = -1
        self._post_fill_bars = 0
        self._trade_filled_ts_utc = None
        self._trade_closed_ts_utc = None
        self._signal_bar_ts = None
        self._signal_price_orig = None
        self._signal_submitted_utc = None
        self._be_moved = False
        self._decay_moved = False
        self._tp1_filled = False
        self._entry_fill_handled = False
        self._stop_history = []

    def _extract_daily_closes_from_bars(self, raw_bars: Any, today_et: date) -> pd.Series:
        """Parse IBKR BarDataList or TradingView DataFrame of 1-day bars into completed-day close Series."""
        if raw_bars is None:
            return pd.Series(dtype=float)
        if isinstance(raw_bars, pd.DataFrame):
            df_d = raw_bars.copy()
        else:
            df_d = util.df(raw_bars)
        if df_d is None or df_d.empty:
            return pd.Series(dtype=float)
        if "timestamp" not in df_d.columns and "date" in df_d.columns:
            df_d = df_d.rename(columns={"date": "timestamp"})
        if "timestamp" not in df_d.columns or "close" not in df_d.columns:
            return pd.Series(dtype=float)

        import pytz

        tz = pytz.timezone(self.cfg.risk.timezone)
        parsed: dict[date, float] = {}
        for _, row in df_d.iterrows():
            ts_val = row["timestamp"]
            c_val = float(row["close"])
            if pd.isna(c_val):
                continue
            if isinstance(ts_val, datetime):
                if ts_val.tzinfo is not None:
                    d_et = ts_val.astimezone(tz).date()
                else:
                    d_et = ts_val.date()
            elif isinstance(ts_val, date):
                d_et = ts_val
            else:
                ts_pd = pd.Timestamp(ts_val)
                if pd.isna(ts_pd):
                    continue
                if ts_pd.tzinfo is not None and not (ts_pd.hour == 0 and ts_pd.minute == 0 and ts_pd.second == 0):
                    d_et = ts_pd.tz_convert(tz).date()
                else:
                    d_et = ts_pd.date()
            if d_et < today_et:
                parsed[d_et] = c_val

        if not parsed:
            return pd.Series(dtype=float)
        idx_dates = sorted(parsed.keys())
        return pd.Series([parsed[d] for d in idx_dates], index=idx_dates, dtype=float)

    def _refresh_daily_regime(self, ref_ts: Optional[pd.Timestamp] = None) -> bool:
        """
        Compute causal daily bull_50 regime (`prev_close > EMA50_d`) for MR-B using
        completed daily RTH bars strictly before today's ET trading session.
        Fails closed (`self._daily_bull_50 = False`) if daily history is unavailable.
        """
        if not self.cfg.strategy.enable_mr_b:
            return False

        ts_now = ref_ts if ref_ts is not None else pd.Timestamp.now(tz="UTC")
        today_et = self.risk._to_session_dt(ts_now).date()
        if self._daily_regime_date == today_et and self._daily_bull_50 is not None:
            return True

        now_utc = pd.Timestamp.now(tz="UTC")
        if (
            self._daily_regime_last_attempt_utc is not None
            and (now_utc - self._daily_regime_last_attempt_utc).total_seconds() < 300.0
        ):
            return False
        self._daily_regime_last_attempt_utc = now_utc

        ema_period = int(getattr(self.cfg.strategy, "mr_b_daily_ema_period", 50) or 50)
        completed_closes = pd.Series(dtype=float)

        try:
            is_conn = bool(self.ib.isConnected())
        except Exception:
            is_conn = False

        if is_conn:
            try:
                cont = ContFuture(
                    symbol=self.cfg.contract.symbol,
                    exchange=self.cfg.contract.exchange,
                    currency=self.cfg.contract.currency,
                )
                qualified = self.ib.qualifyContracts(cont)
                target_contract = qualified[0] if qualified else self.contract
                if target_contract is not None:
                    bars_1d = self.ib.reqHistoricalData(
                        target_contract,
                        endDateTime="",
                        durationStr="6 M",
                        barSizeSetting="1 day",
                        whatToShow="TRADES",
                        useRTH=True,
                        formatDate=1,
                        keepUpToDate=False,
                        timeout=30,
                    )
                    completed_closes = self._extract_daily_closes_from_bars(bars_1d, today_et)
            except Exception as exc:
                logger.warning("IBKR daily history fetch failed for MR-B EMA%d regime: %s", ema_period, exc)

        if len(completed_closes) < ema_period and self.tv_feed is not None:
            try:
                tv_daily = self.tv_feed.fetch_daily_bars(
                    base_symbol=self.cfg.contract.symbol,
                    n_bars=max(150, ema_period + 40),
                )
                tv_closes = self._extract_daily_closes_from_bars(tv_daily, today_et)
                if len(tv_closes) >= len(completed_closes):
                    completed_closes = tv_closes
            except Exception as exc:
                logger.warning("TradingView daily history fallback failed for MR-B EMA%d regime: %s", ema_period, exc)

        if len(completed_closes) < ema_period:
            self._daily_bull_50 = False
            logger.warning(
                "MR-B daily EMA%d regime unavailable (completed days=%d < %d); failing closed (bull_50=False).",
                ema_period,
                len(completed_closes),
                ema_period,
            )
            return False

        prev_close = float(completed_closes.iloc[-1])
        ema50_d = float(completed_closes.ewm(span=ema_period, adjust=False).mean().iloc[-1])
        self._daily_prev_close = round(prev_close, 2)
        self._daily_ema50 = round(ema50_d, 2)
        self._daily_bull_50 = bool(prev_close > ema50_d)
        self._daily_regime_date = today_et
        logger.info(
            "MR-B Daily Regime [%s] | PrevClose=%.2f vs EMA%d=%.2f -> bull_50=%s (%d completed days)",
            today_et,
            prev_close,
            ema_period,
            ema50_d,
            self._daily_bull_50,
            len(completed_closes),
        )
        return True

    def _cancel_unfilled_entry_orders(
        self,
        reason: str,
        last_price: float,
        trigger: str,
        elapsed_sec: Optional[float] = None,
        runaway_pts: Optional[float] = None,
    ) -> bool:
        """
        Cancel working unfilled parent/child entry orders WITHOUT ever placing a MarketOrder.
        Refuses to cancel if any filled qty or live IBKR position exists on the contract.
        """
        if self.contract is None or self._active_parent_order is None or self._active_signal is None:
            return False
        if self._current_trade_filled_qty > 0 or self._entry_fill_handled:
            return False
        for pos in self.ib.positions():
            if pos.contract.conId == self.contract.conId and pos.position != 0:
                return False

        terminal = ("Cancelled", "ApiCancelled", "Filled", "Inactive")
        for trade in self.ib.openTrades():
            if (
                trade.contract.conId == self.contract.conId
                and getattr(getattr(trade, "orderStatus", None), "status", "") not in terminal
            ):
                self.ib.cancelOrder(trade.order)

        cancelled_sig = self._active_signal
        cancelled_strat = self._active_strategy_id
        bars_waited = max(0, self.bar_count - self._entry_submitted_bar) if self._entry_submitted_bar >= 0 else 0
        self._active_parent_order = None
        self._active_parent_orders = []
        self._active_tp_orders = []
        self._active_stop_orders = []
        self._active_tp1_order = None
        self._active_tp_order = None
        self._active_stop_order = None
        self._active_signal = None
        self._active_strategy_id = "v4.2"

        logger.warning(
            "Cancelled unfilled [%s] %s %s entry @ %.2f (%s | last=%.2f | bars=%d)",
            cancelled_strat,
            cancelled_sig.direction,
            self.contract.localSymbol,
            cancelled_sig.entry_price,
            reason,
            last_price,
            bars_waited,
        )
        self.journal.log(
            "entry_cancelled",
            bar_ts=self._signal_bar_ts,
            strategy_id=cancelled_strat,
            direction=cancelled_sig.direction,
            signal_price=self._signal_price_orig,
            last_close=last_price,
            bars_waited=bars_waited,
            cancel_trigger=trigger,
            elapsed_sec=elapsed_sec,
            runaway_pts=runaway_pts,
            reason=reason,
        )
        strat_badge = f"[{cancelled_strat}] " if cancelled_strat == "MR-B" else ""
        self._send_discord_alert(
            title=f"⚪ {strat_badge}ENTRY CANCELLED (NOT FILLED) — {self.contract.localSymbol} {cancelled_sig.direction}",
            description=(
                f"**Strategy:** `{cancelled_strat}` | **Trigger:** `{reason}`\n"
                f"Limit entry @ `{cancelled_sig.entry_price:.2f}` was cancelled while flat (last price `{last_price:.2f}`).\n"
                f"**No position was opened — PnL `$0.00`.**"
            ),
            color=0x95A5A6,
        )
        return True

    def _is_signal_bar_stale(self, processed_lag_sec: float) -> tuple[bool, str]:
        """Block fresh signal submission when live feed lag stalls between max_signal_bar_lag_sec and 300s."""
        max_lag = float(getattr(self.cfg.risk, "max_signal_bar_lag_sec", 0.0) or 0.0)
        if max_lag > 0.0 and max_lag < processed_lag_sec < 300.0:
            return True, f"Stale signal bar lag ({processed_lag_sec:.1f}s > {max_lag:.0f}s)"
        return False, ""

    def _check_pending_entry_guards(self, raw_bars: BarDataList | pd.DataFrame) -> bool:
        """
        Evaluate real-time sub-bar / inter-poll P1 guards on an unfilled working parent order:
        1) Runaway Price Cancel: market ran >= `runaway_cancel_points` in signal direction without filling us
        2) Seconds Timeout: unfilled entry older than `cancel_unfilled_entry_seconds`
        """
        if (
            self.contract is None
            or self._active_parent_order is None
            or self._active_signal is None
            or self._current_trade_filled_qty > 0
            or self._entry_fill_handled
        ):
            return False
        for pos in self.ib.positions():
            if pos.contract.conId == self.contract.conId and pos.position != 0:
                return False

        def _to_utc(ts_val) -> Optional[pd.Timestamp]:
            if ts_val is None:
                return None
            t = pd.Timestamp(ts_val)
            if pd.isna(t):
                return None
            return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

        live_ts_utc: Optional[pd.Timestamp] = None
        live_high: Optional[float] = None
        live_low: Optional[float] = None
        live_close: Optional[float] = None
        if isinstance(raw_bars, pd.DataFrame):
            if not raw_bars.empty:
                last_row = raw_bars.iloc[-1]
                live_ts_utc = _to_utc(last_row.get("timestamp"))
                live_high = float(last_row["high"])
                live_low = float(last_row["low"])
                live_close = float(last_row["close"])
        elif raw_bars:
            try:
                last_bar = raw_bars[-1]
                live_ts_utc = _to_utc(getattr(last_bar, "date", None))
                live_high = float(last_bar.high)
                live_low = float(last_bar.low)
                live_close = float(last_bar.close)
            except Exception:
                pass

        sig_ts_utc = _to_utc(self._signal_bar_ts)
        elapsed_sec = (
            (pd.Timestamp.now(tz="UTC") - self._signal_submitted_utc).total_seconds()
            if self._signal_submitted_utc is not None
            else None
        )

        runaway_limit = float(getattr(self.cfg.risk, "runaway_cancel_points", 0.0) or 0.0)
        if (
            runaway_limit > 0.0
            and live_ts_utc is not None
            and sig_ts_utc is not None
            and live_ts_utc > sig_ts_utc
            and live_high is not None
            and live_low is not None
        ):
            sig_p = float(self._signal_price_orig if self._signal_price_orig is not None else self._active_signal.entry_price)
            fav_pts = (live_high - sig_p) if self._active_signal.direction == "LONG" else (sig_p - live_low)
            if fav_pts >= runaway_limit:
                return self._cancel_unfilled_entry_orders(
                    reason=f"Runaway Price Cancel (+{fav_pts:.2f} pts >= {runaway_limit:.1f} pts without fill)",
                    last_price=live_close if live_close is not None else sig_p,
                    trigger="runaway_price",
                    elapsed_sec=round(elapsed_sec, 1) if elapsed_sec is not None else None,
                    runaway_pts=round(fav_pts, 2),
                )

        max_sec = float(getattr(self.cfg.risk, "cancel_unfilled_entry_seconds", 0.0) or 0.0)
        if max_sec > 0.0 and elapsed_sec is not None and elapsed_sec >= max_sec:
            fallback_p = float(self._active_signal.entry_price)
            return self._cancel_unfilled_entry_orders(
                reason=f"Unfilled Entry Timeout ({elapsed_sec:.0f}s >= {max_sec:.0f}s)",
                last_price=live_close if live_close is not None else fallback_p,
                trigger="timeout_seconds",
                elapsed_sec=round(elapsed_sec, 1),
            )

        return False

    def _get_dead_order_ids(self) -> set[int]:
        """Collect order IDs already in terminal IBKR states across both trades() and openTrades()."""
        dead: set[int] = set()
        terminal = ("Cancelled", "ApiCancelled", "Inactive", "Filled")
        for getter_name in ("trades", "openTrades"):
            getter = getattr(self.ib, getter_name, None)
            if not callable(getter):
                continue
            try:
                items = getter()
            except Exception:
                items = ()
            for t in items or ():
                oid = getattr(getattr(t, "order", None), "orderId", 0)
                st = getattr(getattr(t, "orderStatus", None), "status", "")
                if oid and st in terminal:
                    dead.add(int(oid))
        return dead

    def _modify_active_stop_orders(self, new_sl: float) -> None:
        """Update stopPrice on every live stop order belonging to the current trade."""
        targets = list(self._active_stop_orders) if self._active_stop_orders else (
            [self._active_stop_order] if self._active_stop_order is not None else []
        )
        if not targets or self.contract is None:
            return
        dead_ids = self._get_dead_order_ids()
        for stp in targets:
            oid = getattr(stp, "orderId", 0)
            if oid and int(oid) in dead_ids:
                continue
            stp.auxPrice = new_sl
            stp.cashQty = UNSET_DOUBLE
            stp.overridePercentageConstraints = True
            stp.transmit = True
            self.ib.placeOrder(self.contract, stp)

    def place_bracket_order(
        self,
        signal: TradeSignal,
        quantity: int,
        use_market_entry: bool = False,
    ) -> List[Trade]:
        """
        Submit atomic Server-Side Bracket Order(s) to IBKR.
        When quantity >= 2 and enable_partial_tp is True, splits into 2 independent brackets:
        - Bracket 1 (leg1_qty): TP1 (+2.0R RTH / +1.4R PM)
        - Bracket 2 (leg2_qty): Runner TP2 (+5.0R RTH / +2.0R PM)
        """
        if self.contract is None:
            raise RuntimeError("Contract is not resolved.")

        from indicators import round_to_tick

        action = "BUY" if signal.direction == "LONG" else "SELL"
        reverse_action = "SELL" if action == "BUY" else "BUY"
        tick_size = self.cfg.contract.tick_size

        sig_et = pd.Timestamp(signal.timestamp)
        sig_et = (
            sig_et.tz_localize(self.cfg.risk.timezone)
            if sig_et.tzinfo is None
            else sig_et.tz_convert(self.cfg.risk.timezone)
        )
        strat_id = getattr(signal, "strategy_id", "v4.2")
        is_mr_b = strat_id == "MR-B"
        is_pm = (not is_mr_b) and (sig_et.strftime("%H:%M") < "09:30")
        # MR-B always submits a single unified bracket targeting fixed Session VWAP (no TP1/TP2 split)
        use_scale_out = (not is_mr_b) and quantity >= 2 and getattr(self.cfg.risk, "enable_partial_tp", True)

        def _build_bracket(qty: int, tp_price: float) -> tuple[Order, LimitOrder, StopOrder]:
            parent_id = self.ib.client.getReqId()
            order_ref = f"{strat_id}:{parent_id}"
            if use_market_entry:
                parent: Order = MarketOrder(
                    action=action,
                    totalQuantity=qty,
                    orderId=parent_id,
                    transmit=False,
                    outsideRth=not self.cfg.risk.rth_only,
                    tif="DAY",
                    orderRef=order_ref,
                )
            else:
                slip_pts = self.cfg.risk.entry_limit_slippage_ticks * tick_size
                lmt_price = (
                    round_to_tick(signal.entry_price + slip_pts, tick_size)
                    if action == "BUY"
                    else round_to_tick(signal.entry_price - slip_pts, tick_size)
                )
                collar_pts = float(getattr(self.cfg.risk, "entry_collar_points", 0.0) or 0.0)
                if collar_pts > 0.0:
                    stp_trigger = (
                        round_to_tick(signal.entry_price - collar_pts, tick_size)
                        if action == "BUY"
                        else round_to_tick(signal.entry_price + collar_pts, tick_size)
                    )
                    parent = StopLimitOrder(
                        action=action,
                        totalQuantity=qty,
                        lmtPrice=lmt_price,
                        stopPrice=stp_trigger,
                        orderId=parent_id,
                        transmit=False,
                        outsideRth=not self.cfg.risk.rth_only,
                        tif="DAY",
                        orderRef=order_ref,
                    )
                else:
                    parent = LimitOrder(
                        action=action,
                        totalQuantity=qty,
                        lmtPrice=lmt_price,
                        orderId=parent_id,
                        transmit=False,
                        outsideRth=not self.cfg.risk.rth_only,
                        tif="DAY",
                        orderRef=order_ref,
                    )

            take_profit = LimitOrder(
                action=reverse_action,
                totalQuantity=qty,
                lmtPrice=tp_price,
                parentId=parent_id,
                transmit=False,
                outsideRth=not self.cfg.risk.rth_only,
                tif="DAY",
                orderRef=order_ref,
            )
            take_profit.orderId = self.ib.client.getReqId()

            stop_loss = StopOrder(
                action=reverse_action,
                totalQuantity=qty,
                stopPrice=signal.stop_loss,
                parentId=parent_id,
                transmit=True,  # Final child order transmits this bracket atomically
                outsideRth=not self.cfg.risk.rth_only,
                tif="DAY",
                orderRef=order_ref,
            )
            stop_loss.orderId = self.ib.client.getReqId()
            return parent, take_profit, stop_loss

        parent_orders: List[Order] = []
        tp_orders: List[Order] = []
        stop_orders: List[Order] = []
        tp1_order: Optional[Order] = None
        tp1_price: Optional[float] = None

        if use_scale_out:
            leg1_qty = quantity // 2
            leg2_qty = quantity - leg1_qty
            tp1_r = (
                getattr(self.cfg.risk, "premarket_partial_tp1_r", 1.4)
                if is_pm
                else getattr(self.cfg.risk, "partial_tp1_r", 2.0)
            )
            tp2_r = (
                getattr(self.cfg.risk, "premarket_reward_risk_ratio", self.cfg.risk.reward_risk_ratio)
                if is_pm
                else getattr(self.cfg.risk, "runner_reward_risk_ratio", 5.0)
            )
            tp1_dist = round_to_tick(signal.stop_distance_points * tp1_r, tick_size)
            tp2_dist = round_to_tick(signal.stop_distance_points * tp2_r, tick_size)
            tp1_price = (
                round_to_tick(signal.entry_price + tp1_dist, tick_size)
                if signal.direction == "LONG"
                else round_to_tick(signal.entry_price - tp1_dist, tick_size)
            )
            tp2_price = (
                round_to_tick(signal.entry_price + tp2_dist, tick_size)
                if signal.direction == "LONG"
                else round_to_tick(signal.entry_price - tp2_dist, tick_size)
            )
            signal.take_profit = tp2_price

            p1, tp1, sl1 = _build_bracket(leg1_qty, tp1_price)
            p2, tp2, sl2 = _build_bracket(leg2_qty, tp2_price)
            orders_to_place: tuple[Order, ...] = (p1, tp1, sl1, p2, tp2, sl2)
            parent_orders = [p1, p2]
            tp_orders = [tp1, tp2]
            stop_orders = [sl1, sl2]
            tp1_order = tp1
        else:
            if is_mr_b and signal.stop_distance_points > 0:
                tp2_r = (signal.take_profit - signal.entry_price) / signal.stop_distance_points
            else:
                tp2_r = (
                    getattr(self.cfg.risk, "premarket_reward_risk_ratio", self.cfg.risk.reward_risk_ratio)
                    if is_pm
                    else self.cfg.risk.reward_risk_ratio
                )
            p1, tp1, sl1 = _build_bracket(quantity, signal.take_profit)
            orders_to_place = (p1, tp1, sl1)
            parent_orders = [p1]
            tp_orders = [tp1]
            stop_orders = [sl1]

        trades: List[Trade] = []
        try:
            is_conn = bool(self.ib.isConnected())
        except Exception:
            is_conn = False
        pace = is_conn and hasattr(self.ib, "sleep")
        for idx_ord, o in enumerate(orders_to_place):
            self._order_strategy_map[int(getattr(o, "orderId", 0) or 0)] = strat_id
            trades.append(self.ib.placeOrder(self.contract, o))
            if pace and use_scale_out and idx_ord in (0, 2, 3):
                self.ib.sleep(0.05)

        self._active_signal = signal
        self._active_strategy_id = strat_id
        self._active_parent_orders = parent_orders
        self._active_tp_orders = tp_orders
        self._active_stop_orders = stop_orders
        self._active_parent_order = parent_orders[0]
        self._active_tp1_order = tp1_order
        self._active_tp_order = tp_orders[-1]
        self._active_stop_order = stop_orders[-1]
        self._entry_submitted_bar = self.bar_count
        self._trade_filled_bar = -1
        self._post_fill_bars = 0
        self._trade_filled_ts_utc = None
        self._trade_closed_ts_utc = None
        self._be_moved = False
        self._decay_moved = False
        self._tp1_filled = False
        self._entry_fill_handled = False
        self._current_trade_filled_qty = 0
        self._current_trade_closed_qty = 0
        self._closing_exec_qty = 0
        self._last_exit_exec_utc = None
        self._current_trade_realized_pnl = 0.0
        self._current_trade_commissions = 0.0
        self._trade_entry_price = signal.entry_price
        self._trade_direction = signal.direction
        self._trade_initial_sl = signal.stop_loss
        self._trade_initial_tp1 = tp1_price
        self._trade_initial_tp = signal.take_profit
        self._stop_history = []
        self._signal_bar_ts = pd.Timestamp(signal.timestamp)
        self._signal_price_orig = float(signal.entry_price)
        self._signal_submitted_utc = pd.Timestamp.now(tz="UTC")

        self.journal.log(
            "order_submitted",
            bar_ts=signal.timestamp,
            strategy_id=strat_id,
            direction=signal.direction,
            quantity=quantity,
            order_type=type(parent_orders[0]).__name__,
            parent_order_id=parent_orders[0].orderId,
            limit_price=getattr(parent_orders[0], "lmtPrice", None),
            stop_trigger=getattr(parent_orders[0], "auxPrice", None),
            signal_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            take_profit_1=tp1_price,
        )
        logger.info(
            "BRACKET ORDER SUBMITTED [%s | %s] %s %d %s @ %.2f | SL=%.2f | TP=%s | Reason: %s",
            strat_id,
            signal.direction,
            action,
            quantity,
            self.contract.localSymbol,
            signal.entry_price,
            signal.stop_loss,
            f"TP1={tp1_price:.2f} / TP2={signal.take_profit:.2f}" if tp1_price is not None else f"{signal.take_profit:.2f}",
            signal.reason,
        )
        strat_badge = f"[{strat_id}] " if is_mr_b else ""
        icon = "🟢 LONG ORDER SUBMITTED" if signal.direction == "LONG" else "🔴 SHORT ORDER SUBMITTED"
        color = 0x1ABC9C if is_mr_b else (0x2ECC71 if signal.direction == "LONG" else 0xE74C3C)
        if tp1_price is not None:
            tp_desc = (
                f"**TP1 (`+{tp1_r:.2g}R`, `1 ct`):** `{tp1_price:.2f}`\n"
                f"**TP2 Runner (`+{tp2_r:.2g}R`, `1 ct`):** `{signal.take_profit:.2f}`"
            )
        elif is_mr_b:
            tp_desc = f"**Take Profit (Fixed VWAP, `+{tp2_r:.2f}R`):** `{signal.take_profit:.2f}`"
        else:
            tp_desc = f"**Take Profit (`+{tp2_r:.2g}R`):** `{signal.take_profit:.2f}`"
        slip_pts_msg = self.cfg.risk.entry_limit_slippage_ticks * tick_size
        cancel_sec_msg = float(getattr(self.cfg.risk, "cancel_unfilled_entry_seconds", 0.0) or 0.0)
        runaway_msg = float(getattr(self.cfg.risk, "runaway_cancel_points", 0.0) or 0.0)
        self._send_discord_alert(
            title=f"{icon} {strat_badge}— {action} {quantity} {self.contract.localSymbol} @ {signal.entry_price:.2f}",
            description=(
                f"**Mode:** `{self.mode.upper()}` | **Strategy:** `{strat_id}`\n"
                f"**Entry Price:** `{signal.entry_price:.2f}`\n"
                f"**Stop Loss (`-1R`):** `{signal.stop_loss:.2f}`\n"
                f"{tp_desc}\n"
                f"**Signal Setup:** {signal.reason}\n"
                f"_Stop-Limit collar (`±{slip_pts_msg:.2f} pt` LMT) working — a separate ✅ FILLED alert follows if executed, "
                f"or ⚪ CANCELLED after `{self.cfg.risk.cancel_unfilled_entry_bars}b / {cancel_sec_msg:.0f}s` or `+{runaway_msg:.0f}pt` runaway._"
            ),
            color=color,
        )
        return trades

    def _preempt_mr_b_for_v42(
        self,
        v42_signal: TradeSignal,
        latest_closed_ts: pd.Timestamp,
        close_p: float,
        has_pos: bool,
    ) -> bool:
        """
        Single-account Yield + Auto-Flatten (`PREEMPT_V42`):
        When an MR-B working order or open position exists and v4.2 fires a signal,
        flatten/cancel MR-B, wait for IBKR flat confirmation before submitting v4.2.
        """
        sym = self.contract.localSymbol if self.contract else self.cfg.contract.symbol
        if not has_pos:
            cancelled = self._cancel_unfilled_entry_orders(
                reason=f"Preempted by v4.2 {v42_signal.direction} signal (Yield)",
                last_price=close_p,
                trigger="preempt_v42",
            )
            if not cancelled or not self._wait_until_flat(timeout_sec=5.0):
                logger.error("PREEMPT_V42 ABORTED: MR-B working order did not clear cleanly; skipping v4.2 entry.")
                return False
        else:
            planned_qty = (
                sum(int(getattr(p, "totalQuantity", 1) or 1) for p in self._active_parent_orders)
                if self._active_parent_orders
                else int(getattr(self._active_parent_order, "totalQuantity", 1) or 1)
            )
            snapshot = {
                "strategy_id": "MR-B",
                "expected_qty": max(1, self._current_trade_filled_qty or planned_qty),
                "closed_qty": self._current_trade_closed_qty,
                "realized_pnl": self._current_trade_realized_pnl,
                "commissions": self._current_trade_commissions,
                "entry_price": self._trade_entry_price or (self._active_signal.entry_price if self._active_signal else close_p),
                "direction": self._trade_direction or "LONG",
                "sl_price": self._trade_initial_sl,
                "tp_price": self._trade_initial_tp,
                "signal_price": self._signal_price_orig,
                "signal_bar_ts": self._signal_bar_ts or latest_closed_ts,
                "filled_ts_utc": self._trade_filled_ts_utc,
                "filled_bar": self._trade_filled_bar,
                "submitted_bar": self._entry_submitted_bar,
                "be_moved": self._be_moved,
                "decay_moved": self._decay_moved,
                "stop_history": list(self._stop_history),
                "exit_reason": "PREEMPT_V42",
                "exit_bar_count": self.bar_count,
                "keys": [],
            }
            close_orders = self.flatten_all(
                reason=f"PREEMPT_V42: Yielding MR-B to v4.2 {v42_signal.direction}",
                order_ref="MR-B:PREEMPT_V42",
            )
            if not self._wait_until_flat(timeout_sec=8.0):
                logger.error("PREEMPT_V42 ABORTED: IBKR did not confirm flat after flatten timeout; skipping v4.2 entry.")
                self._send_discord_alert(
                    title=f"🚨 PREEMPT_V42 ABORTED — {sym}",
                    description=(
                        f"MR-B flatten did not confirm flat before `v4.2 {v42_signal.direction}` entry.\n"
                        f"**Action:** Skipped `v4.2` order submission to prevent netting conflict."
                    ),
                    color=0xE74C3C,
                )
                return False

            try:
                is_conn = bool(self.ib.isConnected())
            except Exception:
                is_conn = False
            if is_conn and self._active_signal is not None:
                for _ in range(6):
                    self.ib.sleep(0.25)
                    if self._active_signal is None:
                        break

            if self._active_signal is not None:
                snapshot["closed_qty"] = self._current_trade_closed_qty
                snapshot["realized_pnl"] = self._current_trade_realized_pnl
                snapshot["commissions"] = self._current_trade_commissions
                for co in close_orders:
                    oid = int(getattr(co, "orderId", 0) or 0)
                    if oid:
                        self._deferred_closed_trades[oid] = snapshot
                        snapshot["keys"].append(oid)
                if not snapshot["keys"]:
                    self._deferred_closed_trades["MR-B:PREEMPT_V42"] = snapshot
                    snapshot["keys"].append("MR-B:PREEMPT_V42")
                self._reset_active_trade_state()

        v42_ok_after, block_reason = self.risk.can_open_trade(latest_closed_ts, self.bar_count)
        if not v42_ok_after:
            logger.warning("PREEMPT_V42 cancelled v4.2 submission after flatten: %s", block_reason)
            return False

        qty = self.risk.calculate_contracts(v42_signal.stop_distance_points)
        self.journal.log(
            "preempt_v42",
            bar_ts=latest_closed_ts,
            v42_direction=v42_signal.direction,
            v42_entry_price=v42_signal.entry_price,
            close_price=close_p,
        )
        self.journal.log(
            "signal",
            bar_ts=latest_closed_ts,
            strategy_id="v4.2",
            direction=v42_signal.direction,
            reason=v42_signal.reason,
            entry_price=v42_signal.entry_price,
            stop_loss=v42_signal.stop_loss,
            take_profit=v42_signal.take_profit,
            stop_distance=v42_signal.stop_distance_points,
            atr=v42_signal.atr,
            rsi=v42_signal.rsi,
            vwap=v42_signal.vwap,
            meta_probability=v42_signal.meta_probability,
            quantity=qty,
        )
        self._send_discord_alert(
            title=f"⚡ PREEMPT_V42 — MR-B Yielded to v4.2 {v42_signal.direction} ({sym})",
            description=(
                f"MR-B position/order flattened cleanly @ `{close_p:.2f}`.\n"
                f"Submitting priority **`v4.2 {v42_signal.direction}`** (`{qty} ct`) @ `{v42_signal.entry_price:.2f}`."
            ),
            color=0x9B59B6,
        )
        self.place_bracket_order(v42_signal, quantity=qty)
        return True

    def _bars_to_dataframe(self, bars: BarDataList | pd.DataFrame) -> pd.DataFrame:
        """Convert IBKR BarDataList or TradingView DataFrame to standardized OHLCV DataFrame (excluding incomplete forming bar)."""
        if isinstance(bars, pd.DataFrame):
            if bars.empty:
                return pd.DataFrame()
            df = bars[["timestamp", "open", "high", "low", "close", "volume"]].copy().reset_index(drop=True)
            if len(df) > 1:
                import re
                m_int = re.search(r"(\d+)", self.cfg.strategy.bar_size)
                bar_mins = int(m_int.group(1)) if m_int else 2
                now_utc = pd.Timestamp.now(tz="UTC")
                last_ts = pd.Timestamp(df.iloc[-1]["timestamp"])
                last_ts = last_ts.tz_localize("UTC") if last_ts.tzinfo is None else last_ts.tz_convert("UTC")
                # If the last bar's window has not yet ended in real time, exclude it as still forming
                if last_ts + pd.Timedelta(minutes=bar_mins) > now_utc:
                    df = df.iloc[:-1].copy()
            return df.reset_index(drop=True)

        df = util.df(bars)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"date": "timestamp"})
        # When keepUpToDate=True, the last bar in `bars` is the currently forming bar;
        # evaluate on `df.iloc[:-1]` so signals only trigger on closed bars.
        if len(df) > 1:
            df = df.iloc[:-1].copy()
        return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    def _on_bar_update(self, bars: BarDataList, has_new_bar: bool) -> None:
        """Callback triggered by `ib_async` whenever a new bar forms or updates."""
        if not has_new_bar:
            return

        # Run real-time sub-bar P1 guards on every poll before early-return on identical closed bars
        cancelled_pending_now = self._check_pending_entry_guards(bars)

        df = self._bars_to_dataframe(bars)
        if df.empty:
            return

        latest_closed_ts = pd.Timestamp(df.iloc[-1]["timestamp"])
        if self._last_processed_bar_time is not None and latest_closed_ts <= self._last_processed_bar_time:
            return
        self._last_processed_bar_time = latest_closed_ts
        self.bar_count += 1

        # Journal the closed bar exactly as the live bot saw it (for feed-accuracy reconciliation)
        _lb = df.iloc[-1]
        _tv = self.tv_feed
        processed_lag_sec = (
            pd.Timestamp.now(tz="UTC")
            - (pd.Timestamp(latest_closed_ts).tz_convert("UTC") + pd.Timedelta(minutes=2))
        ).total_seconds()
        stale_bar, stale_reason = self._is_signal_bar_stale(processed_lag_sec)
        self.journal.log(
            "bar",
            bar_ts=latest_closed_ts,
            open=float(_lb["open"]),
            high=float(_lb["high"]),
            low=float(_lb["low"]),
            close=float(_lb["close"]),
            volume=float(_lb["volume"]),
            feed="tradingview" if isinstance(bars, pd.DataFrame) else "ibkr",
            tv_basis=getattr(_tv, "last_basis_points", None) if _tv else None,
            tv_stitched_bars=getattr(_tv, "last_stitched_bars_count", None) if _tv else None,
            tv_cme_mode=getattr(_tv, "last_cme_update_mode", None) if _tv else None,
            processed_lag_sec=processed_lag_sec,
        )
        # 0. Check if a pending 10-bar post-trade review chart is ready to be dispatched
        if self._pending_post_trade_chart is not None:
            task = self._pending_post_trade_chart
            if self.bar_count >= task["target_bar_count"]:
                self._pending_post_trade_chart = None
                self._dispatch_post_trade_review_chart(task)

        if cancelled_pending_now:
            return

        # 1. Check mandatory Session / EOD Flatten & Send Daily Summary Report (15:55 ET)
        self.risk.check_and_reset_session(latest_closed_ts)
        if self.risk.should_flatten_eod(latest_closed_ts):
            is_eod = self.risk._to_session_dt(latest_closed_ts).time() >= self.risk._flatten_time
            if self.has_open_position_or_orders():
                self.flatten_all(
                    reason="End of Day (15:55 ET) Auto-Flatten"
                    if is_eod
                    else "Pre-Market Close (09:24 ET) Auto-Flatten"
                )
            if self._pending_post_trade_chart is not None:
                task = self._pending_post_trade_chart
                self._pending_post_trade_chart = None
                self._dispatch_post_trade_review_chart(task)
            curr_date = self.risk.state.session_date
            # Only report EOD summary at/after 15:55 ET for the CURRENT ET trading day
            today_et = pd.Timestamp.now(tz=self.cfg.risk.timezone).date()
            if is_eod and curr_date is not None and curr_date == today_et and self._last_summary_date != curr_date:
                self._last_summary_date = curr_date
                self.send_daily_summary_report(df, latest_closed_ts)
            return

        # 1b. Refresh causal daily EMA50_d regime once per ET session when MR-B is enabled
        if self.cfg.strategy.enable_mr_b and self._daily_regime_date != self.risk.state.session_date:
            try:
                is_conn = bool(self.ib.isConnected())
            except Exception:
                is_conn = False
            if is_conn:
                self._refresh_daily_regime(latest_closed_ts)

        # 2. Compute indicators on latest closed bar
        enriched = self.strategy.prepare_dataframe(df)
        last = enriched.iloc[-1]
        close_p = float(last["close"])
        vwap_p = float(last.get("vwap", close_p))
        ema_fast_p = float(last.get("ema_fast", close_p))
        ema_slow_p = float(last.get("ema_slow", close_p))
        vwap_z = float(last.get("vwap_z", 0.0))
        hurst_v = float(last.get("hurst", 0.5))
        kalman_vel = float(last.get("kalman_vel_atr", 0.0))
        et_ts_str = self.risk._to_session_dt(latest_closed_ts).strftime("%H:%M ET")

        # 3. If already in a position or working order, check Yield/PREEMPT_V42, Timeout, BE Stop & MR-B Time Exit
        if self.has_open_position_or_orders():
            has_pos = any(
                pos.contract.conId == self.contract.conId and pos.position != 0
                for pos in self.ib.positions()
            )
            active_strat = (
                getattr(self._active_signal, "strategy_id", self._active_strategy_id)
                if self._active_signal is not None
                else self._active_strategy_id
            )
            logger.info(
                "BAR CLOSE [%s] %s=%.2f | VWAP=%.2f (Z=%+.2f) | EMA9/21=%.2f/%.2f | Hurst=%.2f | KalVel=%+.2f | Status: %s [%s]",
                et_ts_str,
                self.contract.localSymbol if self.contract else "MNQ",
                close_p,
                vwap_p,
                vwap_z,
                ema_fast_p,
                ema_slow_p,
                hurst_v,
                kalman_vel,
                "In Active Position" if has_pos else "Working Entry Order",
                active_strat,
            )

            # 3a. Single-account Yield + Auto-Flatten (PREEMPT_V42) when MR-B is active and v4.2 fires
            if self.cfg.strategy.enable_mr_b and self._active_signal is not None and active_strat == "MR-B":
                v42_ok, _ = self.risk.can_open_trade(latest_closed_ts, self.bar_count)
                v42_sig = self.strategy.evaluate_bar(enriched, idx=-1) if v42_ok else None
                if v42_sig is not None:
                    if stale_bar:
                        logger.warning("Blocked PREEMPT_V42 %s due to %s", v42_sig.direction, stale_reason)
                        self.journal.log(
                            "signal_blocked",
                            bar_ts=latest_closed_ts,
                            strategy_id="v4.2",
                            direction=v42_sig.direction,
                            reason=v42_sig.reason,
                            entry_price=v42_sig.entry_price,
                            atr=v42_sig.atr,
                            block_reason=stale_reason,
                            processed_lag_sec=round(processed_lag_sec, 2),
                        )
                    else:
                        self._preempt_mr_b_for_v42(v42_sig, latest_closed_ts, close_p, has_pos)
                    return

            if (
                not has_pos
                and self._active_parent_order is not None
                and self.cfg.risk.cancel_unfilled_entry_bars > 0
                and (self.bar_count - self._entry_submitted_bar) >= self.cfg.risk.cancel_unfilled_entry_bars
            ):
                elapsed_sec = (
                    (pd.Timestamp.now(tz="UTC") - self._signal_submitted_utc).total_seconds()
                    if self._signal_submitted_utc is not None
                    else None
                )
                bars_cfg = self.cfg.risk.cancel_unfilled_entry_bars
                self._cancel_unfilled_entry_orders(
                    reason=f"Unfilled Entry Order Timeout ({bars_cfg} bar{'s' if bars_cfg != 1 else ''})",
                    last_price=close_p,
                    trigger="timeout_bars",
                    elapsed_sec=round(elapsed_sec, 1) if elapsed_sec is not None else None,
                )
                return

            if (
                has_pos
                and self._active_parent_orders
                and self.cfg.risk.cancel_unfilled_entry_bars > 0
                and (self.bar_count - self._entry_submitted_bar) >= self.cfg.risk.cancel_unfilled_entry_bars
            ):
                parent_ids = {getattr(p, "orderId", 0) for p in self._active_parent_orders if getattr(p, "orderId", 0)}
                for ot in self.ib.openTrades():
                    if (
                        ot.contract.conId == self.contract.conId
                        and ot.order.orderId in parent_ids
                        and ot.orderStatus.status not in ("Cancelled", "ApiCancelled", "Inactive", "Filled")
                    ):
                        self.ib.cancelOrder(ot.order)
                        logger.warning(
                            "Cancelled residual unfilled parent order %s after %d bars while partial position is active.",
                            ot.order.orderId,
                            self.cfg.risk.cancel_unfilled_entry_bars,
                        )

            # 3b. MR-B Pre-FOMC 13:56 ET Auto-Flatten
            if has_pos and active_strat == "MR-B" and self.risk.should_flatten_mr_b_fomc(latest_closed_ts):
                sym_lbl = self.contract.localSymbol if self.contract else self.cfg.contract.symbol
                self.flatten_all(
                    reason="MR-B Pre-FOMC (13:56 ET) Auto-Flatten",
                    order_ref="MR-B:FOMC_FLAT",
                )
                self.journal.log("mr_b_fomc_flatten", bar_ts=latest_closed_ts, close_price=close_p)
                self._send_discord_alert(
                    title=f"🏛️ PRE-FOMC AUTO-FLATTEN [MR-B] — {sym_lbl}",
                    description=f"Flattened open MR-B position @ `{close_p:.2f}` ahead of 14:00 ET FOMC decision.",
                    color=0xE67E22,
                )
                return

            if has_pos and self._active_stop_order is not None and self._active_signal is not None:
                closed_so_far = max(
                    self._current_trade_closed_qty,
                    int(getattr(self, "_closing_exec_qty", 0) or 0),
                )
                expected_open_qty = (
                    (self._current_trade_filled_qty - closed_so_far)
                    if self._current_trade_filled_qty > 0
                    else 1
                )
                last_exit_utc = getattr(self, "_last_exit_exec_utc", None)
                recent_exit = (
                    last_exit_utc is not None
                    and (pd.Timestamp.now(tz="UTC") - last_exit_utc).total_seconds() <= 15.0
                )
                if expected_open_qty > 0 and not recent_exit:
                    # Self-healing StopOrder check: ensure a working STP order is active on IBKR at all times
                    open_stp = [
                        t for t in self.ib.openTrades()
                        if t.contract.conId == self.contract.conId
                        and t.order.orderType in ("STP", "STOP")
                        and t.orderStatus.status not in ("Cancelled", "ApiCancelled", "Inactive", "Filled")
                    ]
                    if not open_stp:
                        pos_obj = next(
                            (p for p in self.ib.positions() if p.contract.conId == self.contract.conId and p.position != 0),
                            None,
                        )
                        if pos_obj is not None:
                            rev_action = "SELL" if pos_obj.position > 0 else "BUY"
                            qty = abs(int(pos_obj.position))
                            if self._current_trade_filled_qty > 0:
                                qty = min(qty, expected_open_qty)
                            if qty > 0:
                                target_sl = float(self._active_stop_order.auxPrice or self._trade_initial_sl or close_p)
                                tp_trade = next(
                                    (
                                        t for t in self.ib.openTrades()
                                        if t.contract.conId == self.contract.conId
                                        and t.order.orderType == "LMT"
                                        and t.orderStatus.status not in ("Cancelled", "ApiCancelled", "Inactive", "Filled")
                                    ),
                                    None,
                                )
                                oca_grp = (
                                    tp_trade.order.ocaGroup
                                    if tp_trade and getattr(tp_trade.order, "ocaGroup", "")
                                    else f"RECOVERY_{self.ib.client.getReqId()}"
                                )
                                new_stp = StopOrder(
                                    action=rev_action,
                                    totalQuantity=qty,
                                    stopPrice=target_sl,
                                    outsideRth=not self.cfg.risk.rth_only,
                                    tif="DAY",
                                    ocaGroup=oca_grp,
                                    ocaType=3,
                                    overridePercentageConstraints=True,
                                    transmit=True,
                                    orderRef=f"{active_strat}:HEAL",
                                )
                                self.ib.placeOrder(self.contract, new_stp)
                                self._active_stop_order = new_stp
                                self._active_stop_orders = [new_stp]
                                logger.warning(
                                    "Self-healed missing StopOrder for open %s position: submitted %s %d STP @ %.2f",
                                    self.contract.localSymbol, rev_action, qty, target_sl,
                                )

            # Delay-aware check: only evaluate BE, Time-Decay, and Time-Exit on bars that
            # closed AT OR AFTER the real-time entry fill timestamp (handles 10-min delayed Paper bars).
            bar_end_utc = pd.Timestamp(latest_closed_ts).tz_convert("UTC") + pd.Timedelta(minutes=2)
            is_post_fill_bar = (
                self._trade_filled_ts_utc is None
                or bar_end_utc >= self._trade_filled_ts_utc
            )

            if has_pos and is_post_fill_bar and self._active_signal is not None:
                self._post_fill_bars += 1

            if (
                has_pos
                and is_post_fill_bar
                and self._active_signal is not None
                and self._active_stop_order is not None
                and active_strat == "MR-B"
            ):
                from indicators import round_to_tick

                sig = self._active_signal
                entry_ref = float(self._trade_entry_price or sig.entry_price)
                one_r = sig.stop_distance_points
                mr_be_r = float(getattr(self.cfg.risk, "mr_b_breakeven_r", 0.6))
                mr_be_off = float(getattr(self.cfg.risk, "mr_b_breakeven_offset_points", 1.5))

                if not self._be_moved and mr_be_r > 0.0 and close_p >= entry_ref + mr_be_r * one_r:
                    new_sl = round_to_tick(entry_ref + mr_be_off, self.cfg.contract.tick_size)
                    if close_p <= new_sl + 0.50:
                        logger.warning(
                            "MR-B BE trigger reached (+%.2fR), but close %.2f <= BE stop %.2f + 0.50; deferring stop move.",
                            mr_be_r,
                            close_p,
                            new_sl,
                        )
                    elif new_sl > self._active_stop_order.auxPrice:
                        self._modify_active_stop_orders(new_sl)
                        self._be_moved = True
                        self._record_stop_level(new_sl, "breakeven")
                        self.journal.log("stop_moved", bar_ts=latest_closed_ts, strategy_id="MR-B", kind="breakeven", direction="LONG", new_stop=new_sl, entry_price=entry_ref)
                        logger.info("Moved MR-B LONG StopOrder to Break-Even+ (%.2f) after reaching +%.2fR", new_sl, mr_be_r)
                        self._send_discord_alert(
                            title=f"🛡️ BREAK-EVEN [MR-B] (+{mr_be_r:.2g}R) LOCKED — {self.contract.localSymbol} LONG",
                            description=f"Moved MR-B Stop Loss up to **`{new_sl:.2f}`** (`+{mr_be_off:.1f} pts` above entry `{entry_ref:.2f}`).",
                            color=0xF1C40F,
                        )

                mr_max_hold = int(getattr(self.cfg.risk, "mr_b_max_hold_bars", 45) or 0)
                if mr_max_hold > 0 and self._post_fill_bars >= mr_max_hold:
                    sym_lbl = self.contract.localSymbol if self.contract else self.cfg.contract.symbol
                    self.flatten_all(
                        reason=f"MR-B Time Exit ({self._post_fill_bars} bars / 90 min)",
                        order_ref="MR-B:TIME_EXIT",
                    )
                    self.journal.log("mr_b_time_exit", bar_ts=latest_closed_ts, bars_held=self._post_fill_bars, close_price=close_p)
                    self._send_discord_alert(
                        title=f"⏱️ MR-B TIME EXIT ({self._post_fill_bars} bars) — {sym_lbl}",
                        description=f"Flattened MR-B LONG @ `{close_p:.2f}` after reaching `{mr_max_hold}`-bar hold limit.",
                        color=0xE67E22,
                    )
                    return

            elif (
                has_pos
                and is_post_fill_bar
                and self.cfg.risk.breakeven_at_1r
                and not self._be_moved
                and self._active_signal is not None
                and self._active_stop_order is not None
            ):
                from indicators import round_to_tick

                latest_bar = df.iloc[-1]
                high_p = float(latest_bar["high"])
                low_p = float(latest_bar["low"])
                sig = self._active_signal
                entry_ref = float(self._trade_entry_price or sig.entry_price)
                one_r = sig.stop_distance_points
                sig_et = pd.Timestamp(sig.timestamp)
                sig_et = (
                    sig_et.tz_localize(self.cfg.risk.timezone)
                    if sig_et.tzinfo is None
                    else sig_et.tz_convert(self.cfg.risk.timezone)
                )
                be_r = (
                    getattr(self.cfg.risk, "premarket_breakeven_r", 1.0)
                    if sig_et.strftime("%H:%M") < "09:30"
                    else 1.0
                )

                if sig.direction == "LONG" and high_p >= entry_ref + be_r * one_r:
                    new_sl = round_to_tick(entry_ref + self.cfg.risk.breakeven_offset_points)
                    if close_p <= new_sl + 0.50:
                        logger.warning(
                            "BE trigger reached (+%.2fR) intra-bar for LONG, but close %.2f <= BE stop %.2f + 0.50; deferring stop move to avoid instant marketable loss order.",
                            be_r,
                            close_p,
                            new_sl,
                        )
                    elif new_sl > self._active_stop_order.auxPrice:
                        self._modify_active_stop_orders(new_sl)
                        self._be_moved = True
                        self._record_stop_level(new_sl, "breakeven")
                        self.journal.log("stop_moved", bar_ts=latest_closed_ts, kind="breakeven", direction="LONG", new_stop=new_sl, entry_price=entry_ref)
                        logger.info("Moved LONG StopOrder(s) to Break-Even+ (%.2f) after reaching +%.2fR", new_sl, be_r)
                        self._send_discord_alert(
                            title=f"🛡️ BREAK-EVEN (+{be_r:.2g}R) LOCKED — {self.contract.localSymbol} LONG",
                            description=f"Moved Stop Loss up to **`{new_sl:.2f}`** (`+{self.cfg.risk.breakeven_offset_points:.1f} pts` above entry `{entry_ref:.2f}`).",
                            color=0xF1C40F,
                        )
                elif sig.direction == "SHORT" and low_p <= entry_ref - be_r * one_r:
                    new_sl = round_to_tick(entry_ref - self.cfg.risk.breakeven_offset_points)
                    if close_p >= new_sl - 0.50:
                        logger.warning(
                            "BE trigger reached (+%.2fR) intra-bar for SHORT, but close %.2f >= BE stop %.2f - 0.50; deferring stop move to avoid instant marketable loss order.",
                            be_r,
                            close_p,
                            new_sl,
                        )
                    elif new_sl < self._active_stop_order.auxPrice:
                        self._modify_active_stop_orders(new_sl)
                        self._be_moved = True
                        self._record_stop_level(new_sl, "breakeven")
                        self.journal.log("stop_moved", bar_ts=latest_closed_ts, kind="breakeven", direction="SHORT", new_stop=new_sl, entry_price=entry_ref)
                        logger.info("Moved SHORT StopOrder(s) to Break-Even+ (%.2f) after reaching +%.2fR", new_sl, be_r)
                        self._send_discord_alert(
                            title=f"🛡️ BREAK-EVEN (+{be_r:.2g}R) LOCKED — {self.contract.localSymbol} SHORT",
                            description=f"Moved Stop Loss down to **`{new_sl:.2f}`** (`-{self.cfg.risk.breakeven_offset_points:.1f} pts` below entry `{entry_ref:.2f}`).",
                            color=0xF1C40F,
                        )
                elif (
                    self.cfg.risk.time_decay_bars > 0
                    and not self._decay_moved
                    and self._post_fill_bars >= self.cfg.risk.time_decay_bars
                ):
                    if sig.direction == "LONG":
                        dec_sl = round_to_tick(entry_ref - self.cfg.risk.time_decay_stop_r * one_r)
                        if low_p > dec_sl and close_p > dec_sl + 0.50 and dec_sl > self._active_stop_order.auxPrice:
                            self._modify_active_stop_orders(dec_sl)
                            self._decay_moved = True
                            self._record_stop_level(dec_sl, "time_decay")
                            self.journal.log("stop_moved", bar_ts=latest_closed_ts, kind="time_decay", direction="LONG", new_stop=dec_sl, entry_price=entry_ref)
                            logger.info("Time-Decay Stop Tightening: Moved LONG StopOrder(s) to -%.2fR (%.2f)", self.cfg.risk.time_decay_stop_r, dec_sl)
                            self._send_discord_alert(
                                title=f"⏳ TIME-DECAY STOP (-{self.cfg.risk.time_decay_stop_r:.2f}R) — {self.contract.localSymbol} LONG",
                                description=f"Tightened Stop Loss to **`{dec_sl:.2f}`** after `{self.cfg.risk.time_decay_bars}` post-entry bars without reaching Break-Even trigger.",
                                color=0xE67E22,
                            )
                    else:
                        dec_sl = round_to_tick(entry_ref + self.cfg.risk.time_decay_stop_r * one_r)
                        if high_p < dec_sl and close_p < dec_sl - 0.50 and dec_sl < self._active_stop_order.auxPrice:
                            self._modify_active_stop_orders(dec_sl)
                            self._decay_moved = True
                            self._record_stop_level(dec_sl, "time_decay")
                            self.journal.log("stop_moved", bar_ts=latest_closed_ts, kind="time_decay", direction="SHORT", new_stop=dec_sl, entry_price=entry_ref)
                            logger.info("Time-Decay Stop Tightening: Moved SHORT StopOrder(s) to -%.2fR (%.2f)", self.cfg.risk.time_decay_stop_r, dec_sl)
                            self._send_discord_alert(
                                title=f"⏳ TIME-DECAY STOP (-{self.cfg.risk.time_decay_stop_r:.2f}R) — {self.contract.localSymbol} SHORT",
                                description=f"Tightened Stop Loss to **`{dec_sl:.2f}`** after `{self.cfg.risk.time_decay_bars}` post-entry bars without reaching Break-Even trigger.",
                                color=0xE67E22,
                            )
            if (
                has_pos
                and is_post_fill_bar
                and not self._be_moved
                and self._active_signal is not None
                and self._active_stop_order is not None
                and getattr(self.cfg.risk, "use_news_calendar_filter", True)
            ):
                pre_news_mins = int(getattr(self.cfg.risk, "pre_news_tighten_stop_mins", 4))
                imminent_ev = self.calendar.upcoming_event_within(
                    datetime.now(timezone.utc), minutes=pre_news_mins
                )
                if imminent_ev is not None:
                    from indicators import round_to_tick

                    sig = self._active_signal
                    entry_ref = float(self._trade_entry_price or sig.entry_price)
                    one_r = sig.stop_distance_points
                    news_be_off = (
                        self.cfg.risk.mr_b_breakeven_offset_points
                        if active_strat == "MR-B"
                        else self.cfg.risk.breakeven_offset_points
                    )
                    news_sl, news_kind = compute_pre_news_tightened_stop(
                        direction=sig.direction,
                        entry_ref=entry_ref,
                        one_r=one_r,
                        current_sl=float(self._active_stop_order.auxPrice),
                        close_p=close_p,
                        be_offset_pts=news_be_off,
                        decay_stop_r=self.cfg.risk.time_decay_stop_r,
                        round_fn=round_to_tick,
                    )
                    if news_sl is not None:
                        self._modify_active_stop_orders(news_sl)
                        if news_kind == "pre_news_be":
                            self._be_moved = True
                        else:
                            self._decay_moved = True
                        self._record_stop_level(news_sl, news_kind)
                        self.journal.log(
                            "stop_moved",
                            bar_ts=latest_closed_ts,
                            strategy_id=active_strat,
                            kind=news_kind,
                            direction=sig.direction,
                            new_stop=news_sl,
                            entry_price=entry_ref,
                            event=imminent_ev.title,
                        )
                        logger.info(
                            "Pre-News Stop Tightening (%s before %s): moved %s SL to %.2f",
                            news_kind,
                            imminent_ev.title,
                            sig.direction,
                            news_sl,
                        )
                        sym_lbl = self.contract.localSymbol if self.contract else self.cfg.contract.symbol
                        self._send_discord_alert(
                            title=f"📰 PRE-NEWS STOP TIGHTENED — {sym_lbl} {sig.direction}",
                            description=(
                                f"Imminent Tier-{imminent_ev.tier} USD release **`{imminent_ev.title}`** (`{imminent_ev.dt_et.strftime('%H:%M')} ET`).\n"
                                f"Moved Stop Loss to **`{news_sl:.2f}`** (`{news_kind}`)."
                            ),
                            color=0xE67E22,
                        )
            return

        # 4. Check intraday risk guardrails (v4.2 has strict priority over MR-B)
        allowed, risk_reason = self.risk.can_open_trade(latest_closed_ts, self.bar_count)
        signal: Optional[TradeSignal] = None
        if allowed:
            signal = self.strategy.evaluate_bar(enriched, idx=-1)
        elif not risk_reason.startswith("Outside"):
            try:
                blocked_sig = self.strategy.evaluate_bar(enriched, idx=-1)
                if blocked_sig is not None:
                    self.journal.log(
                        "signal_blocked",
                        bar_ts=latest_closed_ts,
                        strategy_id="v4.2",
                        direction=blocked_sig.direction,
                        reason=blocked_sig.reason,
                        entry_price=blocked_sig.entry_price,
                        atr=blocked_sig.atr,
                        block_reason=risk_reason,
                    )
            except Exception as exc:
                logger.debug("Blocked-signal journal evaluation failed: %s", exc)

        # 5. Execute priority v4.2 signal or fall back to MR-B when enabled
        if signal is not None:
            if stale_bar:
                logger.warning(
                    "Blocked [v4.2] %s signal @ %.2f due to %s",
                    signal.direction,
                    signal.entry_price,
                    stale_reason,
                )
                self.journal.log(
                    "signal_blocked",
                    bar_ts=latest_closed_ts,
                    strategy_id="v4.2",
                    direction=signal.direction,
                    reason=signal.reason,
                    entry_price=signal.entry_price,
                    atr=signal.atr,
                    block_reason=stale_reason,
                    processed_lag_sec=round(processed_lag_sec, 2),
                )
                return
            qty = self.risk.calculate_contracts(signal.stop_distance_points)
            self.journal.log(
                "signal",
                bar_ts=latest_closed_ts,
                strategy_id="v4.2",
                direction=signal.direction,
                reason=signal.reason,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                stop_distance=signal.stop_distance_points,
                atr=signal.atr,
                rsi=signal.rsi,
                vwap=signal.vwap,
                meta_probability=signal.meta_probability,
                quantity=qty,
            )
            self.place_bracket_order(signal, quantity=qty)
            return

        mr_status = ""
        if self.cfg.strategy.enable_mr_b:
            mr_ok, mr_reason = self.risk.can_open_mr_b_trade(latest_closed_ts, self.bar_count)
            if mr_ok:
                mr_signal = self.strategy.evaluate_mr_b_bar(
                    enriched,
                    idx=-1,
                    daily_bull_50=bool(self._daily_bull_50),
                )
                if mr_signal is not None:
                    if stale_bar:
                        logger.warning(
                            "Blocked [MR-B] %s signal @ %.2f due to %s",
                            mr_signal.direction,
                            mr_signal.entry_price,
                            stale_reason,
                        )
                        self.journal.log(
                            "signal_blocked",
                            bar_ts=latest_closed_ts,
                            strategy_id="MR-B",
                            direction=mr_signal.direction,
                            reason=mr_signal.reason,
                            entry_price=mr_signal.entry_price,
                            atr=mr_signal.atr,
                            block_reason=stale_reason,
                            processed_lag_sec=round(processed_lag_sec, 2),
                        )
                        return
                    mr_qty = self.risk.calculate_mr_b_contracts()
                    raw_atr_pct = last.get("atr_pct", np.nan)
                    self.journal.log(
                        "signal",
                        bar_ts=latest_closed_ts,
                        strategy_id="MR-B",
                        direction=mr_signal.direction,
                        reason=mr_signal.reason,
                        entry_price=mr_signal.entry_price,
                        stop_loss=mr_signal.stop_loss,
                        take_profit=mr_signal.take_profit,
                        stop_distance=mr_signal.stop_distance_points,
                        atr=mr_signal.atr,
                        atr_pct=round(float(raw_atr_pct), 4) if pd.notna(raw_atr_pct) else None,
                        daily_bull_50=bool(self._daily_bull_50),
                        rsi=mr_signal.rsi,
                        vwap=mr_signal.vwap,
                        vwap_z=round(vwap_z, 3),
                        quantity=mr_qty,
                    )
                    self.place_bracket_order(mr_signal, quantity=mr_qty)
                    return
                mr_status = f" | MR-B: Scanning (bull50={bool(self._daily_bull_50)})"
            else:
                if not mr_reason.startswith("Outside"):
                    try:
                        blocked_mr = self.strategy.evaluate_mr_b_bar(
                            enriched,
                            idx=-1,
                            daily_bull_50=bool(self._daily_bull_50),
                        )
                        if blocked_mr is not None:
                            self.journal.log(
                                "signal_blocked",
                                bar_ts=latest_closed_ts,
                                strategy_id="MR-B",
                                direction=blocked_mr.direction,
                                reason=blocked_mr.reason,
                                entry_price=blocked_mr.entry_price,
                                atr=blocked_mr.atr,
                                block_reason=mr_reason,
                            )
                    except Exception as exc:
                        logger.debug("Blocked MR-B journal evaluation failed: %s", exc)
                mr_status = f" | MR-B: {mr_reason}"

        base_status = risk_reason if not allowed else "Scanning (No trigger)"
        logger.info(
            "BAR CLOSE [%s] %s=%.2f | VWAP=%.2f (Z=%+.2f) | EMA9/21=%.2f/%.2f | Hurst=%.2f | KalVel=%+.2f | Status: %s%s",
            et_ts_str,
            self.contract.localSymbol if self.contract else "MNQ",
            close_p,
            vwap_p,
            vwap_z,
            ema_fast_p,
            ema_slow_p,
            hurst_v,
            kalman_vel,
            base_status,
            mr_status,
        )

    def _finalize_deferred_commission(
        self,
        snapshot: dict,
        leg_pnl: float,
        leg_comm: float,
        shares_closed: int,
        fill_price: float,
        exec_time: Any,
    ) -> None:
        """Finalize an already-flattened (e.g. PREEMPT_V42) trade whose commissionReport arrived asynchronously."""
        sym = self.contract.localSymbol if self.contract else self.cfg.contract.symbol
        strat_id = snapshot.get("strategy_id", "MR-B")
        snapshot["realized_pnl"] += leg_pnl
        snapshot["commissions"] += leg_comm
        snapshot["closed_qty"] += shares_closed

        if snapshot["closed_qty"] < snapshot["expected_qty"]:
            logger.info(
                "Deferred [%s] closing leg filled (%d/%d closed, leg PnL $%+.2f @ %.2f); awaiting remaining leg.",
                strat_id,
                snapshot["closed_qty"],
                snapshot["expected_qty"],
                leg_pnl,
                fill_price,
            )
            return

        for key in snapshot.get("keys", ()):
            self._deferred_closed_trades.pop(key, None)

        net_pnl = round(float(snapshot["realized_pnl"]), 2)
        total_comm = float(snapshot["commissions"])
        self.risk.record_trade_closed(
            net_pnl,
            exit_bar_index=int(snapshot.get("exit_bar_count", self.bar_count)),
            strategy_id=strat_id,
        )
        closed_ts = (
            pd.Timestamp(exec_time).tz_convert("UTC")
            if exec_time is not None
            else pd.Timestamp.now(tz="UTC")
        )
        entry_p = float(snapshot.get("entry_price") or fill_price)
        direction = str(snapshot.get("direction") or "LONG")
        gross_pts = (fill_price - entry_p) if direction == "LONG" else (entry_p - fill_price)
        self.journal.log(
            "trade_closed",
            bar_ts=snapshot.get("signal_bar_ts") or self._last_processed_bar_time,
            strategy_id=strat_id,
            direction=direction,
            exit_reason=snapshot.get("exit_reason", "PREEMPT_V42"),
            signal_price=snapshot.get("signal_price"),
            entry_price=entry_p,
            exit_price=fill_price,
            stop_loss=snapshot.get("sl_price"),
            take_profit=snapshot.get("tp_price"),
            be_moved=bool(snapshot.get("be_moved", False)),
            decay_moved=bool(snapshot.get("decay_moved", False)),
            gross_points=round(gross_pts, 2),
            net_pnl_usd=net_pnl,
            commission=total_comm,
            entry_filled_utc=snapshot.get("filled_ts_utc"),
            exit_filled_utc=closed_ts,
        )
        logger.info(
            "Deferred [%s | %s] trade closed | Realized PnL: $%.2f | Session PnL: $%.2f",
            strat_id,
            snapshot.get("exit_reason", "PREEMPT_V42"),
            net_pnl,
            self.risk.state.realized_pnl_usd,
        )
        self._send_discord_alert(
            title=f"⚡ [{strat_id} · {snapshot.get('exit_reason', 'PREEMPT_V42')}] CLOSED — {sym} ({net_pnl:+.2f} USD)",
            description=(
                f"**Exit Fill Price:** `{fill_price:.2f}`\n"
                f"**Trade Realized PnL:** **`${net_pnl:+.2f}`**\n"
                f"**Daily Session PnL:** `${self.risk.state.realized_pnl_usd:+.2f}`"
            ),
            color=0x2ECC71 if net_pnl > 0 else 0xE74C3C,
        )
        if self.risk.state.account_halted:
            self.flatten_all(reason=self.risk.state.account_halt_reason)

    def _on_commission_report(self, trade: Trade, fill, report) -> None:
        """Track realized PnL from IBKR commission reports as positions close (aggregating multi-leg scale-outs)."""
        if report.realizedPNL and report.realizedPNL != 0.0 and abs(report.realizedPNL) < 1e7:
            leg_pnl = float(report.realizedPNL)
            leg_comm = float(getattr(report, "commission", 0.0) or 0.0)
            exec_obj = getattr(fill, "execution", None)
            shares_closed = int(getattr(exec_obj, "shares", 1) or 1)
            fill_price = float(getattr(exec_obj, "price", 0.0) or 0.0)
            exec_time = getattr(exec_obj, "time", None)
            sym = self.contract.localSymbol if self.contract else self.cfg.contract.symbol

            exec_oid = int(getattr(exec_obj, "orderId", 0) or 0) if exec_obj is not None else 0
            trade_oid = int(getattr(getattr(trade, "order", None), "orderId", 0) or 0)
            exec_ref = str(
                (getattr(exec_obj, "orderRef", "") if exec_obj is not None else "")
                or getattr(getattr(trade, "order", None), "orderRef", "")
                or ""
            )
            deferred = (
                self._deferred_closed_trades.get(exec_oid)
                or self._deferred_closed_trades.get(trade_oid)
                or (self._deferred_closed_trades.get(exec_ref) if exec_ref else None)
            )
            if deferred is not None:
                self._finalize_deferred_commission(
                    deferred, leg_pnl, leg_comm, shares_closed, fill_price, exec_time
                )
                return

            self._current_trade_realized_pnl += leg_pnl
            self._current_trade_commissions += leg_comm
            self._current_trade_closed_qty += shares_closed

            planned_qty = (
                sum(int(getattr(p, "totalQuantity", 1) or 1) for p in self._active_parent_orders)
                if self._active_parent_orders
                else (
                    int(getattr(self._active_parent_order, "totalQuantity", 1) or 1)
                    if self._active_parent_order is not None
                    else 1
                )
            )
            expected_qty = max(1, self._current_trade_filled_qty or planned_qty)

            is_tp1_commission = (
                self._active_tp1_order is not None
                and exec_obj is not None
                and getattr(exec_obj, "orderId", None) == self._active_tp1_order.orderId
                and self._current_trade_closed_qty < expected_qty
            )
            if is_tp1_commission:
                self.journal.log(
                    "partial_tp1_closed",
                    bar_ts=self._signal_bar_ts or self._last_processed_bar_time,
                    strategy_id=self._active_strategy_id,
                    direction=self._trade_direction or (self._active_signal.direction if self._active_signal else "LONG"),
                    fill_price=fill_price,
                    shares=shares_closed,
                    leg1_pnl_usd=round(leg_pnl, 2),
                    commission=leg_comm,
                    runner_tp2=self._trade_initial_tp,
                    be_stop=float(self._active_stop_order.auxPrice) if self._active_stop_order is not None else None,
                )
                logger.info(
                    "Partial TP1 filled (%d/%d closed) @ %.2f | Leg 1 PnL: $%+.2f | Runner open -> TP2=%.2f",
                    self._current_trade_closed_qty,
                    expected_qty,
                    fill_price,
                    leg_pnl,
                    float(self._trade_initial_tp or 0.0),
                )
                be_sl_disp = (
                    f"`{float(self._active_stop_order.auxPrice):.2f}`"
                    if self._active_stop_order is not None
                    else "Break-Even"
                )
                tp2_disp = f"`{self._trade_initial_tp:.2f}`" if self._trade_initial_tp is not None else "TP2"
                self._send_discord_alert(
                    title=f"🎯 PARTIAL TP1 LOCKED — {sym} (${leg_pnl:+.2f} USD)",
                    description=(
                        f"**Leg 1 Fill Price (`1 ct`):** `{fill_price:.2f}`\n"
                        f"**Leg 1 Realized PnL:** **`${leg_pnl:+.2f} USD`**\n"
                        f"**Remaining Runner (`{expected_qty - self._current_trade_closed_qty} ct`):** Targeting {tp2_disp} | Stop Loss @ {be_sl_disp}"
                    ),
                    color=0x1ABC9C,
                )
                return

            if self._current_trade_closed_qty < expected_qty:
                logger.info(
                    "Partial closing leg filled (%d/%d closed, leg PnL $%+.2f @ %.2f); awaiting remaining leg before finalizing trade.",
                    self._current_trade_closed_qty,
                    expected_qty,
                    leg_pnl,
                    fill_price,
                )
                return

            closed_strat = (
                (getattr(self._active_signal, "strategy_id", None) if self._active_signal is not None else None)
                or self._active_strategy_id
                or self._order_strategy_map.get(exec_oid)
                or self._order_strategy_map.get(trade_oid)
                or (exec_ref.split(":", 1)[0] if ":" in exec_ref else "")
                or "v4.2"
            )
            net_pnl = round(self._current_trade_realized_pnl, 2)
            total_comm = self._current_trade_commissions
            self.risk.record_trade_closed(
                net_pnl,
                exit_bar_index=self.bar_count,
                strategy_id=closed_strat,
            )
            logger.info(
                "Trade closed [%s] | Realized PnL: $%.2f | Session PnL: $%.2f | v4.2=%d | MR-B=%d",
                closed_strat,
                net_pnl,
                self.risk.state.realized_pnl_usd,
                self.risk.state.trades_taken,
                self.risk.state.mr_b_trades_taken,
            )
            strat_badge = f"[{closed_strat}] " if closed_strat == "MR-B" else ""
            win_str = f"💰 {strat_badge}TRADE CLOSED (PROFIT)" if net_pnl > 0 else f"🛑 {strat_badge}TRADE CLOSED (LOSS)"
            color = 0x2ECC71 if net_pnl > 0 else 0xE74C3C
            self._trade_closed_ts_utc = (
                pd.Timestamp(exec_time).tz_convert("UTC")
                if exec_time is not None
                else pd.Timestamp.now(tz="UTC")
            )
            entry_p = self._trade_entry_price or (self._active_signal.entry_price if self._active_signal else fill_price)
            direction = self._trade_direction or (self._active_signal.direction if self._active_signal else "LONG")
            sl_p = self._trade_initial_sl or (self._active_signal.stop_loss if self._active_signal else None)
            tp1_p = self._trade_initial_tp1
            tp_p = self._trade_initial_tp or (self._active_signal.take_profit if self._active_signal else None)

            gross_pts = (fill_price - entry_p) if direction == "LONG" else (entry_p - fill_price)
            self.journal.log(
                "trade_closed",
                bar_ts=self._signal_bar_ts or self._last_processed_bar_time,
                strategy_id=closed_strat,
                direction=direction,
                signal_price=self._signal_price_orig,
                entry_price=entry_p,
                exit_price=fill_price,
                stop_loss=sl_p,
                take_profit=tp_p,
                take_profit_1=tp1_p,
                partial_tp1_filled=self._tp1_filled,
                be_moved=self._be_moved,
                decay_moved=self._decay_moved,
                gross_points=round(gross_pts, 2),
                net_pnl_usd=net_pnl,
                commission=total_comm,
                entry_filled_utc=self._trade_filled_ts_utc,
                exit_filled_utc=self._trade_closed_ts_utc,
            )

            # Compute trade duration in bars
            entry_offset = max(1, (self.bar_count - (self._trade_filled_bar if self._trade_filled_bar >= 0 else self._entry_submitted_bar))) if self._entry_submitted_bar >= 0 else 6

            # Estimate bar feed lag (e.g., 5 bars = 10 mins on Paper accounts) so post-trade charts include true exit + 10 bars
            delay_bars = 0
            if self._last_processed_bar_time is not None:
                lag_sec = (self._trade_closed_ts_utc - pd.Timestamp(self._last_processed_bar_time).tz_convert("UTC")).total_seconds()
                delay_bars = max(0, int(round(lag_sec / 120.0)) - 1)

            # Render clean dark-theme candlestick chart of the completed trade (with 15 bars prior)
            chart_bytes = self._generate_trade_chart(
                entry_price=entry_p,
                exit_price=fill_price,
                sl_price=sl_p,
                tp_price=tp_p,
                direction=direction,
                pnl=net_pnl,
                entry_bar_offset=entry_offset,
                post_exit_bars=0,
                pre_entry_bars=15,
                entry_ts_utc=self._trade_filled_ts_utc,
                exit_ts_utc=self._trade_closed_ts_utc,
                stop_history=list(self._stop_history),
                tp1_price=tp1_p,
            )

            leg_note = (
                f"**Leg 1 (TP1):** Locked @ `{tp1_p:.2f}` | **Runner Exit:** `{fill_price:.2f}`\n"
                if self._tp1_filled and tp1_p is not None
                else f"**Exit Fill Price:** `{fill_price:.2f}`\n"
            )
            trades_line = (
                f"**Trades Today (`{closed_strat}`):** `{self.risk.state.mr_b_trades_taken}/{self.cfg.risk.mr_b_max_daily_trades}` "
                f"(v4.2 RTH: `{self.risk.state.rth_trades_taken}/{self.cfg.risk.max_daily_trades}`)\n"
                if self.cfg.strategy.enable_mr_b
                else f"**Trades Today:** `{self.risk.state.trades_taken}/{self.cfg.risk.max_daily_trades}`\n"
            )
            self._send_discord_alert(
                title=f"{win_str} — {sym} ({net_pnl:+.2f} USD)",
                description=(
                    f"{leg_note}"
                    f"**Strategy:** `{closed_strat}` | **Trade Realized PnL:** **`${net_pnl:+.2f}`**\n"
                    f"**Daily Session PnL:** `${self.risk.state.realized_pnl_usd:+.2f}`\n"
                    f"{trades_line}"
                    f"_Post-trade context chart (+10 bars after exit) will post automatically once 10 post-exit bars close._"
                ),
                color=color,
                image_bytes=chart_bytes,
            )

            # Schedule comprehensive 10-bar post-trade review chart (accounting for any Paper bar delay)
            self._pending_post_trade_chart = {
                "entry_price": entry_p,
                "exit_price": fill_price,
                "sl_price": sl_p,
                "tp_price": tp_p,
                "tp1_price": tp1_p,
                "direction": direction,
                "pnl": net_pnl,
                "trade_duration_bars": entry_offset,
                "exit_bar_count": self.bar_count,
                "target_bar_count": self.bar_count + 10 + delay_bars,
                "entry_ts_utc": self._trade_filled_ts_utc,
                "exit_ts_utc": self._trade_closed_ts_utc,
                "stop_history": list(self._stop_history),
            }

            # Cancel any lingering open orders for this contract upon full trade close
            if self.contract is not None:
                for ot in self.ib.openTrades():
                    if (
                        ot.contract.conId == self.contract.conId
                        and getattr(getattr(ot, "orderStatus", None), "status", "")
                        not in ("Cancelled", "ApiCancelled", "Filled", "Inactive")
                    ):
                        self.ib.cancelOrder(ot.order)

            # Reset full active trade & signal state once full position has closed
            self._reset_active_trade_state()

            halt_active = (
                self.risk.state.account_halted
                if self.cfg.strategy.enable_mr_b
                else self.risk.state.halted
            )
            if halt_active:
                self.flatten_all(reason=self.risk.state.account_halt_reason or self.risk.state.halt_reason)

    def _on_exec_details(self, trade: Trade, fill) -> None:
        """Handle parent ENTRY fills (with re-anchoring) and TP1 scale-out fills (moving runner SL to Break-Even)."""
        if self.contract is None or fill.contract.conId != self.contract.conId:
            return
        from indicators import round_to_tick

        e = fill.execution
        tick_size = self.cfg.contract.tick_size

        exec_oid = int(getattr(e, "orderId", 0) or 0)
        trade_oid = int(getattr(getattr(trade, "order", None), "orderId", 0) or 0)
        exec_ref = str(
            getattr(e, "orderRef", "")
            or getattr(getattr(trade, "order", None), "orderRef", "")
            or ""
        )
        if (
            (exec_oid and exec_oid in self._deferred_closed_trades)
            or (trade_oid and trade_oid in self._deferred_closed_trades)
            or (exec_ref and exec_ref in self._deferred_closed_trades)
        ):
            return

        # 1. Handle TP1 partial fill -> remove Leg 1 SL and move Runner SL immediately to Break-Even
        if (
            self._active_tp1_order is not None
            and e.orderId == self._active_tp1_order.orderId
            and not self._tp1_filled
        ):
            self._tp1_filled = True
            self._closing_exec_qty += int(getattr(e, "shares", 1) or 1)
            self._last_exit_exec_utc = pd.Timestamp.now(tz="UTC")
            tp1_parent_id = getattr(self._active_tp1_order, "parentId", None)
            self._active_stop_orders = [
                s for s in self._active_stop_orders
                if getattr(s, "parentId", None) != tp1_parent_id
            ]
            if self._active_stop_orders:
                self._active_stop_order = self._active_stop_orders[-1]

            if self.cfg.risk.breakeven_at_1r and not self._be_moved and self._active_signal is not None:
                entry_ref = float(self._trade_entry_price or self._active_signal.entry_price)
                fill_px = float(e.price)
                curr_sl = float(self._active_stop_order.auxPrice) if self._active_stop_order is not None else entry_ref
                if self._active_signal.direction == "LONG":
                    be_sl = round_to_tick(entry_ref + self.cfg.risk.breakeven_offset_points, tick_size)
                    if fill_px > be_sl + 0.50 and be_sl > curr_sl:
                        self._modify_active_stop_orders(be_sl)
                        self._be_moved = True
                        self._record_stop_level(be_sl, "breakeven")
                        self.journal.log("stop_moved", bar_ts=pd.Timestamp.now(tz="UTC"), kind="breakeven", direction="LONG", new_stop=be_sl, entry_price=entry_ref, trigger="tp1_fill")
                else:
                    be_sl = round_to_tick(entry_ref - self.cfg.risk.breakeven_offset_points, tick_size)
                    if fill_px < be_sl - 0.50 and be_sl < curr_sl:
                        self._modify_active_stop_orders(be_sl)
                        self._be_moved = True
                        self._record_stop_level(be_sl, "breakeven")
                        self.journal.log("stop_moved", bar_ts=pd.Timestamp.now(tz="UTC"), kind="breakeven", direction="SHORT", new_stop=be_sl, entry_price=entry_ref, trigger="tp1_fill")
            return

        # 2. Match parent entry fill(s); track closing executions early to prevent self-heal race conditions
        parent_ids = {getattr(p, "orderId", 0) for p in self._active_parent_orders if getattr(p, "orderId", 0)}
        if self._active_parent_order is not None and getattr(self._active_parent_order, "orderId", 0):
            parent_ids.add(self._active_parent_order.orderId)
        is_parent = (
            (e.orderId in parent_ids)
            if parent_ids
            else (self._active_parent_order is not None and e.orderId == self._active_parent_order.orderId)
        )
        if not is_parent:
            is_closing_side = (
                (self._trade_direction == "LONG" and getattr(e, "side", "") == "SLD")
                or (self._trade_direction == "SHORT" and getattr(e, "side", "") == "BOT")
            )
            if self._trade_direction and is_closing_side:
                self._closing_exec_qty += int(getattr(e, "shares", 1) or 1)
                self._last_exit_exec_utc = pd.Timestamp.now(tz="UTC")
                if self._current_trade_filled_qty > 0 and self._closing_exec_qty >= self._current_trade_filled_qty:
                    self._active_stop_order = None
                    self._active_stop_orders = []
                    self._active_tp_order = None
                    self._active_tp_orders = []
                    self._active_tp1_order = None
                    if self.contract is not None:
                        terminal = ("Cancelled", "ApiCancelled", "Filled", "Inactive")
                        for ot in self.ib.openTrades():
                            ot_oid = int(getattr(getattr(ot, "order", None), "orderId", 0) or 0)
                            if (
                                ot.contract.conId == self.contract.conId
                                and ot_oid not in (exec_oid, trade_oid)
                                and getattr(getattr(ot, "orderStatus", None), "status", "") not in terminal
                            ):
                                self.ib.cancelOrder(ot.order)
            return

        self._current_trade_filled_qty += int(e.shares)
        if self._entry_fill_handled:
            return
        self._entry_fill_handled = True

        sig = self._active_signal
        fill_p = float(e.price)
        self._trade_entry_price = fill_p
        self._trade_direction = "LONG" if e.side == "BOT" else "SHORT"
        self._trade_filled_bar = self.bar_count
        self._post_fill_bars = 0
        self._trade_filled_ts_utc = (
            pd.Timestamp(e.time).tz_convert("UTC")
            if getattr(e, "time", None) is not None
            else pd.Timestamp.now(tz="UTC")
        )

        # If actual fill price differs from signal entry price by >= 0.25 pts (1 tick), dynamically re-anchor SL (and TP for v4.2 only)
        entry_strat_id = getattr(sig, "strategy_id", self._active_strategy_id) if sig is not None else self._active_strategy_id
        if sig is not None and abs(fill_p - sig.entry_price) >= 0.25:
            one_r = sig.stop_distance_points
            if entry_strat_id == "MR-B":
                # MR-B targets fixed signal Session VWAP; only SL re-anchors around the actual fill
                new_sl = round_to_tick(fill_p - one_r, tick_size)
                sig.entry_price = fill_p
                sig.stop_loss = new_sl
                self._trade_initial_sl = new_sl
                self._modify_active_stop_orders(new_sl)
                logger.info(
                    "Re-anchored [MR-B] SL to actual fill %.2f -> SL=%.2f (fixed VWAP TP=%.2f preserved)",
                    fill_p,
                    new_sl,
                    float(self._trade_initial_tp or sig.take_profit),
                )
            else:
                sig_et = pd.Timestamp(sig.timestamp)
                sig_et = (
                    sig_et.tz_localize(self.cfg.risk.timezone)
                    if sig_et.tzinfo is None
                    else sig_et.tz_convert(self.cfg.risk.timezone)
                )
                is_pm = sig_et.strftime("%H:%M") < "09:30"
                has_partial = self._active_tp1_order is not None
                if has_partial:
                    tp1_r = (
                        getattr(self.cfg.risk, "premarket_partial_tp1_r", 1.4)
                        if is_pm
                        else getattr(self.cfg.risk, "partial_tp1_r", 2.0)
                    )
                    rr_ratio = (
                        getattr(self.cfg.risk, "premarket_reward_risk_ratio", self.cfg.risk.reward_risk_ratio)
                        if is_pm
                        else getattr(self.cfg.risk, "runner_reward_risk_ratio", 5.0)
                    )
                    tp1_dist = round_to_tick(one_r * tp1_r, tick_size)
                    new_tp1: Optional[float] = (
                        round_to_tick(fill_p + tp1_dist, tick_size)
                        if self._trade_direction == "LONG"
                        else round_to_tick(fill_p - tp1_dist, tick_size)
                    )
                else:
                    rr_ratio = (
                        getattr(self.cfg.risk, "premarket_reward_risk_ratio", self.cfg.risk.reward_risk_ratio)
                        if is_pm
                        else self.cfg.risk.reward_risk_ratio
                    )
                    new_tp1 = None

                tp_dist = round_to_tick(one_r * rr_ratio, tick_size)
                if self._trade_direction == "LONG":
                    new_sl = round_to_tick(fill_p - one_r, tick_size)
                    new_tp = round_to_tick(fill_p + tp_dist, tick_size)
                else:
                    new_sl = round_to_tick(fill_p + one_r, tick_size)
                    new_tp = round_to_tick(fill_p - tp_dist, tick_size)

                sig.entry_price = fill_p
                sig.stop_loss = new_sl
                sig.take_profit = new_tp
                self._trade_initial_sl = new_sl
                self._trade_initial_tp = new_tp
                if new_tp1 is not None:
                    self._trade_initial_tp1 = new_tp1

                dead_ids = self._get_dead_order_ids()
                if (
                    self._active_tp1_order is not None
                    and new_tp1 is not None
                    and getattr(self._active_tp1_order, "orderId", 0) not in dead_ids
                ):
                    self._active_tp1_order.lmtPrice = new_tp1
                    self._active_tp1_order.cashQty = UNSET_DOUBLE
                    self._active_tp1_order.overridePercentageConstraints = True
                    self._active_tp1_order.transmit = True
                    self.ib.placeOrder(self.contract, self._active_tp1_order)
                if (
                    self._active_tp_order is not None
                    and getattr(self._active_tp_order, "orderId", 0) not in dead_ids
                ):
                    self._active_tp_order.lmtPrice = new_tp
                    self._active_tp_order.cashQty = UNSET_DOUBLE
                    self._active_tp_order.overridePercentageConstraints = True
                    self._active_tp_order.transmit = True
                    self.ib.placeOrder(self.contract, self._active_tp_order)
                self._modify_active_stop_orders(new_sl)
                logger.info(
                    "Re-anchored bracket SL/TP to actual fill %.2f -> SL=%.2f | TP=%s",
                    fill_p,
                    new_sl,
                    f"TP1={new_tp1:.2f} / TP2={new_tp:.2f}" if new_tp1 is not None else f"{new_tp:.2f}",
                )

        # Seed the stop history with the live stop actually resting at fill time
        live_sl = None
        if self._active_stop_order is not None and self._active_stop_order.auxPrice not in (None, UNSET_DOUBLE):
            live_sl = float(self._active_stop_order.auxPrice)
        self._stop_history = []
        self._record_stop_level(live_sl if live_sl is not None else self._trade_initial_sl, "initial", self._trade_filled_ts_utc)

        intended_qty = (
            sum(int(getattr(p, "totalQuantity", 1) or 1) for p in self._active_parent_orders)
            if self._active_parent_orders
            else int(e.shares)
        )
        disp_qty = max(int(e.shares), intended_qty)

        logger.info("ENTRY FILLED [%s]: %s %s %s @ %.2f", entry_strat_id, e.side, disp_qty, self.contract.localSymbol, e.price)
        sig_orig = self._signal_price_orig if self._signal_price_orig is not None else fill_p
        slip_pts = (fill_p - sig_orig) if self._trade_direction == "LONG" else (sig_orig - fill_p)
        fill_delay_s = (
            (self._trade_filled_ts_utc - self._signal_submitted_utc).total_seconds()
            if (self._trade_filled_ts_utc is not None and self._signal_submitted_utc is not None)
            else None
        )
        self.journal.log(
            "entry_filled",
            bar_ts=self._signal_bar_ts,
            strategy_id=entry_strat_id,
            direction=self._trade_direction,
            quantity=disp_qty,
            signal_price=sig_orig,
            fill_price=fill_p,
            slippage_points=round(slip_pts, 2),
            stop_loss=sig.stop_loss if sig else self._trade_initial_sl,
            take_profit=sig.take_profit if sig else self._trade_initial_tp,
            take_profit_1=self._trade_initial_tp1,
            fill_delay_sec=round(fill_delay_s, 2) if fill_delay_s is not None else None,
            filled_at_utc=self._trade_filled_ts_utc,
        )
        strat_badge = f"[{entry_strat_id}] " if entry_strat_id == "MR-B" else ""
        desc = f"**Strategy:** `{entry_strat_id}` | **Fill Price:** `{e.price:.2f}` | **Qty:** `{disp_qty}`\n"
        if sig is not None:
            if self._trade_initial_tp1 is not None:
                desc += f"**Stop Loss:** `{sig.stop_loss:.2f}` | **TP1:** `{self._trade_initial_tp1:.2f}` | **TP2 Runner:** `{sig.take_profit:.2f}`"
            else:
                desc += f"**Stop Loss:** `{sig.stop_loss:.2f}` | **Take Profit:** `{sig.take_profit:.2f}`"
        self._send_discord_alert(
            title=f"✅ {strat_badge}ENTRY FILLED — {'BUY' if e.side == 'BOT' else 'SELL'} {disp_qty} {self.contract.localSymbol} @ {e.price:.2f}",
            description=desc,
            color=0x1ABC9C if entry_strat_id == "MR-B" else (0x2ECC71 if e.side == "BOT" else 0xE74C3C),
        )

    def _restore_daily_risk_state(self) -> None:
        """Reconstruct today's DailyRiskState from the JSONL trade journal across container restarts."""
        if not getattr(self.journal, "enabled", False):
            return
        try:
            now_et = pd.Timestamp.now(tz=self.cfg.risk.timezone)
            day_str = now_et.strftime("%Y-%m-%d")
            events = load_journal(day_str, base_dir=str(self.journal.base_dir))
            closed_events = [ev for ev in events if ev.get("event") == "trade_closed"]
            self.risk.check_and_reset_session(now_et)
            if not closed_events:
                return
            for ev in closed_events:
                ts_raw = ev.get("exit_filled_utc") or ev.get("bar_ts") or ev.get("logged_at")
                if not ts_raw:
                    continue
                ev_ts = pd.Timestamp(ts_raw)
                ev_ts = ev_ts.tz_localize("UTC") if ev_ts.tzinfo is None else ev_ts.tz_convert("UTC")
                self.risk.check_and_reset_session(ev_ts)
                net_pnl = float(ev.get("net_pnl_usd", 0.0) or 0.0)
                strat_id = str(ev.get("strategy_id") or "v4.2")
                cd_offset = (
                    -int(getattr(self.cfg.risk, "mr_b_cooldown_bars", 3))
                    if strat_id == "MR-B"
                    else -self.cfg.risk.cooldown_bars_after_exit
                )
                self.risk.record_trade_closed(
                    net_pnl,
                    exit_bar_index=cd_offset,
                    strategy_id=strat_id,
                )
            # Refresh current session phase / RTH transition after replaying history
            self.risk.check_and_reset_session(now_et)
            st = self.risk.state
            logger.info(
                "Restored DailyRiskState from %s journal: v4.2=%d (PM=%d, RTH=%d, ConsecL=%d) | MR-B=%d (L=%d, PnL=$%+.2f) | Session PnL=$%+.2f (RTH=$%+.2f) | Halted=%s (Acct=%s)",
                day_str,
                st.trades_taken,
                st.premarket_trades_taken + st.premarket_vrev_trades_taken,
                st.rth_trades_taken,
                st.consecutive_losses,
                st.mr_b_trades_taken,
                st.mr_b_losses,
                st.mr_b_realized_pnl_usd,
                st.realized_pnl_usd,
                st.rth_realized_pnl_usd,
                st.halted,
                st.account_halted,
            )
        except Exception as exc:
            logger.warning("Failed to restore DailyRiskState from journal: %s", exc)

    def _recover_active_trade(self) -> None:
        """
        After a container/gateway restart, re-attach to an existing position and its live bracket
        StopOrder so +1R Break-Even and Time-Decay stop management keep working.
        """
        if self.contract is None:
            return
        pos = next(
            (p for p in self.ib.positions() if p.contract.conId == self.contract.conId and p.position != 0),
            None,
        )
        if pos is None:
            return

        open_trades = [t for t in self.ib.reqAllOpenOrders() if t.contract.conId == self.contract.conId]
        stop_trades = [t for t in open_trades if t.order.orderType in ("STP", "STOP")]
        tp_trades = [t for t in open_trades if t.order.orderType in ("LMT",) and t.order.parentId]
        if not stop_trades:
            logger.warning("Recovered %s position (%s) but no working StopOrder found!", self.contract.localSymbol, pos.position)
            return

        direction = "LONG" if pos.position > 0 else "SHORT"
        mult = float(self.contract.multiplier or self.cfg.contract.point_value)
        entry = round(pos.avgCost / mult, 2)
        stop_t = stop_trades[-1]
        tp_trades_sorted = sorted(tp_trades, key=lambda t: abs(float(t.order.lmtPrice) - entry))
        tp_t = tp_trades_sorted[-1] if tp_trades_sorted else None
        tp1_t = tp_trades_sorted[0] if len(tp_trades_sorted) >= 2 else None
        stop = float(stop_t.order.auxPrice)
        tp = float(tp_t.order.lmtPrice) if tp_t else entry

        recovered_strat = "v4.2"
        for ord_cand in (stop_t.order, tp_t.order if tp_t else None):
            ref = str(getattr(ord_cand, "orderRef", "") or "")
            if ref.startswith("MR-B"):
                recovered_strat = "MR-B"
                break
        if recovered_strat == "v4.2" and getattr(self.journal, "enabled", False):
            try:
                day_str = pd.Timestamp.now(tz=self.cfg.risk.timezone).strftime("%Y-%m-%d")
                for ev in reversed(load_journal(day_str, base_dir=str(self.journal.base_dir))):
                    if ev.get("event") in ("order_submitted", "entry_filled") and ev.get("strategy_id"):
                        recovered_strat = str(ev["strategy_id"])
                        break
            except Exception:
                pass

        self._active_strategy_id = recovered_strat
        self._trade_entry_price = entry
        self._trade_direction = direction
        self._trade_initial_sl = stop
        self._trade_initial_tp1 = float(tp1_t.order.lmtPrice) if tp1_t is not None else None
        self._trade_initial_tp = tp if tp_t else None
        self._trade_filled_bar = self.bar_count
        self._post_fill_bars = 0
        self._trade_filled_ts_utc = pd.Timestamp.now(tz="UTC")
        self._entry_fill_handled = True
        self._current_trade_filled_qty = abs(int(pos.position))
        self._current_trade_closed_qty = 0
        self._closing_exec_qty = 0
        self._last_exit_exec_utc = None
        self._current_trade_realized_pnl = 0.0
        self._current_trade_commissions = 0.0
        self._tp1_filled = (recovered_strat == "v4.2") and len(tp_trades_sorted) == 1 and self.cfg.risk.fixed_contracts >= 2
        self._active_signal = TradeSignal(
            timestamp=pd.Timestamp.now(tz="UTC"),
            direction=direction,
            reason=f"Recovered {recovered_strat} after restart",
            entry_price=entry,
            stop_loss=stop,
            take_profit=tp,
            stop_distance_points=abs(entry - stop),
            atr=0.0,
            rsi=50.0,
            vwap=entry,
            ema_fast=entry,
            ema_slow=entry,
            strategy_id=recovered_strat,
        )
        self._active_stop_orders = []
        for st in stop_trades:
            st.order.cashQty = UNSET_DOUBLE
            st.order.overridePercentageConstraints = True
            self._active_stop_orders.append(st.order)
        self._active_stop_order = self._active_stop_orders[-1]
        self._active_tp_orders = []
        for tpt in tp_trades_sorted:
            tpt.order.cashQty = UNSET_DOUBLE
            tpt.order.overridePercentageConstraints = True
            self._active_tp_orders.append(tpt.order)
        self._active_tp1_order = tp1_t.order if tp1_t is not None else None
        self._active_tp_order = tp_t.order if tp_t is not None else None
        self._entry_submitted_bar = self.bar_count
        # If the stop is already at/through entry, Break-Even was already applied before the restart.
        self._be_moved = (direction == "LONG" and stop >= entry) or (direction == "SHORT" and stop <= entry)
        self._decay_moved = False
        self._stop_history = []
        self._record_stop_level(stop, "breakeven" if self._be_moved else "initial", self._trade_filled_ts_utc)
        logger.warning(
            "RECOVERED active %s %s position | entry=%.2f stop=%.2f tp=%.2f (BE already moved=%s)",
            direction, self.contract.localSymbol, entry, stop, tp, self._be_moved,
        )

    def send_daily_summary_report(self, df: pd.DataFrame, latest_ts: pd.Timestamp) -> None:
        """Send an End-of-Day (15:55 ET) Session Summary Report to Discord and console logs."""
        st = self.risk.state
        sym = self.contract.localSymbol if self.contract else self.cfg.contract.symbol
        session_date_str = str(st.session_date or self.risk._to_session_dt(latest_ts).date())

        enriched = self.strategy.prepare_dataframe(df)
        last = enriched.iloc[-1]
        close_p = float(last["close"])
        vwap_p = float(last.get("vwap", close_p))

        # Calculate today's RTH session High & Low
        ts_et = pd.to_datetime(enriched["timestamp"], utc=True).dt.tz_convert(self.cfg.risk.timezone)
        today_mask = ts_et.dt.date == st.session_date
        if today_mask.any():
            day_high = float(enriched.loc[today_mask, "high"].max())
            day_low = float(enriched.loc[today_mask, "low"].min())
        else:
            day_high = float(enriched["high"].tail(100).max())
            day_low = float(enriched["low"].tail(100).min())

        total_trades = st.trades_taken + st.mr_b_trades_taken
        total_wins = st.wins + st.mr_b_wins
        total_losses = st.losses + st.mr_b_losses
        win_rate = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
        pnl = st.realized_pnl_usd

        if total_trades == 0:
            color = 0x3498DB
            outcome_icon = "📊 DAILY SUMMARY (NO TRADES)"
        elif pnl > 0:
            color = 0x2ECC71
            outcome_icon = "🏆 DAILY SUMMARY (PROFITABLE DAY)"
        else:
            color = 0xE74C3C
            outcome_icon = "📉 DAILY SUMMARY (NET LOSS)"

        status_line = (
            f"Halted ({st.account_halt_reason or st.halt_reason})"
            if (st.account_halted or st.halted)
            else "Completed Normally (15:55 ET EOD)"
        )
        v42_rth_pnl = st.rth_realized_pnl_usd - st.mr_b_realized_pnl_usd

        logger.info(
            "EOD DAILY SUMMARY [%s] %s | Trades: %d (W:%d / L:%d, %.1f%%) | v4.2=%d | MR-B=%d | Net PnL: $%+.2f | Close: %.2f | VWAP: %.2f",
            session_date_str,
            sym,
            total_trades,
            total_wins,
            total_losses,
            win_rate,
            st.trades_taken,
            st.mr_b_trades_taken,
            pnl,
            close_p,
            vwap_p,
        )

        breakdown_line = (
            f"**Strategy Breakdown:** `v4.2: {st.rth_trades_taken}/{self.cfg.risk.max_daily_trades} (${v42_rth_pnl:+.2f})` | "
            f"`MR-B: {st.mr_b_trades_taken}/{self.cfg.risk.mr_b_max_daily_trades} (L:{st.mr_b_losses}/{self.cfg.risk.mr_b_max_daily_losses}, ${st.mr_b_realized_pnl_usd:+.2f})`\n"
            if self.cfg.strategy.enable_mr_b
            else ""
        )
        trades_cap_disp = (
            f"`{total_trades}` (Wins: `{total_wins}` | Losses: `{total_losses}` | WR: `{win_rate:.1f}%`)"
            if self.cfg.strategy.enable_mr_b
            else f"`{st.trades_taken}/{self.cfg.risk.max_daily_trades}` (Wins: `{total_wins}` | Losses: `{total_losses}` | WR: `{win_rate:.1f}%`)"
        )
        self._send_discord_alert(
            title=f"{outcome_icon} — {sym} ({session_date_str})",
            description=(
                f"**Mode:** `{self.mode.upper()}` | **Contract:** `{sym}`\n"
                f"**Session Net PnL:** **`${pnl:+.2f} USD`**\n"
                f"**Trades Taken:** {trades_cap_disp}\n"
                f"{breakdown_line}"
                f"**Closing Price (15:55 ET):** `{close_p:.2f}` (VWAP: `{vwap_p:.2f}`)\n"
                f"**Session Range:** `{day_low:.2f} – {day_high:.2f}` (`{day_high - day_low:.2f} pts`)\n"
                f"**Session Status:** `{status_line}`"
            ),
            color=color,
        )

    def _on_ib_error(self, reqId: int, errorCode: int, errorString: str, contract=None) -> None:
        """Forward critical IBKR order/connectivity errors to Discord (rate-limited per errorCode)."""
        import time

        # Ignore routine informational messages and benign historical query cancellations
        critical_codes = {103, 110, 200, 201, 203, 320, 354, 451, 502, 504, 1100, 1300, 10141, 10197}
        if errorCode not in critical_codes:
            return

        now = time.time()
        last_sent = self._last_error_alert_ts.get(errorCode, 0.0)
        if (now - last_sent) < 300.0:  # Max 1 Discord alert per error code every 5 minutes
            return
        self._last_error_alert_ts[errorCode] = now

        sym = self.contract.localSymbol if self.contract else self.cfg.contract.symbol
        self._send_discord_alert(
            title=f"⚠️ IBKR API ERROR [Code {errorCode}] — {sym}",
            description=(
                f"**Request ID:** `{reqId}` | **Error Code:** `{errorCode}`\n"
                f"**IBKR Message:** `{errorString}`\n"
                f"**Mode:** `{self.mode.upper()}`"
            ),
            color=0xE67E22,
        )

    def _fetch_latest_bars(self) -> BarDataList | pd.DataFrame | None:
        """
        Fetch the latest 2-minute OHLCV bars from TradingView (0-second real-time / hybrid CME feed)
        when `use_tradingview_data` is enabled, automatically falling back to IBKR `reqHistoricalData`
        if TradingView is temporarily unreachable.
        """
        if self.tv_feed is not None and self.contract is not None:
            try:
                tv_df = self.tv_feed.fetch_bars(
                    local_symbol=self.contract.localSymbol,
                    base_symbol=self.cfg.contract.symbol,
                    bar_size=self.cfg.strategy.bar_size,
                    n_bars=1500,
                )
                if tv_df is not None and not tv_df.empty:
                    return tv_df
            except Exception as exc:
                logger.warning("TradingView data poll failed (%s); falling back to IBKR reqHistoricalData.", exc)

        return self.ib.reqHistoricalData(
            self.contract,
            endDateTime="",
            durationStr=self.cfg.strategy.history_duration,
            barSizeSetting=self.cfg.strategy.bar_size,
            whatToShow="TRADES",
            useRTH=self.cfg.risk.rth_only,
            formatDate=1,
            keepUpToDate=False,
        )

    def run_streaming(self) -> None:
        """Start real-time bar subscription and run the IBKR event loop."""
        self.connect()
        try:
            self.resolve_front_month_contract()
            self.ib.commissionReportEvent += self._on_commission_report
            self.ib.execDetailsEvent += self._on_exec_details
            self.ib.errorEvent += self._on_ib_error
            if self.cfg.strategy.enable_mr_b:
                self._refresh_daily_regime(pd.Timestamp.now(tz="UTC"))

            feed_label = (
                f"TradingView Real-Time Hybrid ({self.contract.localSymbol} + {self.cfg.ibkr.tradingview_realtime_proxy})"
                if self.tv_feed is not None
                else "IBKR Historical/Streaming"
            )
            logger.info(
                "Subscribing to %s bars via %s (%s, RTH=%s)...",
                self.contract.localSymbol,
                feed_label,
                self.cfg.strategy.bar_size,
                self.cfg.risk.rth_only,
            )
            self.bars = self._fetch_latest_bars()
            poll_interval = 10 if self.tv_feed is not None else 20
            logger.info("Bot is now live and polling 2-min bar closes every %ds (%s). Press Ctrl+C to stop.", poll_interval, feed_label)
            window_desc = (
                f"`{self.cfg.risk.premarket_start_time}–{self.cfg.risk.premarket_end_time} ET (Pre-Market)` + "
                f"`{self.cfg.risk.entry_start_time}–{self.cfg.risk.entry_end_time} ET (RTH)`"
                if getattr(self.cfg.risk, "use_premarket_session", False)
                else f"`{self.cfg.risk.entry_start_time}–{self.cfg.risk.entry_end_time} ET`"
            )
            self._sync_discord_webhook_profile()
            try:
                self.calendar.fetch_and_cache()
                today_news = self.calendar.get_today_events()
                news_summary = self.calendar.format_events_summary(today_news)
            except Exception as exc:
                logger.warning("EconomicCalendar startup fetch error: %s", exc)
                news_summary = "Unavailable"
            bot_name = getattr(self.cfg.ibkr, "discord_bot_name", "NQ Quant Bot (v4.2)")
            pm_be = getattr(self.cfg.risk, "premarket_breakeven_r", 1.0)
            pm_rr = getattr(self.cfg.risk, "premarket_reward_risk_ratio", self.cfg.risk.reward_risk_ratio)
            is_partial = (
                self.cfg.risk.fixed_contracts >= 2
                and getattr(self.cfg.risk, "enable_partial_tp", True)
            )
            if is_partial:
                tp1_r = getattr(self.cfg.risk, "partial_tp1_r", 2.0)
                tp2_r = getattr(self.cfg.risk, "runner_reward_risk_ratio", 5.0)
                pm_tp1_r = getattr(self.cfg.risk, "premarket_partial_tp1_r", 1.4)
                tp_rule_str = (
                    f"RTH: +1R → +{self.cfg.risk.breakeven_offset_points:.1f}pt BE / TP1 +{tp1_r:.1f}R & Runner TP2 +{tp2_r:.1f}R | "
                    f"PM: +{pm_be:.2g}R BE / TP1 +{pm_tp1_r:.1f}R & TP2 +{pm_rr:.1f}R"
                )
            else:
                tp_rule_str = (
                    f"RTH: +1R → +{self.cfg.risk.breakeven_offset_points:.1f}pt BE / {self.cfg.risk.reward_risk_ratio:.1f}R TP | "
                    f"PM: +{pm_be:.2g}R BE / {pm_rr:.1f}R TP"
                )
            slip_pts_cfg = self.cfg.risk.entry_limit_slippage_ticks * self.cfg.contract.tick_size
            cancel_bars_cfg = int(getattr(self.cfg.risk, "cancel_unfilled_entry_bars", 1) or 0)
            cancel_sec_cfg = float(getattr(self.cfg.risk, "cancel_unfilled_entry_seconds", 0.0) or 0.0)
            runaway_cfg = float(getattr(self.cfg.risk, "runaway_cancel_points", 0.0) or 0.0)
            hv_atr_thresh = float(getattr(self.cfg.risk, "high_vol_atr_threshold", 60.0) or 60.0)
            hv_max_stop = float(getattr(self.cfg.risk, "high_vol_max_stop_points", self.cfg.risk.max_stop_points) or self.cfg.risk.max_stop_points)
            mr_b_line = ""
            if self.cfg.strategy.enable_mr_b:
                mr_qty_cfg = self.risk.calculate_mr_b_contracts()
                regime_str = (
                    f"bull_50={self._daily_bull_50} ({self._daily_prev_close} vs EMA50={self._daily_ema50})"
                    if self._daily_prev_close is not None and self._daily_ema50 is not None
                    else f"bull_50={bool(self._daily_bull_50)}"
                )
                mr_b_line = (
                    f"\n**MR-B Dual-Engine:** `ON ({mr_qty_cfg} ct | ATR% {self.cfg.strategy.mr_b_atr_pct_min:.3f}–{self.cfg.strategy.mr_b_atr_pct_max:.3f}% | "
                    f"SL cap {self.cfg.strategy.mr_b_max_sl_points:.0f}b | +{self.cfg.risk.mr_b_breakeven_r:.1f}R→+{self.cfg.risk.mr_b_breakeven_offset_points:.1f}b BE | "
                    f"{self.cfg.risk.mr_b_max_hold_bars}b TIME | FOMC 13:56 | Yield+PREEMPT_V42 | {regime_str})`"
                )
            self._send_discord_alert(
                title=f"🤖 {bot_name} Online — {self.contract.localSymbol} ({self.mode.upper()})",
                description=(
                    f"**Profile:** `{bot_name}`\n"
                    f"**Contract:** `{self.contract.localSymbol}` (Expiry `{self.contract.lastTradeDateOrContractMonth}` | `${self.cfg.contract.point_value:.0f}/pt`)\n"
                    f"**ATR Window:** `RTH {self.cfg.strategy.min_atr_points:.0f}–{self.cfg.strategy.max_atr_points:.0f} pts`\n"
                    f"**SL Cap:** `{self.cfg.risk.max_stop_points:.0f} pts ({hv_max_stop:.0f} pts if ATR > {hv_atr_thresh:.0f})`\n"
                    f"**Entry Guard:** `±{slip_pts_cfg:.2f}pt LMT | Cancel {cancel_bars_cfg}b / {cancel_sec_cfg:.0f}s / +{runaway_cfg:.0f}pt`\n"
                    f"**Stop & TP Rules:** `{tp_rule_str} | {self.cfg.risk.time_decay_bars}b → -{self.cfg.risk.time_decay_stop_r:.2f}R Decay`\n"
                    f"**Data Feed:** `{feed_label}`\n"
                    f"**Timeframe:** `{self.cfg.strategy.bar_size}` | **Contracts:** `{self.cfg.risk.fixed_contracts}` | **Max Daily Loss:** `${self.cfg.risk.max_daily_loss_usd:.0f}`\n"
                    f"**Session Windows:** {window_desc}\n"
                    f"**Today's USD High-Impact News:** `{news_summary}`\n"
                    f"**News Blackout Guard:** `T1 (-{self.cfg.risk.news_mega_pre_blackout_mins}m/+{self.cfg.risk.news_mega_post_blackout_mins}m) | T2 (-{self.cfg.risk.news_pre_blackout_mins}m/+{self.cfg.risk.news_post_blackout_mins}m) | Tighten {self.cfg.risk.pre_news_tighten_stop_mins}m prior`"
                    f"{mr_b_line}"
                ),
                color=0x3498DB,
            )
            # Reconstruct today's intraday risk counters & re-attach any live position before processing closed bars
            self._restore_daily_risk_state()
            self._recover_active_trade()
            self._maybe_send_daily_calendar_briefing()
            if self.bars is not None and len(self.bars) > 0:
                self._on_bar_update(self.bars, has_new_bar=True)

            # Poll every 10s (TradingView) / 20s (IBKR) so 2-minute bar closes are processed immediately
            consecutive_empty_polls = 0
            while self.ib.isConnected():
                self.ib.sleep(poll_interval)
                try:
                    self.calendar.refresh_if_needed()
                    self._maybe_send_daily_calendar_briefing()
                    fresh_bars = self._fetch_latest_bars()
                    if fresh_bars is not None and len(fresh_bars) > 0:
                        consecutive_empty_polls = 0
                        self.bars = fresh_bars
                        self._on_bar_update(self.bars, has_new_bar=True)
                    else:
                        consecutive_empty_polls += 1
                        logger.warning(
                            "Empty/timed-out bar poll (%d/3).",
                            consecutive_empty_polls,
                        )
                        if consecutive_empty_polls >= 3:
                            sym = self.contract.localSymbol if self.contract else self.cfg.contract.symbol
                            self._send_discord_alert(
                                title=f"⚠️ DATA FEED STALLED — Auto-Restarting ({sym})",
                                description=(
                                    f"**Status:** Bar feed timed out `3` times consecutively.\n"
                                    f"**Action:** Disconnecting bot so Docker & the Host Watchdog automatically restart `ib-gateway` and `nq-ibkr-bot`.\n"
                                    f"_You will receive a 🤖 NQ Quant Bot Online alert once reconnected._"
                                ),
                                color=0xE67E22,
                                blocking=True,
                            )
                            raise RuntimeError("Bar data feed timed out 3 times consecutively")
                except RuntimeError:
                    logger.error("Historical feed stalled after 3 consecutive timeouts; disconnecting to trigger clean restart.")
                    break
                except Exception as exc:  # never let one transient error kill a bot that manages live stops
                    logger.exception("Error during bar polling/processing (bot continues): %s", exc)

            if not self.ib.isConnected():
                sym = self.contract.localSymbol if self.contract else self.cfg.contract.symbol
                self._send_discord_alert(
                    title=f"⚠️ IBKR CONNECTION LOST — Reconnecting ({sym})",
                    description=(
                        f"**Status:** Active socket connection to IB Gateway (`{self.cfg.ibkr.host}`) was closed.\n"
                        f"**Action:** Bot process is exiting so Docker automatically reconnects."
                    ),
                    color=0xE67E22,
                    blocking=True,
                )
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user (Ctrl+C).")
        except Exception as exc:
            sym = self.contract.localSymbol if self.contract else self.cfg.contract.symbol
            self._send_discord_alert(
                title=f"🚨 BOT STOPPED (UNHANDLED ERROR) — {sym}",
                description=(
                    f"**Error Type:** `{type(exc).__name__}`\n"
                    f"**Details:** `{exc}`\n"
                    f"_Docker (`restart: unless-stopped`) will automatically restart the bot._"
                ),
                color=0xE74C3C,
                blocking=True,
            )
            raise
        finally:
            if self.ib.isConnected():
                self.ib.disconnect()
                logger.info("Disconnected from IBKR.")

    def download_historical_csv(self, output_path: str, duration: str = "5 D") -> pd.DataFrame:
        """Download historical 1m/2m/5m NQ/MNQ bars from IBKR (using ContFuture for multi-month spans) and save to CSV."""
        from ib_async import ContFuture

        self.connect()
        try:
            # Use Continuous Future (ContFuture) for multi-week / multi-month historical downloads
            # so contract rollovers (e.g., Sep U6 -> Dec Z6) are stitched seamlessly by IBKR.
            if any(unit in duration.upper() for unit in ("M", "W")) or (
                "D" in duration.upper() and int(duration.split()[0]) > 15
            ):
                cont = ContFuture(
                    symbol=self.cfg.contract.symbol,
                    exchange=self.cfg.contract.exchange,
                    currency=self.cfg.contract.currency,
                )
                qualified = self.ib.qualifyContracts(cont)
                target_contract = qualified[0] if qualified else self.resolve_front_month_contract()
                logger.info("Using Continuous Future (%s) for multi-month history download (%s)...", target_contract.localSymbol, duration)
            else:
                target_contract = self.resolve_front_month_contract()

            bars = self.ib.reqHistoricalData(
                target_contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=self.cfg.strategy.bar_size,
                whatToShow="TRADES",
                useRTH=self.cfg.risk.rth_only,
                formatDate=1,
                keepUpToDate=False,
                timeout=120,
            )
            df = util.df(bars)
            if df is None or df.empty:
                raise RuntimeError("No historical bars returned from IBKR.")
            df = df.rename(columns={"date": "timestamp"})
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df.to_csv(output_path, index=False)
            logger.info("Saved %d historical bars to %s", len(df), output_path)
            return df
        finally:
            if self.ib.isConnected():
                self.ib.disconnect()
