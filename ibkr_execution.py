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
from datetime import date, datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from ib_async import IB, BarDataList, Contract, Future, LimitOrder, MarketOrder, Order, StopOrder, Trade, util

from config import BotConfig
from risk_manager import RiskManager
from strategy import NQIntradayStrategy, TradeSignal

logger = logging.getLogger("nq_ibkr_bot")


class IBKRNQBotRunner:
    """Live / Paper execution engine for NQ & MNQ futures on Interactive Brokers."""

    def __init__(self, config: BotConfig, mode: str = "paper") -> None:
        self.cfg = config
        self.mode = mode.lower()
        self.ib = IB()
        self.strategy = NQIntradayStrategy(config.strategy, config.risk, config.contract)
        self.risk = RiskManager(config.risk, config.contract)

        self.contract: Optional[Contract] = None
        self.bars: Optional[BarDataList] = None
        self.bar_count: int = 0
        self._last_processed_bar_time: Optional[datetime] = None

        # Active bracket order tracking for +1R Break-Even, Time-Decay Stop & Unfilled Entry Timeout
        self._active_signal: Optional[TradeSignal] = None
        self._active_parent_order: Optional[Order] = None
        self._active_stop_order: Optional[Order] = None
        self._entry_submitted_bar: int = -1
        self._trade_filled_bar: int = -1
        self._be_moved: bool = False
        self._decay_moved: bool = False
        self._last_summary_date: Optional[date] = None
        self._trade_entry_price: Optional[float] = None
        self._trade_direction: Optional[str] = None
        self._trade_initial_sl: Optional[float] = None
        self._trade_initial_tp: Optional[float] = None
        self._pending_post_trade_chart: Optional[dict] = None

    def _send_discord_alert(
        self,
        title: str,
        description: str,
        color: int = 0x3498DB,
        image_bytes: Optional[bytes] = None,
    ) -> None:
        """Send an asynchronous rich embed notification (with optional chart image) to Discord."""
        url = (self.cfg.ibkr.discord_webhook_url or "").strip()
        if not url:
            return

        payload = {
            "username": "NQ Quant Bot (v4)",
            "embeds": [
                {
                    "title": title,
                    "description": description,
                    "color": color,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            ],
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
                        "User-Agent": "NQ-IBKR-Quant-Bot/4.0",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15):
                    pass
            except Exception as exc:
                logger.warning("Failed to send Discord alert: %s", exc)

        threading.Thread(target=_post, daemon=True).start()

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
    ) -> Optional[bytes]:
        """Generate a clean dark-theme candlestick chart of the trade with surrounding context for Discord."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            if not self.bars:
                return None

            df = self._bars_to_dataframe(self.bars)
            if df.empty or len(df) < 5:
                return None

            enriched = self.strategy.prepare_dataframe(df)
            total_needed = max(35, pre_entry_bars + entry_bar_offset + post_exit_bars)
            chart_df = enriched.tail(total_needed).copy().reset_index(drop=True)
            if len(chart_df) < 5:
                return None

            # Compute New York ET time string for each candle
            ts_et = pd.to_datetime(chart_df["timestamp"], utc=True).dt.tz_convert(self.cfg.risk.timezone)
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

            # Determine entry and exit x indices
            exit_idx = max(0, len(chart_df) - 1 - post_exit_bars)
            entry_idx = max(0, exit_idx - max(1, entry_bar_offset))
            if entry_idx >= exit_idx and exit_idx > 0:
                entry_idx = max(0, exit_idx - 1)

            entry_time = time_strs[entry_idx]
            exit_time = time_strs[exit_idx]

            # Horizontal Lines for Initial SL & TP (spanning from entry to exit + 4 bars or chart end)
            span_end = min(len(chart_df) - 0.5, exit_idx + 6)
            span_x = [entry_idx, span_end]
            if sl_price is not None:
                ax.hlines(sl_price, xmin=span_x[0], xmax=span_x[1], colors="#e74c3c", linestyles="--", linewidth=1.4, zorder=4)
                ax.text(
                    span_x[1], sl_price, f" Initial SL: {sl_price:.2f} [Set @ {entry_time} ET]",
                    color="#e74c3c", fontsize=8, fontweight="bold", va="center", ha="left"
                )

            if tp_price is not None:
                ax.hlines(tp_price, xmin=span_x[0], xmax=span_x[1], colors="#2ecc71", linestyles="--", linewidth=1.4, zorder=4)
                ax.text(
                    span_x[1], tp_price, f" Initial TP: {tp_price:.2f} [Set @ {entry_time} ET]",
                    color="#2ecc71", fontsize=8, fontweight="bold", va="center", ha="left"
                )

            # Initial Entry reference guideline
            ax.hlines(entry_price, xmin=span_x[0], xmax=span_x[1], colors="#787b86", linestyles=":", linewidth=1.0, zorder=4)

            # Entry Annotation with exact time
            entry_color = "#e74c3c" if direction == "SHORT" else "#2ecc71"
            ax.annotate(
                f"{direction} ENTRY [{entry_time} ET]\n{entry_price:.2f}",
                xy=(entry_idx, entry_price),
                xytext=(max(0, entry_idx - 3.5), entry_price + (26 if direction == "SHORT" else -26)),
                arrowprops=dict(facecolor=entry_color, edgecolor=entry_color, shrink=0.08, width=2, headwidth=7),
                fontsize=8.5,
                fontweight="bold",
                color="#ffffff",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=entry_color, alpha=0.88, edgecolor="none"),
                zorder=6,
            )

            # Exit Annotation with exact time
            exit_color = "#2ecc71" if pnl >= 0 else "#e74c3c"
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            ax.annotate(
                f"EXIT [{exit_time} ET] ({pnl_str})\n{exit_price:.2f}",
                xy=(exit_idx, exit_price),
                xytext=(max(0, exit_idx - 3.5), exit_price - 22 if pnl >= 0 else exit_price + 22),
                arrowprops=dict(facecolor=exit_color, edgecolor=exit_color, shrink=0.08, width=2, headwidth=7),
                fontsize=8.5,
                fontweight="bold",
                color="#ffffff",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=exit_color, alpha=0.88, edgecolor="none"),
                zorder=6,
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
            ax.legend(loc="upper right", framealpha=0.4, facecolor="#1e222d", edgecolor="none", fontsize=8)

            # Adjust limits for SL/TP and text labels
            valid_targets = [p for p in (sl_price, tp_price, entry_price, exit_price) if p is not None]
            y_min = min(chart_df["low"].min(), min(valid_targets)) - 10
            y_max = max(chart_df["high"].max(), max(valid_targets)) + 12
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(-1, len(chart_df) + 3.5)

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

    def flatten_all(self, reason: str = "Manual/EOD Flatten") -> None:
        """Cancel all open orders for the contract and close any open position with a MarketOrder."""
        if self.contract is None:
            return

        logger.warning("Flattening all orders and positions for %s (%s)", self.contract.localSymbol, reason)

        # 1. Cancel open orders on this contract
        for trade in self.ib.openTrades():
            if trade.contract.conId == self.contract.conId:
                self.ib.cancelOrder(trade.order)

        # 2. Close net position if non-zero
        for pos in self.ib.positions():
            if pos.contract.conId == self.contract.conId and pos.position != 0:
                action = "SELL" if pos.position > 0 else "BUY"
                qty = abs(int(pos.position))
                close_order = MarketOrder(action=action, totalQuantity=qty, tif="DAY")
                self.ib.placeOrder(self.contract, close_order)
                logger.warning("Submitted %s %d %s MarketOrder to flatten position.", action, qty, self.contract.localSymbol)

    def place_bracket_order(
        self,
        signal: TradeSignal,
        quantity: int,
        use_market_entry: bool = False,
    ) -> List[Trade]:
        """
        Submit an atomic Server-Side Bracket Order to IBKR:
        - Parent Order: Limit (or Market) entry
        - Child 1: Take Profit LimitOrder
        - Child 2: Stop Loss StopOrder
        """
        if self.contract is None:
            raise RuntimeError("Contract is not resolved.")

        action = "BUY" if signal.direction == "LONG" else "SELL"
        reverse_action = "SELL" if action == "BUY" else "BUY"

        parent_id = self.ib.client.getReqId()

        if use_market_entry:
            parent: Order = MarketOrder(
                action=action,
                totalQuantity=quantity,
                orderId=parent_id,
                transmit=False,
                outsideRth=not self.cfg.risk.rth_only,
                tif="DAY",
            )
        else:
            from indicators import round_to_tick

            slip_pts = self.cfg.risk.entry_limit_slippage_ticks * self.cfg.contract.tick_size
            lmt_price = (
                round_to_tick(signal.entry_price + slip_pts, self.cfg.contract.tick_size)
                if action == "BUY"
                else round_to_tick(signal.entry_price - slip_pts, self.cfg.contract.tick_size)
            )
            parent = LimitOrder(
                action=action,
                totalQuantity=quantity,
                lmtPrice=lmt_price,
                orderId=parent_id,
                transmit=False,
                outsideRth=not self.cfg.risk.rth_only,
                tif="DAY",
            )

        take_profit = LimitOrder(
            action=reverse_action,
            totalQuantity=quantity,
            lmtPrice=signal.take_profit,
            parentId=parent_id,
            transmit=False,
            outsideRth=not self.cfg.risk.rth_only,
            tif="DAY",
        )
        take_profit.orderId = self.ib.client.getReqId()

        stop_loss = StopOrder(
            action=reverse_action,
            totalQuantity=quantity,
            stopPrice=signal.stop_loss,
            parentId=parent_id,
            transmit=True,  # Final child order transmits the entire bracket atomically
            outsideRth=not self.cfg.risk.rth_only,
            tif="DAY",
        )
        stop_loss.orderId = self.ib.client.getReqId()

        trades: List[Trade] = []
        for o in (parent, take_profit, stop_loss):
            trades.append(self.ib.placeOrder(self.contract, o))

        self._active_signal = signal
        self._active_parent_order = parent
        self._active_stop_order = stop_loss
        self._entry_submitted_bar = self.bar_count
        self._be_moved = False
        self._decay_moved = False
        self._trade_entry_price = signal.entry_price
        self._trade_direction = signal.direction
        self._trade_initial_sl = signal.stop_loss
        self._trade_initial_tp = signal.take_profit

        logger.info(
            "BRACKET ORDER SUBMITTED [%s] %s %d %s @ %.2f | SL=%.2f | TP=%.2f | Reason: %s",
            signal.direction,
            action,
            quantity,
            self.contract.localSymbol,
            signal.entry_price,
            signal.stop_loss,
            signal.take_profit,
            signal.reason,
        )
        icon = "🟢 LONG ORDER SUBMITTED" if signal.direction == "LONG" else "🔴 SHORT ORDER SUBMITTED"
        color = 0x2ECC71 if signal.direction == "LONG" else 0xE74C3C
        self._send_discord_alert(
            title=f"{icon} — {action} {quantity} {self.contract.localSymbol} @ {signal.entry_price:.2f}",
            description=(
                f"**Mode:** `{self.mode.upper()}`\n"
                f"**Entry Price:** `{signal.entry_price:.2f}`\n"
                f"**Stop Loss (`-1R`):** `{signal.stop_loss:.2f}`\n"
                f"**Take Profit (`+3R`):** `{signal.take_profit:.2f}`\n"
                f"**Signal Setup:** {signal.reason}\n"
                f"_Limit order working — a separate ✅ FILLED alert follows if executed, "
                f"or ⚪ CANCELLED after {self.cfg.risk.cancel_unfilled_entry_bars} bars._"
            ),
            color=color,
        )
        return trades

    def _bars_to_dataframe(self, bars: BarDataList) -> pd.DataFrame:
        """Convert IBKR BarDataList to standardized OHLCV DataFrame (excluding incomplete forming bar)."""
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

        df = self._bars_to_dataframe(bars)
        if df.empty:
            return

        latest_closed_ts = pd.Timestamp(df.iloc[-1]["timestamp"])
        if self._last_processed_bar_time == latest_closed_ts:
            return
        self._last_processed_bar_time = latest_closed_ts
        self.bar_count += 1

        # 0. Check if a pending 10-bar post-trade review chart is ready to be dispatched
        if self._pending_post_trade_chart is not None:
            task = self._pending_post_trade_chart
            if self.bar_count >= task["target_bar_count"]:
                self._pending_post_trade_chart = None
                self._dispatch_post_trade_review_chart(task)

        # 1. Check mandatory EOD Flatten & Send Daily Summary Report (15:55 ET)
        self.risk.check_and_reset_session(latest_closed_ts)
        if self.risk.should_flatten_eod(latest_closed_ts):
            if self.has_open_position_or_orders():
                self.flatten_all(reason="End of Day (15:55 ET) Auto-Flatten")
            if self._pending_post_trade_chart is not None:
                task = self._pending_post_trade_chart
                self._pending_post_trade_chart = None
                self._dispatch_post_trade_review_chart(task)
            curr_date = self.risk.state.session_date
            # Only report for the CURRENT ET trading day (prevents stale duplicate summaries
            # after nightly IB Gateway restarts or weekend reconnects).
            today_et = pd.Timestamp.now(tz=self.cfg.risk.timezone).date()
            if curr_date is not None and curr_date == today_et and self._last_summary_date != curr_date:
                self._last_summary_date = curr_date
                self.send_daily_summary_report(df, latest_closed_ts)
            return

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

        # 3. If already in a position or working order, check Unfilled Timeout & +1R Break-Even Stop
        if self.has_open_position_or_orders():
            has_pos = any(
                pos.contract.conId == self.contract.conId and pos.position != 0
                for pos in self.ib.positions()
            )
            logger.info(
                "BAR CLOSE [%s] %s=%.2f | VWAP=%.2f (Z=%+.2f) | EMA9/21=%.2f/%.2f | Hurst=%.2f | KalVel=%+.2f | Status: %s",
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
            )
            if (
                not has_pos
                and self._active_parent_order is not None
                and self.cfg.risk.cancel_unfilled_entry_bars > 0
                and (self.bar_count - self._entry_submitted_bar) >= self.cfg.risk.cancel_unfilled_entry_bars
            ):
                cancelled_sig = self._active_signal
                self.flatten_all(reason=f"Unfilled Entry Order Timeout ({self.cfg.risk.cancel_unfilled_entry_bars} bars)")
                self._active_parent_order = None
                self._active_signal = None
                if cancelled_sig is not None:
                    self._send_discord_alert(
                        title=f"⚪ ENTRY CANCELLED (NOT FILLED) — {self.contract.localSymbol} {cancelled_sig.direction}",
                        description=(
                            f"Limit entry @ `{cancelled_sig.entry_price:.2f}` was not filled within "
                            f"`{self.cfg.risk.cancel_unfilled_entry_bars}` bars and was cancelled.\n"
                            f"**No position was opened — PnL `$0.00`.**"
                        ),
                        color=0x95A5A6,
                    )
                return

            if (
                has_pos
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
                one_r = sig.stop_distance_points

                if sig.direction == "LONG" and high_p >= sig.entry_price + one_r:
                    new_sl = round_to_tick(sig.entry_price + self.cfg.risk.breakeven_offset_points)
                    if new_sl > self._active_stop_order.auxPrice:
                        self._active_stop_order.auxPrice = new_sl
                        self.ib.placeOrder(self.contract, self._active_stop_order)
                        self._be_moved = True
                        logger.info("Moved LONG StopOrder to Break-Even+ (%.2f) after reaching +1R", new_sl)
                        self._send_discord_alert(
                            title=f"🛡️ BREAK-EVEN (+1R) LOCKED — {self.contract.localSymbol} LONG",
                            description=f"Moved Stop Loss up to **`{new_sl:.2f}`** (`+1.0 pt` above entry `{sig.entry_price:.2f}`).",
                            color=0xF1C40F,
                        )
                elif sig.direction == "SHORT" and low_p <= sig.entry_price - one_r:
                    new_sl = round_to_tick(sig.entry_price - self.cfg.risk.breakeven_offset_points)
                    if new_sl < self._active_stop_order.auxPrice:
                        self._active_stop_order.auxPrice = new_sl
                        self.ib.placeOrder(self.contract, self._active_stop_order)
                        self._be_moved = True
                        logger.info("Moved SHORT StopOrder to Break-Even+ (%.2f) after reaching +1R", new_sl)
                        self._send_discord_alert(
                            title=f"🛡️ BREAK-EVEN (+1R) LOCKED — {self.contract.localSymbol} SHORT",
                            description=f"Moved Stop Loss down to **`{new_sl:.2f}`** (`-1.0 pt` below entry `{sig.entry_price:.2f}`).",
                            color=0xF1C40F,
                        )
                elif (
                    self.cfg.risk.time_decay_bars > 0
                    and not self._decay_moved
                    and (self.bar_count - self._entry_submitted_bar) >= self.cfg.risk.time_decay_bars
                ):
                    if sig.direction == "LONG":
                        dec_sl = round_to_tick(sig.entry_price - self.cfg.risk.time_decay_stop_r * one_r)
                        if low_p > dec_sl and dec_sl > self._active_stop_order.auxPrice:
                            self._active_stop_order.auxPrice = dec_sl
                            self.ib.placeOrder(self.contract, self._active_stop_order)
                            self._decay_moved = True
                            logger.info("Time-Decay Stop Tightening: Moved LONG StopOrder to -0.5R (%.2f)", dec_sl)
                            self._send_discord_alert(
                                title=f"⏳ TIME-DECAY STOP (-0.5R) — {self.contract.localSymbol} LONG",
                                description=f"Tightened Stop Loss to **`{dec_sl:.2f}`** after 6 bars without reaching `+1R`.",
                                color=0xE67E22,
                            )
                    else:
                        dec_sl = round_to_tick(sig.entry_price + self.cfg.risk.time_decay_stop_r * one_r)
                        if high_p < dec_sl and dec_sl < self._active_stop_order.auxPrice:
                            self._active_stop_order.auxPrice = dec_sl
                            self.ib.placeOrder(self.contract, self._active_stop_order)
                            self._decay_moved = True
                            logger.info("Time-Decay Stop Tightening: Moved SHORT StopOrder to -0.5R (%.2f)", dec_sl)
                            self._send_discord_alert(
                                title=f"⏳ TIME-DECAY STOP (-0.5R) — {self.contract.localSymbol} SHORT",
                                description=f"Tightened Stop Loss to **`{dec_sl:.2f}`** after 6 bars without reaching `+1R`.",
                                color=0xE67E22,
                            )
            return

        # 4. Check intraday risk guardrails
        allowed, risk_reason = self.risk.can_open_trade(latest_closed_ts, self.bar_count)
        if not allowed:
            logger.info(
                "BAR CLOSE [%s] %s=%.2f | VWAP=%.2f (Z=%+.2f) | EMA9/21=%.2f/%.2f | Hurst=%.2f | KalVel=%+.2f | Status: %s",
                et_ts_str,
                self.contract.localSymbol if self.contract else "MNQ",
                close_p,
                vwap_p,
                vwap_z,
                ema_fast_p,
                ema_slow_p,
                hurst_v,
                kalman_vel,
                risk_reason,
            )
            return

        # 5. Evaluate strategy on latest closed bar
        signal = self.strategy.evaluate_bar(enriched, idx=-1)
        if signal is not None:
            qty = self.risk.calculate_contracts(signal.stop_distance_points)
            self.place_bracket_order(signal, quantity=qty)
        else:
            logger.info(
                "BAR CLOSE [%s] %s=%.2f | VWAP=%.2f (Z=%+.2f) | EMA9/21=%.2f/%.2f | Hurst=%.2f | KalVel=%+.2f | Status: Scanning (No trigger)",
                et_ts_str,
                self.contract.localSymbol if self.contract else "MNQ",
                close_p,
                vwap_p,
                vwap_z,
                ema_fast_p,
                ema_slow_p,
                hurst_v,
                kalman_vel,
            )

    def _on_commission_report(self, trade: Trade, fill, report) -> None:
        """Track realized PnL from IBKR commission reports as positions close."""
        if report.realizedPNL and report.realizedPNL != 0.0 and abs(report.realizedPNL) < 1e7:
            net_pnl = float(report.realizedPNL)
            self.risk.record_trade_closed(net_pnl, exit_bar_index=self.bar_count)
            logger.info(
                "Trade closed | Realized PnL: $%.2f | Session PnL: $%.2f | Trades Today: %d",
                net_pnl,
                self.risk.state.realized_pnl_usd,
                self.risk.state.trades_taken,
            )
            win_str = "💰 TRADE CLOSED (PROFIT)" if net_pnl > 0 else "🛑 TRADE CLOSED (LOSS)"
            color = 0x2ECC71 if net_pnl > 0 else 0xE74C3C
            sym = self.contract.localSymbol if self.contract else self.cfg.contract.symbol
            fill_price = getattr(getattr(fill, "execution", None), "price", 0.0)
            entry_p = self._trade_entry_price or (self._active_signal.entry_price if self._active_signal else fill_price)
            direction = self._trade_direction or (self._active_signal.direction if self._active_signal else "LONG")
            sl_p = self._trade_initial_sl or (self._active_signal.stop_loss if self._active_signal else None)
            tp_p = self._trade_initial_tp or (self._active_signal.take_profit if self._active_signal else None)

            # Compute trade duration in bars
            entry_offset = max(1, (self.bar_count - (self._trade_filled_bar if self._trade_filled_bar >= 0 else self._entry_submitted_bar))) if self._entry_submitted_bar >= 0 else 6

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
            )

            self._send_discord_alert(
                title=f"{win_str} — {sym} ({net_pnl:+.2f} USD)",
                description=(
                    f"**Exit Fill Price:** `{fill_price:.2f}`\n"
                    f"**Trade Realized PnL:** **`${net_pnl:+.2f}`**\n"
                    f"**Daily Session PnL:** `${self.risk.state.realized_pnl_usd:+.2f}`\n"
                    f"**Trades Today:** `{self.risk.state.trades_taken}/{self.cfg.risk.max_daily_trades}`\n"
                    f"_Post-trade context chart (+10 bars after exit) will post automatically in 20 mins._"
                ),
                color=color,
                image_bytes=chart_bytes,
            )

            # Schedule comprehensive 10-bar post-trade review chart
            self._pending_post_trade_chart = {
                "entry_price": entry_p,
                "exit_price": fill_price,
                "sl_price": sl_p,
                "tp_price": tp_p,
                "direction": direction,
                "pnl": net_pnl,
                "trade_duration_bars": entry_offset,
                "exit_bar_count": self.bar_count,
                "target_bar_count": self.bar_count + 10,
            }

            if self.risk.state.halted:
                self.flatten_all(reason=self.risk.state.halt_reason)

    def _on_exec_details(self, trade: Trade, fill) -> None:
        """Send a Discord alert when the bracket's parent ENTRY order actually fills."""
        parent = self._active_parent_order
        if parent is None or self.contract is None:
            return
        if fill.contract.conId != self.contract.conId or fill.execution.orderId != parent.orderId:
            return
        sig = self._active_signal
        e = fill.execution
        self._trade_entry_price = float(e.price)
        self._trade_direction = "LONG" if e.side == "BOT" else "SHORT"
        self._trade_filled_bar = self.bar_count
        logger.info("ENTRY FILLED: %s %s %s @ %.2f", e.side, e.shares, self.contract.localSymbol, e.price)
        desc = f"**Fill Price:** `{e.price:.2f}` | **Qty:** `{int(e.shares)}`\n"
        if sig is not None:
            desc += f"**Stop Loss:** `{sig.stop_loss:.2f}` | **Take Profit:** `{sig.take_profit:.2f}`"
        self._send_discord_alert(
            title=f"✅ ENTRY FILLED — {'BUY' if e.side == 'BOT' else 'SELL'} {int(e.shares)} {self.contract.localSymbol} @ {e.price:.2f}",
            description=desc,
            color=0x2ECC71 if e.side == "BOT" else 0xE74C3C,
        )

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
        stop_t = next((t for t in open_trades if t.order.orderType in ("STP", "STOP")), None)
        tp_t = next((t for t in open_trades if t.order.orderType in ("LMT",) and t.order.parentId), None)
        if stop_t is None:
            logger.warning("Recovered %s position (%s) but no working StopOrder found!", self.contract.localSymbol, pos.position)
            return

        direction = "LONG" if pos.position > 0 else "SHORT"
        mult = float(self.contract.multiplier or self.cfg.contract.point_value)
        entry = round(pos.avgCost / mult, 2)
        stop = float(stop_t.order.auxPrice)
        tp = float(tp_t.order.lmtPrice) if tp_t else entry
        self._trade_entry_price = entry
        self._trade_direction = direction
        self._trade_initial_sl = stop
        self._trade_initial_tp = tp if tp_t else None
        self._trade_filled_bar = self.bar_count
        self._active_signal = TradeSignal(
            timestamp=pd.Timestamp.now(tz="UTC"),
            direction=direction,
            reason="Recovered after restart",
            entry_price=entry,
            stop_loss=stop,
            take_profit=tp,
            stop_distance_points=abs(entry - stop),
            atr=0.0,
            rsi=50.0,
            vwap=entry,
            ema_fast=entry,
            ema_slow=entry,
        )
        self._active_stop_order = stop_t.order
        self._entry_submitted_bar = self.bar_count
        # If the stop is already at/through entry, Break-Even was already applied before the restart.
        self._be_moved = (direction == "LONG" and stop >= entry) or (direction == "SHORT" and stop <= entry)
        self._decay_moved = False
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

        win_rate = (st.wins / st.trades_taken * 100.0) if st.trades_taken > 0 else 0.0
        pnl = st.realized_pnl_usd

        if st.trades_taken == 0:
            color = 0x3498DB
            outcome_icon = "📊 DAILY SUMMARY (NO TRADES)"
        elif pnl > 0:
            color = 0x2ECC71
            outcome_icon = "🏆 DAILY SUMMARY (PROFITABLE DAY)"
        else:
            color = 0xE74C3C
            outcome_icon = "📉 DAILY SUMMARY (NET LOSS)"

        status_line = f"Halted ({st.halt_reason})" if st.halted else "Completed Normally (15:55 ET EOD)"

        logger.info(
            "EOD DAILY SUMMARY [%s] %s | Trades: %d (W:%d / L:%d, %.1f%%) | Net PnL: $%+.2f | Close: %.2f | VWAP: %.2f",
            session_date_str,
            sym,
            st.trades_taken,
            st.wins,
            st.losses,
            win_rate,
            pnl,
            close_p,
            vwap_p,
        )

        self._send_discord_alert(
            title=f"{outcome_icon} — {sym} ({session_date_str})",
            description=(
                f"**Mode:** `{self.mode.upper()}` | **Contract:** `{sym}`\n"
                f"**Session Net PnL:** **`${pnl:+.2f} USD`**\n"
                f"**Trades Taken:** `{st.trades_taken}/{self.cfg.risk.max_daily_trades}` "
                f"(Wins: `{st.wins}` | Losses: `{st.losses}` | WR: `{win_rate:.1f}%`)\n"
                f"**Closing Price (15:55 ET):** `{close_p:.2f}` (VWAP: `{vwap_p:.2f}`)\n"
                f"**Session Range:** `{day_low:.2f} – {day_high:.2f}` (`{day_high - day_low:.2f} pts`)\n"
                f"**Session Status:** `{status_line}`"
            ),
            color=color,
        )

    def run_streaming(self) -> None:
        """Start real-time bar subscription and run the IBKR event loop."""
        self.connect()
        try:
            self.resolve_front_month_contract()
            self.ib.commissionReportEvent += self._on_commission_report
            self.ib.execDetailsEvent += self._on_exec_details

            logger.info(
                "Subscribing to %s real-time bars (%s, RTH=%s)...",
                self.contract.localSymbol,
                self.cfg.strategy.bar_size,
                self.cfg.risk.rth_only,
            )
            self.bars = self.ib.reqHistoricalData(
                self.contract,
                endDateTime="",
                durationStr=self.cfg.strategy.history_duration,
                barSizeSetting=self.cfg.strategy.bar_size,
                whatToShow="TRADES",
                useRTH=self.cfg.risk.rth_only,
                formatDate=1,
                keepUpToDate=False,
            )
            logger.info("Bot is now live and polling 2-min bar closes every 20s. Press Ctrl+C to stop.")
            self._send_discord_alert(
                title=f"🤖 NQ Quant Bot Online — {self.contract.localSymbol} ({self.mode.upper()})",
                description=(
                    f"**Contract:** `{self.contract.localSymbol}` (Expiry `{self.contract.lastTradeDateOrContractMonth}`)\n"
                    f"**Timeframe:** `{self.cfg.strategy.bar_size}` | **Contracts:** `{self.cfg.risk.fixed_contracts}`\n"
                    f"**Window:** `{self.cfg.risk.entry_start_time}–{self.cfg.risk.entry_end_time} ET`"
                ),
                color=0x3498DB,
            )
            # Process the latest closed historical bar immediately on startup
            self._recover_active_trade()
            self._on_bar_update(self.bars, has_new_bar=True)

            # Poll every 20 seconds so both real-time CME subscriptions and 10-min delayed Paper feeds
            # reliably advance on every 2-minute bar close without stalling.
            while self.ib.isConnected():
                self.ib.sleep(20)
                try:
                    fresh_bars = self.ib.reqHistoricalData(
                        self.contract,
                        endDateTime="",
                        durationStr=self.cfg.strategy.history_duration,
                        barSizeSetting=self.cfg.strategy.bar_size,
                        whatToShow="TRADES",
                        useRTH=self.cfg.risk.rth_only,
                        formatDate=1,
                        keepUpToDate=False,
                    )
                    if fresh_bars:
                        self.bars = fresh_bars
                        self._on_bar_update(self.bars, has_new_bar=True)
                except Exception as exc:  # never let one bad bar/timeout kill a bot that manages live stops
                    logger.exception("Error during bar polling/processing (bot continues): %s", exc)
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user (Ctrl+C).")
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
