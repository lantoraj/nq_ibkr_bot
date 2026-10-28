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

import logging
from datetime import datetime
from typing import List, Optional

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
        self._be_moved: bool = False
        self._decay_moved: bool = False

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

        # 1. Check mandatory EOD Flatten
        if self.risk.should_flatten_eod(latest_closed_ts):
            if self.has_open_position_or_orders():
                self.flatten_all(reason="End of Day (15:55 ET) Auto-Flatten")
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
                self.flatten_all(reason=f"Unfilled Entry Order Timeout ({self.cfg.risk.cancel_unfilled_entry_bars} bars)")
                self._active_parent_order = None
                self._active_signal = None
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
                    if new_sl > self._active_stop_order.stopPrice:
                        self._active_stop_order.stopPrice = new_sl
                        self.ib.placeOrder(self.contract, self._active_stop_order)
                        self._be_moved = True
                        logger.info("Moved LONG StopOrder to Break-Even+ (%.2f) after reaching +1R", new_sl)
                elif sig.direction == "SHORT" and low_p <= sig.entry_price - one_r:
                    new_sl = round_to_tick(sig.entry_price - self.cfg.risk.breakeven_offset_points)
                    if new_sl < self._active_stop_order.stopPrice:
                        self._active_stop_order.stopPrice = new_sl
                        self.ib.placeOrder(self.contract, self._active_stop_order)
                        self._be_moved = True
                        logger.info("Moved SHORT StopOrder to Break-Even+ (%.2f) after reaching +1R", new_sl)
                elif (
                    self.cfg.risk.time_decay_bars > 0
                    and not self._decay_moved
                    and (self.bar_count - self._entry_submitted_bar) >= self.cfg.risk.time_decay_bars
                ):
                    if sig.direction == "LONG":
                        dec_sl = round_to_tick(sig.entry_price - self.cfg.risk.time_decay_stop_r * one_r)
                        if low_p > dec_sl and dec_sl > self._active_stop_order.stopPrice:
                            self._active_stop_order.stopPrice = dec_sl
                            self.ib.placeOrder(self.contract, self._active_stop_order)
                            self._decay_moved = True
                            logger.info("Time-Decay Stop Tightening: Moved LONG StopOrder to -0.5R (%.2f)", dec_sl)
                    else:
                        dec_sl = round_to_tick(sig.entry_price + self.cfg.risk.time_decay_stop_r * one_r)
                        if high_p < dec_sl and dec_sl < self._active_stop_order.stopPrice:
                            self._active_stop_order.stopPrice = dec_sl
                            self.ib.placeOrder(self.contract, self._active_stop_order)
                            self._decay_moved = True
                            logger.info("Time-Decay Stop Tightening: Moved SHORT StopOrder to -0.5R (%.2f)", dec_sl)
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
            if self.risk.state.halted:
                self.flatten_all(reason=self.risk.state.halt_reason)

    def run_streaming(self) -> None:
        """Start real-time bar subscription and run the IBKR event loop."""
        self.connect()
        try:
            self.resolve_front_month_contract()
            self.ib.commissionReportEvent += self._on_commission_report

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
            # Process the latest closed historical bar immediately on startup
            self._on_bar_update(self.bars, has_new_bar=True)

            # Poll every 20 seconds so both real-time CME subscriptions and 10-min delayed Paper feeds
            # reliably advance on every 2-minute bar close without stalling.
            while self.ib.isConnected():
                self.ib.sleep(20)
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
