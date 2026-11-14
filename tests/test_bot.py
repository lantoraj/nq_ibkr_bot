"""
Unit and integration tests for the NQ / MNQ Futures IBKR Trading Bot.
Verifies:
- Tick rounding (0.25 increments) and Session-Anchored VWAP daily resets
- EMA + VWAP Crossover & Pullback signal generation
- Intraday Risk Manager guardrails (RTH window, EOD flatten, Max Daily Loss, Consecutive Losses)
- End-to-end Backtester simulation
"""
from __future__ import annotations

import pandas as pd

from backtest import Backtester, generate_synthetic_nq_data
from config import BotConfig
from indicators import compute_session_vwap, enrich_indicators, round_to_tick
from risk_manager import RiskManager
from strategy import NQIntradayStrategy


def test_round_to_tick() -> None:
    assert round_to_tick(20123.11, 0.25) == 20123.00
    assert round_to_tick(20123.14, 0.25) == 20123.25
    assert round_to_tick(20123.89, 0.25) == 20124.00


def test_session_vwap_resets_each_day() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-10-26 09:30:00", tz="America/New_York"),
                pd.Timestamp("2026-10-26 09:31:00", tz="America/New_York"),
                pd.Timestamp("2026-10-27 09:30:00", tz="America/New_York"),
            ],
            "high": [20010.0, 20030.0, 20510.0],
            "low": [19990.0, 20010.0, 20490.0],
            "close": [20000.0, 20020.0, 20500.0],
            "volume": [1000, 1000, 500],
        }
    )
    vwap = compute_session_vwap(df)
    assert vwap.iloc[0] == 20000.0
    assert vwap.iloc[1] == 20010.0
    # Second day first bar must reset strictly to its own typical price (20500.0)
    assert vwap.iloc[2] == 20500.0


def test_risk_manager_guardrails_and_eod_flatten() -> None:
    cfg = BotConfig()
    cfg.risk.max_daily_loss_usd = 200.0
    cfg.risk.max_consecutive_losses = 2
    cfg.risk.cooldown_bars_after_exit = 2
    rm = RiskManager(cfg.risk, cfg.contract)

    # 1. Pre-open (09:31 ET is before 09:35 entry_start_time)
    ts_early = pd.Timestamp("2026-10-27 09:31:00", tz="America/New_York")
    allowed, reason = rm.can_open_trade(ts_early, current_bar_index=1)
    assert not allowed
    assert "Outside allowed RTH" in reason

    # 2. Valid RTH entry window (10:15 ET)
    ts_valid = pd.Timestamp("2026-10-27 10:15:00", tz="America/New_York")
    allowed, _ = rm.can_open_trade(ts_valid, current_bar_index=10)
    assert allowed

    # 3. Record losing trade and test cooldown
    rm.record_trade_closed(-100.0, exit_bar_index=10)
    allowed_cd, reason_cd = rm.can_open_trade(ts_valid, current_bar_index=11)
    assert not allowed_cd
    assert "Cooldown" in reason_cd

    # 4. Second consecutive loss halts trading for the day
    rm.record_trade_closed(-50.0, exit_bar_index=15)
    allowed_halt, reason_halt = rm.can_open_trade(ts_valid, current_bar_index=20)
    assert not allowed_halt
    assert "halted" in reason_halt.lower()

    # 5. Next day automatically resets session state
    ts_next_day = pd.Timestamp("2026-10-28 10:15:00", tz="America/New_York")
    allowed_next, _ = rm.can_open_trade(ts_next_day, current_bar_index=25)
    assert allowed_next

    # 6. Check 15:55 ET End-of-Day auto-flatten trigger
    ts_eod = pd.Timestamp("2026-10-28 15:55:00", tz="America/New_York")
    assert rm.should_flatten_eod(ts_eod)


def test_strategy_and_backtester_end_to_end() -> None:
    cfg = BotConfig()
    cfg.strategy.min_atr_points = 3.0
    df = generate_synthetic_nq_data(num_days=5, seed=42)
    enriched = enrich_indicators(df)
    assert "vwap" in enriched.columns
    assert "ema_fast" in enriched.columns
    assert "atr" in enriched.columns

    bt = Backtester(cfg)
    result = bt.run(df)
    metrics = result["metrics"]
    trades = result["trades"]

    assert metrics["total_trades"] > 0
    for t in trades:
        # Verify tick alignment on stop loss & take profit
        assert round(t.stop_loss % 0.25, 5) == 0.0
        assert round(t.take_profit % 0.25, 5) == 0.0
        if t.direction == "LONG":
            assert t.take_profit > t.entry_price
        else:
            assert t.take_profit < t.entry_price


def test_trade_chart_generation_immediate_and_post_exit() -> None:
    from ibkr_execution import IBKRNQBotRunner
    from ib_async import BarData

    cfg = BotConfig()
    runner = IBKRNQBotRunner(cfg, mode="paper")

    # Mock bar data list
    df = generate_synthetic_nq_data(num_days=1, seed=42)
    fake_bars = []
    for _, r in df.iterrows():
        fake_bars.append(
            BarData(
                date=r["timestamp"],
                open=r["open"],
                high=r["high"],
                low=r["low"],
                close=r["close"],
                volume=r["volume"],
            )
        )
    runner.bars = fake_bars
    runner.bar_count = len(df)
    runner._entry_submitted_bar = len(df) - 10

    # 1. Immediate trade exit chart
    img_bytes = runner._generate_trade_chart(
        entry_price=28648.25,
        exit_price=28535.75,
        sl_price=28685.75,
        tp_price=28535.75,
        direction="SHORT",
        pnl=223.28,
        entry_bar_offset=10,
        post_exit_bars=0,
        pre_entry_bars=15,
    )
    assert img_bytes is not None
    assert img_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    # 2. Delayed post-trade review chart (+10 bars context)
    review_bytes = runner._generate_trade_chart(
        entry_price=28648.25,
        exit_price=28535.75,
        sl_price=28685.75,
        tp_price=28535.75,
        direction="SHORT",
        pnl=223.28,
        entry_bar_offset=10,
        post_exit_bars=10,
        pre_entry_bars=15,
    )
    assert review_bytes is not None
    assert review_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_dual_session_vwap_reanchoring_and_premarket_risk() -> None:
    # 1. Verify VWAP accumulates in Pre-Market (04:00 - 09:28 ET) and cleanly re-anchors at 09:30 ET RTH open
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-11-04 04:00:00", tz="America/New_York"),
                pd.Timestamp("2026-11-04 04:02:00", tz="America/New_York"),
                pd.Timestamp("2026-11-04 09:30:00", tz="America/New_York"),
            ],
            "high": [28010.0, 28030.0, 28410.0],
            "low": [27990.0, 28010.0, 28390.0],
            "close": [28000.0, 28020.0, 28400.0],
            "volume": [1000, 1000, 500],
        }
    )
    vwap = compute_session_vwap(df)
    assert vwap.iloc[0] == 28000.0
    assert vwap.iloc[1] == 28010.0
    # At 09:30 ET Cash Open, VWAP must re-anchor strictly to the RTH opening bar (28400.0)
    assert vwap.iloc[2] == 28400.0

    # 2. Verify Dual-Session RiskManager entry windows and independent Pre-Market / RTH trade budgets
    cfg = BotConfig()
    rm = RiskManager(cfg.risk, cfg.contract)

    # 03:30 ET (Asian session) -> blocked
    allowed_asia, _ = rm.can_open_trade(pd.Timestamp("2026-11-04 03:30:00", tz="America/New_York"), 1)
    assert not allowed_asia

    # 04:30 ET (European Pre-Market) -> allowed
    ts_pm = pd.Timestamp("2026-11-04 04:30:00", tz="America/New_York")
    allowed_pm, _ = rm.can_open_trade(ts_pm, 5)
    assert allowed_pm

    # Exhaust Pre-Market max trades (2 trades)
    rm.record_trade_closed(80.0, exit_bar_index=6)
    rm.record_trade_closed(60.0, exit_bar_index=10)
    allowed_pm_after, reason_pm = rm.can_open_trade(pd.Timestamp("2026-11-04 06:00:00", tz="America/New_York"), 15)
    assert not allowed_pm_after
    assert "pre-market trades reached" in reason_pm

    # 08:30 ET (US Macro News release window) -> blocked
    allowed_news, _ = rm.can_open_trade(pd.Timestamp("2026-11-04 08:30:00", tz="America/New_York"), 20)
    assert not allowed_news

    # 09:50 ET (US RTH Session) -> RTH trade budget is fresh even though 2 premarket trades were taken!
    allowed_rth, _ = rm.can_open_trade(pd.Timestamp("2026-11-04 09:50:00", tz="America/New_York"), 30)
    assert allowed_rth
    assert rm.state.trades_taken == 2
    assert rm.state.rth_trades_taken == 0


def test_delayed_paper_execution_collar_and_unset_double() -> None:
    """Verify StopLimitOrder collar, UNSET_DOUBLE (never NaN) on cashQty, and dynamic SL/TP re-anchoring."""
    from types import SimpleNamespace
    import math
    from ib_async import Future, StopLimitOrder
    from ib_async.objects import UNSET_DOUBLE
    from ibkr_execution import IBKRNQBotRunner
    from strategy import TradeSignal

    cfg = BotConfig()
    runner = IBKRNQBotRunner(cfg, mode="paper")
    runner.contract = Future(conId=815824267, symbol="MNQ", localSymbol="MNQZ6", multiplier="2", exchange="CME", currency="USD")

    placed_orders = []
    runner.ib.client.getReqId = lambda: len(placed_orders) + 100
    runner.ib.placeOrder = lambda contract, order: placed_orders.append(order) or SimpleNamespace(order=order)

    sig = TradeSignal(
        timestamp=pd.Timestamp("2026-11-06 15:30:00", tz="UTC"),
        direction="SHORT",
        reason="Bearish EMA(9/21) Crossover < VWAP",
        entry_price=27100.25,
        stop_loss=27140.25,
        take_profit=26980.25,
        stop_distance_points=40.0,
        atr=35.0,
        rsi=42.0,
        vwap=27150.0,
        ema_fast=27110.0,
        ema_slow=27120.0,
    )

    runner.place_bracket_order(sig, quantity=1)
    assert len(placed_orders) == 3
    parent, tp, sl = placed_orders
    assert isinstance(parent, StopLimitOrder)
    assert parent.action == "SELL"
    assert parent.lmtPrice == 27100.00
    assert parent.auxPrice == 27106.25  # +6.0 pt collar prevents filling at 27147.75 / 27252.00

    # Simulate fill within collar at 27104.25 (+4.0 pts above signal) and verify SL/TP re-anchor & UNSET_DOUBLE
    fake_fill = SimpleNamespace(
        contract=runner.contract,
        execution=SimpleNamespace(
            orderId=parent.orderId,
            price=27104.25,
            side="SLD",
            shares=1.0,
            time=pd.Timestamp("2026-11-06 15:42:50", tz="UTC").to_pydatetime(),
        ),
    )
    runner._on_exec_details(None, fake_fill)
    assert runner._active_stop_order.auxPrice == 27144.25
    assert runner._active_tp_order.lmtPrice == 26984.25
    assert runner._active_stop_order.cashQty == UNSET_DOUBLE
    assert not math.isnan(runner._active_stop_order.cashQty)


def test_tradingview_hybrid_feed_and_dataframe_bars() -> None:
    """Verify TradingView symbol resolution, 1m->2m resampling, basis stitching, and runner DataFrame ingestion."""
    from ibkr_execution import IBKRNQBotRunner
    from tv_data_feed import TradingViewDataFeed, ibkr_symbol_to_tv_cme_symbol, resample_1m_to_target

    assert ibkr_symbol_to_tv_cme_symbol("MNQZ6", "MNQ") == "CME_MINI:MNQZ2026"
    assert ibkr_symbol_to_tv_cme_symbol("NQH7", "NQ") == "CME_MINI:NQH2027"

    # 1m -> 2m resampling check
    df_1m = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-11-06 14:30:00", periods=4, freq="1min", tz="UTC"),
            "open": [27000.0, 27005.0, 27010.0, 27008.0],
            "high": [27006.0, 27012.0, 27015.0, 27014.0],
            "low": [26998.0, 27002.0, 27007.0, 27004.0],
            "close": [27005.0, 27010.0, 27008.0, 27012.0],
            "volume": [100.0, 150.0, 200.0, 120.0],
        }
    )
    df_2m = resample_1m_to_target(df_1m, target_minutes=2)
    assert len(df_2m) == 2
    assert df_2m.iloc[0]["open"] == 27000.0
    assert df_2m.iloc[0]["high"] == 27012.0
    assert df_2m.iloc[0]["low"] == 26998.0
    assert df_2m.iloc[0]["close"] == 27010.0
    assert df_2m.iloc[0]["volume"] == 250.0

    # Basis stitching check: CME delayed by 5 bars (10 mins), proxy up-to-date with +100.0 pt basis
    ts_all = pd.date_range("2026-11-06 14:00:00", periods=20, freq="2min", tz="UTC")
    prx_df = pd.DataFrame(
        {
            "timestamp": ts_all,
            "open": [26900.0 + i for i in range(20)],
            "high": [26905.0 + i for i in range(20)],
            "low": [26895.0 + i for i in range(20)],
            "close": [26902.0 + i for i in range(20)],
            "volume": [100.0] * 20,
        }
    )
    cme_df = prx_df.iloc[:15].copy()
    for col in ("open", "high", "low", "close"):
        cme_df[col] = cme_df[col] + 100.0
    cme_df["volume"] = 500.0

    tv = TradingViewDataFeed()
    tv._fetch_dual_series = lambda **kwargs: (cme_df, "delayed_streaming_600", prx_df, "streaming")
    combined = tv.fetch_bars("MNQZ6", "MNQ", "2 mins", 20)
    assert len(combined) == 20
    assert tv.last_basis_points == 100.0
    assert tv.last_stitched_bars_count == 6
    assert combined.iloc[-1]["close"] == 26902.0 + 19 + 100.0

    # Runner DataFrame bar ingestion check
    cfg = BotConfig()
    runner = IBKRNQBotRunner(cfg, mode="paper")
    parsed_df = runner._bars_to_dataframe(combined)
    assert not parsed_df.empty
    assert list(parsed_df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_v41_adaptive_volatility_gate_and_premarket_cutoff() -> None:
    """Verify v4.2 2-setup engine defaults, London Open PM scalp settings, and RTH blackout windows."""
    cfg = BotConfig()
    assert cfg.strategy.max_crossover_atr == 0.0
    assert cfg.strategy.use_quant_meta_filter is False
    assert cfg.strategy.enable_v_reversal is False
    assert cfg.strategy.min_atr_points == 22.0
    assert cfg.strategy.premarket_min_atr_points == 16.0
    assert cfg.risk.breakeven_offset_points == 2.0
    assert cfg.risk.time_decay_stop_r == 0.40
    assert cfg.risk.premarket_trend_end_time == "05:00"
    assert cfg.risk.premarket_reward_risk_ratio == 2.0
    assert cfg.risk.premarket_breakeven_r == 0.80
    assert cfg.risk.entry_end_time == "14:58"
    assert cfg.risk.rth_blackout_windows == (("10:00", "10:14"), ("11:30", "11:58"))


def test_v42_blackouts_session_risk_isolation_and_marketable_be_guard(tmp_path) -> None:
    """Verify v4.2 blackouts, PM/RTH loss isolation, marketable BE stop guard, stale bar guard, and journal recovery."""
    from types import SimpleNamespace
    from ib_async import BarData, Future, StopOrder
    from ibkr_execution import IBKRNQBotRunner
    from strategy import TradeSignal
    from trade_journal import TradeJournal

    cfg = BotConfig()
    cfg.risk.max_daily_loss_usd = 250.0
    rm = RiskManager(cfg.risk, cfg.contract)

    # 1. RTH Blackout windows & 14:58 cutoff
    allowed_macro, reason_macro = rm.can_open_trade(pd.Timestamp("2026-11-13 10:05:00", tz="America/New_York"), 10)
    assert not allowed_macro and "blackout" in reason_macro.lower()
    allowed_eu, reason_eu = rm.can_open_trade(pd.Timestamp("2026-11-13 11:50:00", tz="America/New_York"), 15)
    assert not allowed_eu and "blackout" in reason_eu.lower()
    allowed_moc, _ = rm.can_open_trade(pd.Timestamp("2026-11-13 15:00:00", tz="America/New_York"), 20)
    assert not allowed_moc
    allowed_ok, _ = rm.can_open_trade(pd.Timestamp("2026-11-13 10:30:00", tz="America/New_York"), 25)
    assert allowed_ok

    # 2. Pre-Market losses do NOT eat into RTH daily loss budget or keep RTH halted after 09:30 ET
    rm_iso = RiskManager(cfg.risk, cfg.contract)
    rm_iso.can_open_trade(pd.Timestamp("2026-11-13 04:20:00", tz="America/New_York"), 5)
    rm_iso.record_trade_closed(-260.0, exit_bar_index=6)
    assert rm_iso.state.halted
    allowed_rth_fresh, _ = rm_iso.can_open_trade(pd.Timestamp("2026-11-13 09:46:00", tz="America/New_York"), 50)
    assert allowed_rth_fresh
    assert not rm_iso.state.halted
    assert rm_iso.state.rth_realized_pnl_usd == 0.0
    assert rm_iso.state.premarket_realized_pnl_usd == -260.0

    # 3. Bug #1 fix: Intra-bar +1R touch followed by reversal past BE stop must NOT submit marketable BUY STOP
    runner = IBKRNQBotRunner(cfg, mode="paper")
    runner.journal = TradeJournal(base_dir=str(tmp_path / "j"), timezone="America/New_York")
    runner.contract = Future(conId=815824267, symbol="MNQ", localSymbol="MNQZ6", multiplier="2", exchange="CME", currency="USD")
    runner.strategy.prepare_dataframe = lambda d: pd.DataFrame(
        {
            "timestamp": d["timestamp"],
            "open": d["open"],
            "high": d["high"],
            "low": d["low"],
            "close": d["close"],
            "volume": d["volume"],
            "vwap": d["close"],
            "ema_fast": d["close"],
            "ema_slow": d["close"],
            "ema_trend": d["close"],
            "vwap_z": 0.5,
            "hurst": 0.5,
            "kalman_vel_atr": 0.0,
            "atr": 25.0,
            "rsi": 50.0,
            "vol_ratio": 1.0,
            "ema_spread_atr": 0.2,
            "slow_slope_atr": 0.05,
            "body_ratio": 0.5,
            "upper_wick_ratio": 0.1,
            "lower_wick_ratio": 0.1,
            "prev_3_high": d["high"],
            "prev_3_low": d["low"],
            "orb_high": float("nan"),
            "orb_low": float("nan"),
            "pdh": float("nan"),
            "pdl": float("nan"),
            "roll_vwap_z_max": 0.5,
            "roll_vwap_z_min": 0.5,
        }
    )

    placed = []
    runner.ib.placeOrder = lambda contract, order: placed.append(order) or SimpleNamespace(order=order)
    runner.ib.positions = lambda: [SimpleNamespace(contract=runner.contract, position=-1.0)]
    stp = StopOrder(action="BUY", totalQuantity=1, stopPrice=27776.25)
    runner.ib.openTrades = lambda: [
        SimpleNamespace(contract=runner.contract, order=stp, orderStatus=SimpleNamespace(status="Submitted"))
    ]
    runner._active_stop_order = stp
    runner._trade_entry_price = 27736.00
    runner._trade_direction = "SHORT"
    runner._trade_initial_sl = 27776.25
    runner._trade_filled_ts_utc = pd.Timestamp("2026-11-13 16:14:00", tz="UTC")
    runner._active_signal = TradeSignal(
        timestamp=pd.Timestamp("2026-11-13 16:14:00", tz="UTC"),
        direction="SHORT",
        reason="Bearish Pullback",
        entry_price=27736.00,
        stop_loss=27776.25,
        take_profit=27616.25,
        stop_distance_points=40.0,
        atr=35.0,
        rsi=45.0,
        vwap=27750.0,
        ema_fast=27740.0,
        ema_slow=27750.0,
    )

    # Wicked to 27695.75 (+1R touched) but closed at 27760.25 (> BE stop 27734.00) -> must defer BE!
    bar_df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-11-13 16:20:00", tz="UTC"),
                pd.Timestamp("2026-11-13 16:22:00", tz="UTC"),
            ],
            "open": [27735.0, 27720.0],
            "high": [27740.0, 27762.0],
            "low": [27718.0, 27695.75],
            "close": [27720.0, 27760.25],
            "volume": [1000, 1500],
        }
    )
    # Note: _bars_to_dataframe treats DataFrame rows with ts + 2m <= now as closed (16:22 UTC is in the past)
    runner._on_bar_update(bar_df, has_new_bar=True)
    assert not runner._be_moved
    assert stp.auxPrice == 27776.25
    assert len(placed) == 0

    # Stale out-of-order bar check (older timestamp must be ignored)
    prev_count = runner.bar_count
    stale_df = bar_df.iloc[:1].copy()
    runner._on_bar_update(stale_df, has_new_bar=True)
    assert runner.bar_count == prev_count

    # 4. Journal state recovery across container restart
    now_utc = pd.Timestamp.now(tz="UTC")
    j_recover = TradeJournal(base_dir=str(tmp_path / "j_rec"), timezone="America/New_York")
    j_recover.log("trade_closed", bar_ts=now_utc, direction="LONG", net_pnl_usd=-120.0)
    runner2 = IBKRNQBotRunner(cfg, mode="paper")
    runner2.journal = j_recover
    runner2._restore_daily_risk_state()
    assert runner2.risk.state.trades_taken == 1
    assert runner2.risk.state.realized_pnl_usd == -120.0
    assert runner2.risk.state.consecutive_losses == 1


def test_trade_journal_and_reconcile(tmp_path) -> None:
    """Verify append-only JSONL TradeJournal and live-vs-backtest reconciliation."""
    from reconcile import reconcile_day
    from trade_journal import TradeJournal, load_journal

    jdir = str(tmp_path / "journal")
    j = TradeJournal(base_dir=jdir, timezone="America/New_York")
    ts = pd.Timestamp("2026-11-06 15:00:00", tz="UTC")
    j.log("bar", bar_ts=ts, open=27000.0, high=27010.0, low=26995.0, close=27005.0, volume=500.0)
    j.log("signal", bar_ts=ts, direction="LONG", reason="Bullish Pullback", entry_price=27005.0, atr=18.5)
    j.log("entry_filled", bar_ts=ts, direction="LONG", signal_price=27005.0, fill_price=27005.25, slippage_points=0.25, fill_delay_sec=1.4)
    j.log("trade_closed", bar_ts=ts, direction="LONG", signal_price=27005.0, entry_price=27005.25, exit_price=27050.25, net_pnl_usd=88.76)

    events = load_journal("2026-11-06", base_dir=jdir)
    assert len(events) == 4
    res = reconcile_day("2026-11-06", base_dir=jdir)
    assert res["bars_logged"] == 1
    assert res["live_signals"] == 1
    assert res["live_fills"] == 1
    assert res["live_closed"] == 1
    assert res["avg_slippage_pts"] == 0.25
    assert res["live_net_pnl_usd"] == 88.76



