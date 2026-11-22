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

    # 2. Verify RTH-only default blocks Pre-Market, and opt-in Dual-Session enables isolated Pre-Market / RTH budgets
    default_cfg = BotConfig()
    assert default_cfg.risk.use_premarket_session is False
    rm_default = RiskManager(default_cfg.risk, default_cfg.contract)
    ts_pm = pd.Timestamp("2026-11-04 04:30:00", tz="America/New_York")
    allowed_pm_default, _ = rm_default.can_open_trade(ts_pm, 5)
    assert not allowed_pm_default

    cfg = BotConfig()
    cfg.risk.use_premarket_session = True
    rm = RiskManager(cfg.risk, cfg.contract)

    # 03:30 ET (Asian session) -> blocked
    allowed_asia, _ = rm.can_open_trade(pd.Timestamp("2026-11-04 03:30:00", tz="America/New_York"), 1)
    assert not allowed_asia

    # 04:30 ET (European Pre-Market, opt-in enabled) -> allowed
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
    assert parent.lmtPrice == 27098.25  # 8 ticks (2.00 pts) marketable below 27100.25 signal
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

    # 2. Default config blocks 04:20 ET pre-market; when PM is enabled, PM losses do NOT halt RTH after 09:30 ET
    assert not RiskManager(cfg.risk, cfg.contract).can_open_trade(pd.Timestamp("2026-11-13 04:20:00", tz="America/New_York"), 5)[0]
    cfg.risk.use_premarket_session = True
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





def test_economic_calendar_tier_classification_and_blackouts(tmp_path):
    from datetime import date
    import pandas as pd
    from economic_calendar import EconomicCalendar

    cal = EconomicCalendar(
        cache_path=str(tmp_path / "ff_cache.json"),
        report_path=str(tmp_path / "ff_report.json"),
    )
    assert (cal.tier1_pre_mins, cal.tier1_post_mins) == (10, 10)
    assert (cal.tier2_pre_mins, cal.tier2_post_mins) == (5, 5)

    count = cal.load_events_from_list(
        [
            {
                "title": "CPI m/m",
                "country": "USD",
                "date": "2026-11-10T08:30:00-05:00",
                "impact": "High",
                "forecast": "0.3%",
                "previous": "0.2%",
            },
            {
                "title": "FOMC Statement",
                "country": "USD",
                "date": "2026-11-11T14:00:00-05:00",
                "impact": "High",
                "forecast": "",
                "previous": "",
            },
            {
                "title": "Fed Chairman Warsh Speaks",
                "country": "USD",
                "date": "2026-11-13T14:00:00-05:00",
                "impact": "High",
                "forecast": "",
                "previous": "",
            },
            {
                "title": "Prelim UoM Consumer Sentiment",
                "country": "USD",
                "date": "2026-11-13T10:30:00-05:00",
                "impact": "High",
                "forecast": "71.0",
                "previous": "70.5",
            },
            {
                "title": "ECB President Lagarde Speaks",
                "country": "EUR",
                "date": "2026-11-13T10:15:00-05:00",
                "impact": "Medium",
                "forecast": "",
                "previous": "",
            },
        ]
    )
    assert count == 5
    tiers = {ev.title: ev.tier for ev in cal.events}
    assert tiers["CPI m/m"] == 1
    assert tiers["FOMC Statement"] == 1
    assert tiers["Fed Chairman Warsh Speaks"] == 1
    assert tiers["Prelim UoM Consumer Sentiment"] == 2
    assert tiers["ECB President Lagarde Speaks"] == 0

    # Tier-1 FOMC window: 13:50 - 14:10 ET (-10m / +10m)
    assert cal.is_in_news_blackout(pd.Timestamp("2026-11-11 13:49:00-05:00")) == (False, "")
    in_b, reason = cal.is_in_news_blackout(pd.Timestamp("2026-11-11 13:50:00-05:00"))
    assert in_b and "Tier-1" in reason and "FOMC Statement" in reason
    assert cal.is_in_news_blackout(pd.Timestamp("2026-11-11 14:10:00-05:00"))[0] is True
    assert cal.is_in_news_blackout(pd.Timestamp("2026-11-11 14:11:00-05:00")) == (False, "")

    # Tier-2 window: 10:25 - 10:35 ET (-5m / +5m)
    assert cal.is_in_news_blackout(pd.Timestamp("2026-11-13 10:24:00-05:00"))[0] is False
    assert cal.is_in_news_blackout(pd.Timestamp("2026-11-13 10:25:00-05:00"))[0] is True
    assert cal.is_in_news_blackout(pd.Timestamp("2026-11-13 10:35:00-05:00"))[0] is True
    assert cal.is_in_news_blackout(pd.Timestamp("2026-11-13 10:36:00-05:00"))[0] is False

    # Non-USD Medium does not block
    assert cal.is_in_news_blackout(pd.Timestamp("2026-11-13 10:15:00-05:00"))[0] is False

    # Daily CET schedule, Discord formatter, and ForexFactory-style PNG table renderer
    nov13_events = cal.get_events_for_cet_date(date(2026, 11, 13))
    assert len(nov13_events) == 3
    title, desc = cal.format_daily_discord_message(date(2026, 11, 13), nov13_events)
    assert "09:00 CET" in title
    assert "Prelim UoM Consumer Sentiment" in desc
    assert "Fed Chairman Warsh Speaks" in desc
    assert "Full Day Schedule" not in desc
    assert "ECB President Lagarde Speaks" not in desc
    img_bytes = cal.render_daily_calendar_image(date(2026, 11, 13), nov13_events)
    assert img_bytes is not None and img_bytes.startswith(b"\x89PNG")

    # Report export verification
    import json
    report_file = tmp_path / "ff_report.json"
    cal.export_report_json(str(report_file))
    exported = json.loads(report_file.read_text(encoding="utf-8"))
    assert len(exported["events"]) == 5
    assert exported["rules"] == {
        "tier1_pre_mins": 10,
        "tier1_post_mins": 10,
        "tier2_pre_mins": 5,
        "tier2_post_mins": 5,
    }
    assert exported["events"][0]["time_et"] == "08:30"
    assert exported["events"][0]["time_cet"] == "14:30"


def test_risk_manager_blocks_during_news_blackout(tmp_path):
    import pandas as pd
    from config import BotConfig
    from economic_calendar import EconomicCalendar
    from risk_manager import RiskManager

    cal = EconomicCalendar(cache_path=str(tmp_path / "ff_cache.json"), report_path=None)
    cal.load_events_from_list(
        [
            {
                "title": "FOMC Statement",
                "country": "USD",
                "date": "2026-11-11T14:00:00-05:00",
                "impact": "High",
            }
        ]
    )
    cfg = BotConfig()
    rm = RiskManager(cfg.risk, cfg.contract, calendar=cal)

    ok_block, reason_block = rm.can_open_trade(pd.Timestamp("2026-11-11 14:05:00-05:00"), 100)
    assert ok_block is False
    assert "FOMC Statement" in reason_block

    ok_after, _ = rm.can_open_trade(pd.Timestamp("2026-11-11 14:35:00-05:00"), 115)
    assert ok_after is True

    # Filter toggle disables news blocking cleanly
    cfg.risk.use_news_calendar_filter = False
    ok_disabled, _ = rm.can_open_trade(pd.Timestamp("2026-11-11 14:05:00-05:00"), 100)
    assert ok_disabled is True


def test_upcoming_event_within_and_pre_news_stop_tightening(tmp_path):
    import pandas as pd
    from economic_calendar import EconomicCalendar
    from ibkr_execution import compute_pre_news_tightened_stop

    cal = EconomicCalendar(cache_path=str(tmp_path / "ff_cache.json"), report_path=None)
    cal.load_events_from_list(
        [
            {
                "title": "FOMC Statement",
                "country": "USD",
                "date": "2026-11-11T14:00:00-05:00",
                "impact": "High",
            }
        ]
    )
    assert cal.upcoming_event_within(pd.Timestamp("2026-11-11 13:55:00-05:00"), minutes=4) is None
    ev = cal.upcoming_event_within(pd.Timestamp("2026-11-11 13:56:00-05:00"), minutes=4)
    assert ev is not None and ev.title == "FOMC Statement"
    assert cal.upcoming_event_within(pd.Timestamp("2026-11-11 14:01:00-05:00"), minutes=4) is None

    # LONG >= +0.30R moves SL to BE (+1.5 pts)
    sl_be, kind_be = compute_pre_news_tightened_stop(
        direction="LONG",
        entry_ref=25000.0,
        one_r=20.0,
        current_sl=24980.0,
        close_p=25008.0,
        be_offset_pts=1.5,
        decay_stop_r=0.40,
    )
    assert (sl_be, kind_be) == (25001.5, "pre_news_be")

    # LONG < +0.30R tightens SL to -0.40R (24992.0)
    sl_dec, kind_dec = compute_pre_news_tightened_stop(
        direction="LONG",
        entry_ref=25000.0,
        one_r=20.0,
        current_sl=24980.0,
        close_p=25002.0,
        be_offset_pts=1.5,
        decay_stop_r=0.40,
    )
    assert (sl_dec, kind_dec) == (24992.0, "pre_news_tighten")

    # Existing tighter stop is not loosened
    assert compute_pre_news_tightened_stop("LONG", 25000.0, 20.0, 25005.0, 25008.0, 1.5, 0.40) == (None, "")

    # SHORT cases
    assert compute_pre_news_tightened_stop("SHORT", 25000.0, 20.0, 25020.0, 24993.0, 1.5, 0.40) == (24998.5, "pre_news_be")
    assert compute_pre_news_tightened_stop("SHORT", 25000.0, 20.0, 25020.0, 25002.0, 1.5, 0.40) == (25008.0, "pre_news_tighten")


def test_backtester_2ct_partial_scale_out_rth_and_pm() -> None:
    """Verify 2-contract Backtester locks TP1 (+2R RTH / +1.4R PM), moves runner SL to BE (+2pt), and targets +5R / +2R."""
    from backtest import Backtester
    from strategy import TradeSignal

    cfg = BotConfig()
    cfg.risk.fixed_contracts = 2
    cfg.risk.max_daily_loss_usd = 500.0
    cfg.risk.enable_partial_tp = True
    cfg.risk.partial_tp1_r = 2.0
    cfg.risk.runner_reward_risk_ratio = 5.0
    cfg.risk.premarket_partial_tp1_r = 1.4
    cfg.risk.premarket_reward_risk_ratio = 2.0
    cfg.risk.breakeven_offset_points = 2.0

    # Construct 4 RTH 2-min bars on 2026-11-12 starting 10:20 ET (outside blackouts):
    # Bar 0: signal fires @ 25000.0 (slippage fill 25000.25), 1R=20 -> SL=24980, TP1=25040.25, TP2=25100
    # Bar 1: touches +2.0R (high=25042) and closes 25035 -> Leg 1 closes @ TP1 (+40 pts), SL moves to BE (25002.25)
    # Bar 2: spikes to 25102 -> Runner Leg 2 closes @ TP2 (+99.75 pts from 25000.25)
    ts = pd.date_range("2026-11-12 10:20:00", periods=3, freq="2min", tz="America/New_York").tz_convert("UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [24998.0, 25001.0, 25035.0],
            "high": [25002.0, 25042.0, 25102.0],
            "low": [24995.0, 25000.0, 25030.0],
            "close": [25000.0, 25035.0, 25098.0],
            "volume": [1000, 1200, 1400],
        }
    )

    bt = Backtester(cfg, commission_per_side_usd=0.62, slippage_ticks=1)
    bt.strategy.prepare_dataframe = lambda d: d
    bt.strategy.evaluate_bar = lambda d, idx=-1: (
        TradeSignal(
            timestamp=d.iloc[idx]["timestamp"],
            direction="LONG",
            reason="Test Scale-Out",
            entry_price=25000.0,
            stop_loss=24980.0,
            take_profit=25060.0,
            stop_distance_points=20.0,
            atr=25.0,
            rsi=55.0,
            vwap=24995.0,
            ema_fast=24998.0,
            ema_slow=24992.0,
        )
        if idx == 0
        else None
    )

    res = bt.run(df)
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t.contracts == 2
    assert t.take_profit == 25100.0  # +5.0R runner ceiling
    assert t.exit_reason == "TP1 (+2.0R) + Take Profit"
    # Leg 1: (25040.25 - 25000.25)*$2 - $1.24 = $80 - $1.24 = $78.76
    # Leg 2: (25100.00 - 25000.25)*$2 - $1.24 = $199.50 - $1.24 = $198.26
    assert abs(t.net_pnl_usd - 277.02) < 1e-2

    # Second scenario: Leg 1 fills @ +2R, Runner pulls back on Bar 2 to Break-Even (25002.25 - 0.25 slip = 25002.0)
    df_be = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [24998.0, 25001.0, 25035.0],
            "high": [25002.0, 25042.0, 25038.0],
            "low": [24995.0, 25000.0, 25000.0],
            "close": [25000.0, 25035.0, 25005.0],
            "volume": [1000, 1200, 1400],
        }
    )
    bt2 = Backtester(cfg, commission_per_side_usd=0.62, slippage_ticks=1)
    bt2.strategy.prepare_dataframe = lambda d: d
    bt2.strategy.evaluate_bar = bt.strategy.evaluate_bar
    res_be = bt2.run(df_be)
    assert len(res_be["trades"]) == 1
    t_be = res_be["trades"][0]
    assert t_be.exit_reason == "TP1 (+2.0R) + Break-Even Stop"
    # Leg 1 = $78.76; Leg 2 = (25002.00 - 25000.25)*$2 - $1.24 = $3.50 - $1.24 = $2.26 -> Total $81.02
    assert abs(t_be.net_pnl_usd - 81.02) < 1e-2


def test_runner_2ct_scale_out_brackets_and_commission_aggregation(tmp_path) -> None:
    """Verify IBKRNQBotRunner submits 2 independent brackets for 2 MNQ contracts and aggregates multi-leg PnL once."""
    from types import SimpleNamespace
    from ib_async import Future
    from ibkr_execution import IBKRNQBotRunner
    from strategy import TradeSignal
    from trade_journal import TradeJournal

    cfg = BotConfig()
    cfg.risk.fixed_contracts = 2
    cfg.risk.max_daily_loss_usd = 500.0
    cfg.risk.enable_partial_tp = True
    cfg.risk.partial_tp1_r = 2.0
    cfg.risk.runner_reward_risk_ratio = 5.0

    runner = IBKRNQBotRunner(cfg, mode="paper")
    runner.journal = TradeJournal(base_dir=str(tmp_path / "j"), timezone="America/New_York")
    runner.contract = Future(conId=815824267, symbol="MNQ", localSymbol="MNQZ6", multiplier="2", exchange="CME", currency="USD")
    runner._send_discord_alert = lambda *a, **k: None

    req_counter = iter(range(1000, 1050))
    runner.ib.client = SimpleNamespace(getReqId=lambda: next(req_counter), isConnected=lambda: False)
    placed_orders = []
    runner.ib.placeOrder = lambda c, o: placed_orders.append(o) or SimpleNamespace(order=o)
    runner.ib.openTrades = lambda: []

    sig = TradeSignal(
        timestamp=pd.Timestamp("2026-11-12 10:20:00-05:00"),
        direction="LONG",
        reason="Bullish Pullback",
        entry_price=25000.0,
        stop_loss=24980.0,
        take_profit=25060.0,
        stop_distance_points=20.0,
        atr=25.0,
        rsi=55.0,
        vwap=24995.0,
        ema_fast=24998.0,
        ema_slow=24992.0,
    )
    runner.place_bracket_order(sig, quantity=2)

    # Expect 6 orders: p1, tp1 (+2R = 25040), sl1, p2, tp2 (+5R = 25100), sl2
    assert len(placed_orders) == 6
    assert len(runner._active_parent_orders) == 2
    assert runner._active_tp1_order.lmtPrice == 25040.0
    assert runner._active_tp_order.lmtPrice == 25100.0
    assert runner._active_tp1_order.totalQuantity == 1
    assert runner._active_tp_order.totalQuantity == 1

    # Simulate both parent entry legs filling @ 25000.25
    p1, p2 = runner._active_parent_orders
    fill_t = pd.Timestamp("2026-11-12 15:20:05", tz="UTC")
    for p_ord, ex_id in [(p1, "E1"), (p2, "E2")]:
        fill_obj = SimpleNamespace(
            contract=runner.contract,
            execution=SimpleNamespace(orderId=p_ord.orderId, execId=ex_id, side="BOT", shares=1, price=25000.25, time=fill_t),
        )
        runner._on_exec_details(SimpleNamespace(), fill_obj)

    assert runner._current_trade_filled_qty == 2
    # Both TPs & SLs re-anchored to fill 25000.25
    assert runner._active_tp1_order.lmtPrice == 25040.25
    assert runner._active_tp_order.lmtPrice == 25100.25
    assert runner._active_stop_orders[0].auxPrice == 24980.25
    assert runner._active_stop_orders[1].auxPrice == 24980.25

    # Simulate TP1 fill -> moves remaining runner stop immediately to BE (25002.25) and waits before closing trade
    tp1_exec = SimpleNamespace(orderId=runner._active_tp1_order.orderId, shares=1, price=25040.25, time=fill_t + pd.Timedelta(minutes=2))
    runner._on_exec_details(SimpleNamespace(), SimpleNamespace(contract=runner.contract, execution=tp1_exec))
    assert runner._tp1_filled is True
    assert runner._be_moved is True
    assert len(runner._active_stop_orders) == 1
    assert runner._active_stop_orders[0].auxPrice == 25002.25

    runner._on_commission_report(
        SimpleNamespace(),
        SimpleNamespace(execution=tp1_exec),
        SimpleNamespace(realizedPNL=78.76, commission=1.24),
    )
    assert runner.risk.state.trades_taken == 0  # Still open with Runner leg
    assert runner.risk.state.realized_pnl_usd == 0.0

    # Runner exits @ 25100.25 (+$198.76) -> finalizes trade ONCE with total $277.52
    tp2_exec = SimpleNamespace(orderId=runner._active_tp_order.orderId, shares=1, price=25100.25, time=fill_t + pd.Timedelta(minutes=6))
    runner._on_commission_report(
        SimpleNamespace(),
        SimpleNamespace(execution=tp2_exec),
        SimpleNamespace(realizedPNL=198.76, commission=1.24),
    )
    assert runner.risk.state.trades_taken == 1
    assert abs(runner.risk.state.realized_pnl_usd - 277.52) < 1e-6

    # Verify stop-out on both legs counts as 1 trade & 1 consecutive loss (not 2)
    req_counter2 = iter(range(2000, 2050))
    runner.ib.client = SimpleNamespace(getReqId=lambda: next(req_counter2), isConnected=lambda: False)
    placed_orders.clear()
    runner.place_bracket_order(sig, quantity=2)
    p1b, p2b = runner._active_parent_orders
    for p_ord in (p1b, p2b):
        runner._on_exec_details(
            SimpleNamespace(),
            SimpleNamespace(contract=runner.contract, execution=SimpleNamespace(orderId=p_ord.orderId, side="BOT", shares=1, price=25000.0, time=fill_t)),
        )
    sl1, sl2 = runner._active_stop_orders
    runner._on_commission_report(
        SimpleNamespace(),
        SimpleNamespace(execution=SimpleNamespace(orderId=sl1.orderId, shares=1, price=24980.0, time=fill_t)),
        SimpleNamespace(realizedPNL=-41.24, commission=1.24),
    )
    assert runner.risk.state.trades_taken == 1  # Still waiting for sl2 commission report
    runner._on_commission_report(
        SimpleNamespace(),
        SimpleNamespace(execution=SimpleNamespace(orderId=sl2.orderId, shares=1, price=24980.0, time=fill_t)),
        SimpleNamespace(realizedPNL=-41.24, commission=1.24),
    )
    assert runner.risk.state.trades_taken == 2
    assert runner.risk.state.consecutive_losses == 1  # 1 losing trade, NOT 2!
    assert abs(runner.risk.state.realized_pnl_usd - (277.52 - 82.48)) < 1e-6


def test_v42_execution_bugfixes_self_heal_race_and_tp_reanchor_transmit(tmp_path) -> None:
    """Verify max_atr_points=72.0, transmit=True & dead-order skipping on re-anchor, and TP fill self-heal race block."""
    from types import SimpleNamespace
    from ib_async import Future
    from ibkr_execution import IBKRNQBotRunner
    from strategy import TradeSignal
    from trade_journal import TradeJournal

    # 1. Config ATR cap tuned to 72.0 (2Y RTH-Only global optimum preserving [58-72] trend runners while blocking >=75 panic whipsaws)
    cfg = BotConfig()
    assert cfg.strategy.max_atr_points == 72.0

    # 2. Re-anchoring transmits TP orders (transmit=True) & skips terminal orders from ib.trades()
    cfg2 = BotConfig()
    cfg2.risk.fixed_contracts = 2
    cfg2.risk.max_daily_loss_usd = 500.0
    cfg2.risk.enable_partial_tp = True
    runner = IBKRNQBotRunner(cfg2, mode="paper")
    runner.journal = TradeJournal(base_dir=str(tmp_path / "j_reanchor"), timezone="America/New_York")
    runner.contract = Future(conId=815824267, symbol="MNQ", localSymbol="MNQZ6", multiplier="2", exchange="CME", currency="USD")
    runner._send_discord_alert = lambda *a, **k: None

    req_counter = iter(range(3000, 3050))
    runner.ib.client = SimpleNamespace(getReqId=lambda: next(req_counter), isConnected=lambda: False)
    placed_orders = []
    runner.ib.placeOrder = lambda c, o: placed_orders.append(o) or SimpleNamespace(order=o)
    runner.ib.openTrades = lambda: []
    runner.ib.trades = lambda: []

    sig = TradeSignal(
        timestamp=pd.Timestamp("2026-11-18 11:00:00-05:00"),
        direction="LONG",
        reason="Bullish Pullback",
        entry_price=25000.0,
        stop_loss=24980.0,
        take_profit=25060.0,
        stop_distance_points=20.0,
        atr=30.0,
        rsi=55.0,
        vwap=24995.0,
        ema_fast=24998.0,
        ema_slow=24992.0,
    )
    runner.place_bracket_order(sig, quantity=2)
    sl1, sl2 = runner._active_stop_orders
    tp1 = runner._active_tp1_order
    tp2 = runner._active_tp_order
    p1 = runner._active_parent_orders[0]

    # Simulate Leg 2 stop sl2 already cancelled in IBKR trades() (absent from openTrades())
    runner.ib.trades = lambda: [
        SimpleNamespace(contract=runner.contract, order=sl2, orderStatus=SimpleNamespace(status="Cancelled")),
        SimpleNamespace(contract=runner.contract, order=sl1, orderStatus=SimpleNamespace(status="Submitted")),
    ]
    placed_orders.clear()
    fill_t = pd.Timestamp("2026-11-18 16:00:05", tz="UTC")
    runner._on_exec_details(
        SimpleNamespace(),
        SimpleNamespace(contract=runner.contract, execution=SimpleNamespace(orderId=p1.orderId, side="BOT", shares=1, price=25001.0, time=fill_t)),
    )
    assert tp1.transmit is True
    assert tp2.transmit is True
    assert sl1.transmit is True
    assert tp1 in placed_orders and tp2 in placed_orders and sl1 in placed_orders
    assert sl2.orderId not in [getattr(o, "orderId", None) for o in placed_orders]

    # 3. TP closing fill blocks orphan self-healing StopOrder even if ib.positions() is briefly stale
    cfg_d = BotConfig()
    cfg_d.contract.symbol = "NQ"
    cfg_d.risk.fixed_contracts = 1
    cfg_d.risk.enable_partial_tp = False
    cfg_d.risk.max_daily_loss_usd = 2500.0
    runner_d = IBKRNQBotRunner(cfg_d, mode="paper")
    runner_d.journal = TradeJournal(base_dir=str(tmp_path / "j_race"), timezone="America/New_York")
    runner_d.contract = Future(conId=563947726, symbol="NQ", localSymbol="NQZ6", multiplier="20", exchange="CME", currency="USD")
    runner_d._send_discord_alert = lambda *a, **k: None
    runner_d._generate_trade_chart = lambda *a, **k: None

    req_counter_d = iter(range(4000, 4050))
    runner_d.ib.client = SimpleNamespace(getReqId=lambda: next(req_counter_d), isConnected=lambda: False)
    placed_d = []
    cancelled_d = []
    lingering_ord = SimpleNamespace(orderId=4999, orderType="STP")
    open_trades_d = []
    runner_d.ib.placeOrder = lambda c, o: placed_d.append(o) or SimpleNamespace(order=o)
    runner_d.ib.cancelOrder = lambda o: cancelled_d.append(o)
    runner_d.ib.openTrades = lambda: open_trades_d
    runner_d.ib.trades = lambda: open_trades_d
    runner_d.ib.positions = lambda: []

    sig_d = TradeSignal(
        timestamp=pd.Timestamp("2026-11-18 12:18:00-05:00"),
        direction="LONG",
        reason="Bullish Pullback",
        entry_price=27143.5,
        stop_loss=27103.5,
        take_profit=27263.5,
        stop_distance_points=40.0,
        atr=50.2,
        rsi=62.4,
        vwap=27112.4,
        ema_fast=27135.0,
        ema_slow=27125.0,
    )
    runner_d.place_bracket_order(sig_d, quantity=1)
    p_d = runner_d._active_parent_orders[0]
    tp_d = runner_d._active_tp_order
    fill_d_t = pd.Timestamp("2026-11-18 17:18:07", tz="UTC")
    runner_d._on_exec_details(
        SimpleNamespace(),
        SimpleNamespace(contract=runner_d.contract, execution=SimpleNamespace(orderId=p_d.orderId, side="BOT", shares=1, price=27143.75, time=fill_d_t)),
    )

    # TP Limit fills via execDetails BEFORE commissionReport; positions() is momentarily stale at +1
    placed_d.clear()
    runner_d.ib.positions = lambda: [SimpleNamespace(contract=runner_d.contract, position=1.0)]
    tp_fill_t = pd.Timestamp("2026-11-18 17:26:04", tz="UTC")
    tp_exit_exec = SimpleNamespace(orderId=tp_d.orderId, side="SLD", shares=1, price=27263.50, time=tp_fill_t)
    runner_d._on_exec_details(SimpleNamespace(), SimpleNamespace(contract=runner_d.contract, execution=tp_exit_exec))

    ts_bars = pd.date_range("2026-11-18 15:30:00", periods=60, freq="2min", tz="UTC")
    bar_df = pd.DataFrame(
        {
            "timestamp": ts_bars,
            "open": [27200.0 + i * 0.5 for i in range(60)],
            "high": [27210.0 + i * 0.5 for i in range(60)],
            "low": [27190.0 + i * 0.5 for i in range(60)],
            "close": [27205.0 + i * 0.5 for i in range(60)],
            "volume": [1000] * 60,
        }
    )
    runner_d._on_bar_update(bar_df, has_new_bar=True)
    assert len(placed_d) == 0  # NO orphan self-healed StopOrder submitted!

    # Lingering open order on same contract is cancelled when full trade closes in _on_commission_report
    open_trades_d.append(
        SimpleNamespace(contract=runner_d.contract, order=lingering_ord, orderStatus=SimpleNamespace(status="Submitted"))
    )
    runner_d._on_commission_report(
        SimpleNamespace(),
        SimpleNamespace(execution=tp_exit_exec),
        SimpleNamespace(realizedPNL=2390.52, commission=2.24),
    )
    assert lingering_ord in cancelled_d
    assert runner_d.risk.state.trades_taken == 1
    assert abs(runner_d.risk.state.realized_pnl_usd - 2390.52) < 1e-6


def test_p1_entry_guards_and_p3_dynamic_sl_cap(tmp_path) -> None:
    """Verify P1 marketable-limit/fast-cancel/runaway-cancel/60s-timeout/fill-safety and P3_SL ATR>60 cap."""
    import json
    from types import SimpleNamespace
    from ib_async import Future
    from ibkr_execution import IBKRNQBotRunner
    from strategy import TradeSignal
    from trade_journal import TradeJournal

    # 1. Config defaults match P1 + P3_SL
    cfg = BotConfig()
    assert cfg.risk.entry_limit_slippage_ticks == 8
    assert cfg.risk.cancel_unfilled_entry_bars == 1
    assert cfg.risk.cancel_unfilled_entry_seconds == 60.0
    assert cfg.risk.runaway_cancel_points == 15.0
    assert cfg.risk.max_stop_points == 40.0
    assert cfg.risk.high_vol_atr_threshold == 60.0
    assert cfg.risk.high_vol_max_stop_points == 48.0

    # 2. P3_SL in NQIntradayStrategy.evaluate_bar: ATR=55 -> SL=40.0, ATR=65 -> SL=48.0
    strat = NQIntradayStrategy(cfg.strategy, cfg.risk, cfg.contract)
    n_bars = 70
    ts_rth = pd.date_range("2026-11-20 09:30:00", periods=n_bars, freq="2min", tz="America/New_York")
    base_df = pd.DataFrame(
        {
            "timestamp": ts_rth,
            "open": [25000.0 + i * 2.0 for i in range(n_bars)],
            "high": [25015.0 + i * 2.0 for i in range(n_bars)],
            "low": [24995.0 + i * 2.0 for i in range(n_bars)],
            "close": [25010.0 + i * 2.0 for i in range(n_bars)],
            "volume": [1500.0] * n_bars,
        }
    )
    enriched = strat.prepare_dataframe(base_df)
    idx = n_bars - 1
    prev_idx = idx - 1
    enriched.at[prev_idx, "high"] = 25140.0
    enriched.at[idx, "open"] = 25130.0
    enriched.at[idx, "low"] = 25128.0
    enriched.at[idx, "close"] = 25150.0
    enriched.at[idx, "high"] = 25152.0
    enriched.at[idx, "ema_fast"] = 25132.0
    enriched.at[idx, "ema_slow"] = 25120.0
    enriched.at[idx, "ema_trend"] = 25080.0
    enriched.at[idx, "vwap"] = 25090.0
    enriched.at[idx, "vwap_z"] = 1.10
    enriched.at[idx, "rsi"] = 58.0
    enriched.at[idx, "vol_ratio"] = 1.25
    enriched.at[idx, "ema_spread_atr"] = 0.25
    enriched.at[idx, "slow_slope_atr"] = 0.20
    enriched.at[idx, "body_ratio"] = 0.80
    enriched.at[idx, "upper_wick_ratio"] = 0.08
    enriched.at[idx, "lower_wick_ratio"] = 0.08
    enriched.at[idx, "pdh"] = 26000.0
    enriched.at[idx, "pdl"] = 24000.0
    enriched.at[idx, "orb_high"] = float("nan")
    enriched.at[idx, "orb_low"] = float("nan")

    enriched.at[idx, "atr"] = 55.0
    sig_normal = strat.evaluate_bar(enriched, idx=idx)
    assert sig_normal is not None
    assert sig_normal.stop_distance_points == 40.0

    enriched.at[idx, "atr"] = 65.0
    sig_high_vol = strat.evaluate_bar(enriched, idx=idx)
    assert sig_high_vol is not None
    assert sig_high_vol.stop_distance_points == 48.0

    # Helper to read logged entry_cancelled events from journal
    def _read_cancel_events(j_dir) -> list[dict]:
        evs = []
        for p in sorted((j_dir / "events").glob("*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("event") == "entry_cancelled":
                        evs.append(rec)
        return evs

    # 3. P1 Marketable StopLimitOrder pricing + 4. Intra-bar Runaway Cancel (>=15 pts)
    j_dir = tmp_path / "j_p1"
    cfg_r = BotConfig()
    cfg_r.risk.fixed_contracts = 1
    cfg_r.risk.enable_partial_tp = False
    cfg_r.risk.max_daily_loss_usd = 2500.0
    runner = IBKRNQBotRunner(cfg_r, mode="paper")
    runner.journal = TradeJournal(base_dir=str(j_dir), timezone="America/New_York")
    runner.contract = Future(conId=563947726, symbol="NQ", localSymbol="NQZ6", multiplier="20", exchange="CME", currency="USD")
    runner._send_discord_alert = lambda *a, **k: None

    req_counter = iter(range(5000, 5200))
    runner.ib.client = SimpleNamespace(getReqId=lambda: next(req_counter), isConnected=lambda: False)
    placed_orders = []
    cancelled_orders = []
    open_trades = []

    def _fake_place_order(c, o):
        placed_orders.append(o)
        tr = SimpleNamespace(contract=c, order=o, orderStatus=SimpleNamespace(status="Submitted"))
        open_trades.append(tr)
        return tr

    runner.ib.placeOrder = _fake_place_order
    runner.ib.cancelOrder = lambda o: cancelled_orders.append(o)
    runner.ib.openTrades = lambda: list(open_trades)
    runner.ib.trades = lambda: list(open_trades)
    runner.ib.positions = lambda: []

    now_utc = pd.Timestamp.now(tz="UTC")
    curr_forming_ts = now_utc.floor("2min")
    sig_closed_ts = curr_forming_ts - pd.Timedelta(minutes=2)
    runner._last_processed_bar_time = sig_closed_ts
    runner.bar_count = 60

    sig_short = TradeSignal(
        timestamp=sig_closed_ts,
        direction="SHORT",
        reason="Bearish Pullback",
        entry_price=25616.25,
        stop_loss=25664.25,
        take_profit=25472.25,
        stop_distance_points=48.0,
        atr=62.5,
        rsi=38.0,
        vwap=25800.0,
        ema_fast=25620.0,
        ema_slow=25645.0,
    )
    runner.place_bracket_order(sig_short, quantity=1)
    p_short = runner._active_parent_order
    assert p_short.lmtPrice == 25614.25  # -2.00 pts marketable SHORT limit
    assert p_short.auxPrice == 25622.25  # +6.00 pts collar stop trigger

    # Check LONG limit/collar symmetry before testing runaway cancel
    placed_orders.clear()
    open_trades.clear()
    sig_long = TradeSignal(
        timestamp=sig_closed_ts,
        direction="LONG",
        reason="Bullish Pullback",
        entry_price=25000.00,
        stop_loss=24960.00,
        take_profit=25120.00,
        stop_distance_points=40.0,
        atr=45.0,
        rsi=58.0,
        vwap=24980.0,
        ema_fast=24998.0,
        ema_slow=24990.0,
    )
    runner.place_bracket_order(sig_long, quantity=1)
    p_long = runner._active_parent_order
    assert p_long.lmtPrice == 25002.00  # +2.00 pts marketable LONG limit
    assert p_long.auxPrice == 24994.00  # -6.00 pts collar stop trigger

    # Re-submit SHORT @ 25616.25 and feed forming bar where price runs -21.25 pts (low=25595.00) without fill
    placed_orders.clear()
    cancelled_orders.clear()
    open_trades.clear()
    runner.place_bracket_order(sig_short, quantity=1)
    placed_orders.clear()

    forming_df = pd.DataFrame(
        {
            "timestamp": [sig_closed_ts, curr_forming_ts],
            "open": [25650.0, 25613.0],
            "high": [25652.0, 25613.5],
            "low": [25614.0, 25595.0],   # 25616.25 - 25595.00 = +21.25 pts favorable move >= 15.0 pts!
            "close": [25616.25, 25598.0],
            "volume": [1200.0, 400.0],
        }
    )
    runner._on_bar_update(forming_df, has_new_bar=True)
    assert len(cancelled_orders) == 3  # parent, TP, SL cancelled via cancelOrder
    assert not any(getattr(o, "orderType", "") == "MKT" for o in placed_orders)  # ZERO MarketOrders!
    assert runner._active_parent_order is None
    cancels = _read_cancel_events(j_dir)
    assert len(cancels) == 1
    assert cancels[-1]["cancel_trigger"] == "runaway_price"

    # 5. P1 60-second Timeout Cancel (without runaway move)
    placed_orders.clear()
    cancelled_orders.clear()
    open_trades.clear()
    runner.place_bracket_order(sig_long, quantity=1)
    placed_orders.clear()
    runner._signal_submitted_utc = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=65)
    forming_flat_df = pd.DataFrame(
        {
            "timestamp": [sig_closed_ts, curr_forming_ts],
            "open": [24995.0, 24999.0],
            "high": [25001.0, 25001.0],   # only +1.0 pt favorable move < 15.0 pts
            "low": [24992.0, 24996.0],
            "close": [25000.0, 24998.0],
            "volume": [1000.0, 250.0],
        }
    )
    runner._on_bar_update(forming_flat_df, has_new_bar=True)
    assert len(cancelled_orders) == 3
    assert not any(getattr(o, "orderType", "") == "MKT" for o in placed_orders)
    assert runner._active_parent_order is None
    cancels = _read_cancel_events(j_dir)
    assert len(cancels) == 2
    assert cancels[-1]["cancel_trigger"] == "timeout_seconds"

    # 6. P1 Safety Guard on Filled Position: NEVER cancel when filled_qty > 0 or live position exists
    placed_orders.clear()
    cancelled_orders.clear()
    open_trades.clear()
    runner.place_bracket_order(sig_short, quantity=1)
    runner._signal_submitted_utc = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=95)
    runner._current_trade_filled_qty = 1
    assert runner._check_pending_entry_guards(forming_df) is False
    assert cancelled_orders == []

    runner._current_trade_filled_qty = 0
    runner._entry_fill_handled = False
    runner.ib.positions = lambda: [SimpleNamespace(contract=runner.contract, position=-1.0)]
    assert runner._check_pending_entry_guards(forming_df) is False
    assert cancelled_orders == []


def test_backtester_realistic_limit_execution_p1() -> None:
    """Verify Backtester(use_limit_execution=True) models 2pt marketable fills, re-anchoring, and runaway cancel."""
    from backtest import Backtester
    from strategy import TradeSignal

    cfg = BotConfig()
    cfg.risk.fixed_contracts = 1
    cfg.risk.enable_partial_tp = False
    cfg.risk.max_daily_loss_usd = 500.0

    # Scenario 1: SHORT signal @ 25616.25 (lmt=25614.25), next bar opens -1.25 pts below at 25615.0 ->
    # fills at 25614.75 (25615.0 - 0.25 slippage), re-anchors SL to 25654.75 (+40) and TP to 25494.75 (-120).
    ts1 = pd.date_range("2026-11-20 12:00:00", periods=3, freq="2min", tz="America/New_York").tz_convert("UTC")
    df_gap = pd.DataFrame(
        {
            "timestamp": ts1,
            "open": [25660.0, 25615.0, 25550.0],
            "high": [25662.0, 25620.0, 25560.0],
            "low": [25614.0, 25548.0, 25490.0],
            "close": [25616.25, 25555.0, 25494.0],
            "volume": [1000, 1200, 1400],
        }
    )
    bt1 = Backtester(cfg, commission_per_side_usd=0.62, slippage_ticks=1, use_limit_execution=True)
    bt1.strategy.prepare_dataframe = lambda d: d
    bt1.strategy.evaluate_bar = lambda d, idx=-1: (
        TradeSignal(
            timestamp=d.iloc[idx]["timestamp"],
            direction="SHORT",
            reason="Bearish Pullback",
            entry_price=25616.25,
            stop_loss=25656.25,
            take_profit=25496.25,
            stop_distance_points=40.0,
            atr=50.0,
            rsi=40.0,
            vwap=25700.0,
            ema_fast=25625.0,
            ema_slow=25645.0,
        )
        if idx == 0
        else None
    )
    res1 = bt1.run(df_gap)
    assert len(res1["trades"]) == 1
    tr1 = res1["trades"][0]
    assert tr1.entry_price == 25614.75
    assert tr1.take_profit == 25494.75  # re-anchored -120.0 pts from exact fill 25614.75
    assert tr1.exit_reason == "Take Profit"
    assert tr1.exit_price == 25494.75

    # Scenario 2: SHORT @ 25616.25 (lmt=25614.25), Bar 1 gaps below limit (open=25613.0) and runs -21.25 pts
    # (low=25595.0) before bouncing back up through 25614.25 -> runaway >= 15pt cancels pending order cleanly!
    df_runaway = pd.DataFrame(
        {
            "timestamp": ts1,
            "open": [25660.0, 25613.0, 25610.0],
            "high": [25662.0, 25620.0, 25670.0],
            "low": [25614.0, 25595.0, 25605.0],
            "close": [25616.25, 25618.0, 25665.0],
            "volume": [1000, 1200, 1400],
        }
    )
    bt2 = Backtester(cfg, commission_per_side_usd=0.62, slippage_ticks=1, use_limit_execution=True)
    bt2.strategy.prepare_dataframe = lambda d: d
    bt2.strategy.evaluate_bar = bt1.strategy.evaluate_bar
    res2 = bt2.run(df_runaway)
    assert len(res2["trades"]) == 0


def test_mr_b_signal_filters_sl_cap_and_causal_daily_ema50() -> None:
    """Verify MR-B ATR% [0.050, 0.150) gate, causal daily EMA50_d regime, SL=min(2*ATR,45), and VWAP RR>=0.80."""
    from indicators import compute_daily_ema_regime

    # 1. Causal daily EMA50_d check (today's close must NEVER affect today's regime)
    dates = pd.bdate_range("2026-06-01", periods=60).date
    closes_up = pd.Series([19000.0 + i * 15.0 for i in range(60)], index=dates)
    reg1 = compute_daily_ema_regime(closes_up, ema_period=50)
    assert bool(reg1.iloc[-1]["daily_bull_50"]) is True
    assert reg1.iloc[-1]["daily_prev_close"] == closes_up.iloc[-2]

    # Crash last day's close far below EMA50 -> today's regime (-1) MUST stay unchanged!
    closes_crash_last = closes_up.copy()
    closes_crash_last.iloc[-1] = 15000.0
    reg2 = compute_daily_ema_regime(closes_crash_last, ema_period=50)
    assert bool(reg2.iloc[-1]["daily_bull_50"]) is True

    # 2. Signal filter checks
    cfg = BotConfig()
    cfg.strategy.enable_mr_b = True
    cfg.strategy.mr_b_max_sl_points = 45.0
    strat = NQIntradayStrategy(cfg.strategy, cfg.risk, cfg.contract)

    ts = pd.date_range("2026-10-20 11:00:00", periods=3, freq="2min", tz="America/New_York").tz_convert("UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [25010.0, 25005.0, 24992.0],
            "high": [25012.0, 25006.0, 25004.0],
            "low": [24998.0, 24990.0, 24991.0],
            "close": [25000.0, 24994.0, 25000.0],  # bar 2 > bar 2 open (24992) and > bar 1 close (24994)
            "volume": [1000.0, 1100.0, 1200.0],
            "vwap": [25050.0, 25050.0, 25050.0],
            "vwap_z": [-1.8, -2.2, -2.1],
            "atr": [30.0, 30.0, 30.0],
            "atr_pct": [0.120, 0.120, 0.120],
            "rsi": [40.0, 38.0, 42.0],
            "ema_fast": [25010.0, 25005.0, 25003.0],
            "ema_slow": [25020.0, 25015.0, 25012.0],
            "daily_bull_50": [True, True, True],
        }
    )

    # SL cap: 2 * 30 = 60 -> capped at 45.0 pts; fixed VWAP TP = 25050.0 (RR = 50/45 = 1.11 >= 0.80)
    sig = strat.evaluate_mr_b_bar(df, idx=-1)
    assert sig is not None
    assert sig.strategy_id == "MR-B"
    assert sig.direction == "LONG"
    assert sig.stop_distance_points == 45.0
    assert sig.stop_loss == 24955.0
    assert sig.take_profit == 25050.0

    # Half-open ATR% bounds [0.050, 0.150)
    df_lo = df.copy()
    df_lo.at[2, "atr_pct"] = 0.049
    assert strat.evaluate_mr_b_bar(df_lo, idx=-1) is None

    df_exact_lo = df.copy()
    df_exact_lo.at[2, "atr_pct"] = 0.050
    df_exact_lo.at[2, "atr"] = 12.5
    sig_lo = strat.evaluate_mr_b_bar(df_exact_lo, idx=-1)
    assert sig_lo is not None and sig_lo.stop_distance_points == 25.0

    df_hi = df.copy()
    df_hi.at[2, "atr_pct"] = 0.150
    assert strat.evaluate_mr_b_bar(df_hi, idx=-1) is None

    # Bull50 False or override False blocks signal
    assert strat.evaluate_mr_b_bar(df, idx=-1, daily_bull_50=False) is None

    # Min RR < 0.80 blocks signal (sl_dist=45, VWAP distance=30 -> RR=0.67 < 0.80)
    df_rr = df.copy()
    df_rr.at[2, "vwap"] = 25030.0
    assert strat.evaluate_mr_b_bar(df_rr, idx=-1) is None

    # SL floor 8.0 pts
    df_floor = df.copy()
    df_floor.at[2, "atr"] = 3.0
    df_floor.at[2, "atr_pct"] = 0.060
    df_floor.at[2, "vwap"] = 25010.0  # RR = 10/8 = 1.25 >= 0.80
    sig_floor = strat.evaluate_mr_b_bar(df_floor, idx=-1)
    assert sig_floor is not None and sig_floor.stop_distance_points == 8.0


def test_mr_b_separate_risk_counters_and_shared_kill_switch(tmp_path) -> None:
    """Verify isolated MR-B daily counters, v4.2 independence, FOMC guards, and shared $1000 kill-switch."""
    from economic_calendar import EconomicCalendar

    cal = EconomicCalendar(cache_path=str(tmp_path / "ff.json"), report_path=None)
    cal.load_events_from_list(
        [
            {"title": "FOMC Statement", "country": "USD", "date": "2027-02-03T14:00:00-05:00", "impact": "High"},
            {"title": "FOMC Meeting Minutes", "country": "USD", "date": "2027-02-24T14:00:00-05:00", "impact": "High"},
        ]
    )
    cfg = BotConfig()
    cfg.strategy.enable_mr_b = True
    cfg.risk.fixed_contracts = 2
    cfg.risk.mr_b_contracts = 2
    cfg.risk.max_daily_loss_usd = 1000.0
    cfg.risk.max_consecutive_losses = 2
    rm = RiskManager(cfg.risk, cfg.contract, calendar=cal)

    # 1. FOMC blackout & flatten checks (static 2026-10-28 + live 2027-02-03, ignoring FOMC Minutes on 2027-02-24)
    assert rm.is_fomc_day(pd.Timestamp("2026-10-28 13:56:00-04:00")) is True
    assert rm.should_flatten_mr_b_fomc(pd.Timestamp("2026-10-28 13:56:00-04:00")) is True
    assert rm.can_open_mr_b_trade(pd.Timestamp("2026-10-28 13:52:00-04:00"), 10)[0] is False
    assert rm.is_fomc_day(pd.Timestamp("2027-02-03 13:56:00-05:00")) is True
    assert rm.is_fomc_day(pd.Timestamp("2027-02-24 13:56:00-05:00")) is False

    # 2. MR-B loss blocks MR-B (1/1) while leaving v4.2 counters completely clean
    ts_ok = pd.Timestamp("2026-10-20 11:45:00-04:00")  # inside EU Close blackout (blocks v4.2), valid for MR-B
    assert rm.can_open_trade(ts_ok, 20)[0] is False
    assert rm.can_open_mr_b_trade(ts_ok, 20)[0] is True

    rm.record_trade_closed(-182.48, exit_bar_index=25, strategy_id="MR-B")
    assert rm.state.mr_b_losses == 1
    assert rm.state.mr_b_trades_taken == 1
    assert rm.state.rth_trades_taken == 0
    assert rm.state.consecutive_losses == 0
    assert rm.state.last_exit_bar_index == -999
    assert rm.state.rth_realized_pnl_usd == -182.48
    assert rm.can_open_mr_b_trade(pd.Timestamp("2026-10-20 12:15:00-04:00"), 40)[0] is False
    assert rm.can_open_trade(pd.Timestamp("2026-10-20 12:00:00-04:00"), 26)[0] is True  # zero cooldown leak

    # 3. Next day: v4.2 hits 2 consecutive losses (-$300) -> halts v4.2 ONLY; MR-B stays active until -$1000 shared limit
    ts_d2 = pd.Timestamp("2026-10-21 10:30:00-04:00")
    assert rm.can_open_mr_b_trade(ts_d2, 50)[0] is True
    rm.record_trade_closed(-150.0, exit_bar_index=55, strategy_id="v4.2")
    rm.record_trade_closed(-150.0, exit_bar_index=65, strategy_id="v4.2")
    assert rm.state.halted is True
    assert rm.state.account_halted is False
    assert rm.can_open_trade(pd.Timestamp("2026-10-21 10:50:00-04:00"), 75)[0] is False
    assert rm.can_open_mr_b_trade(pd.Timestamp("2026-10-21 11:00:00-04:00"), 80)[0] is True

    # 4. Shared account daily kill-switch (-$1000) blocks BOTH strategies
    rm.record_trade_closed(-710.0, exit_bar_index=90, strategy_id="MR-B")
    assert rm.state.account_halted is True
    assert rm.can_open_trade(pd.Timestamp("2026-10-21 12:10:00-04:00"), 100)[0] is False
    assert rm.can_open_mr_b_trade(pd.Timestamp("2026-10-21 12:10:00-04:00"), 100)[0] is False


def test_backtester_dual_strategy_preempt_be_time_exit_and_fomc() -> None:
    """Verify Backtester unified v4.2 + MR-B execution: single 2ct VWAP TP, +0.6R BE, 45b TIME, PREEMPT_V42, FOMC_FLAT."""
    from backtest import Backtester
    from strategy import TradeSignal

    # Scenario A: MR-B 2 MNQ no scale-out, +0.6R -> +1.5b BE, skips 6-bar decay, exits at 45 bars (TIME)
    cfg = BotConfig()
    cfg.strategy.enable_mr_b = True
    cfg.risk.fixed_contracts = 2
    cfg.risk.mr_b_contracts = 2
    cfg.risk.enable_partial_tp = True
    cfg.risk.max_daily_loss_usd = 1000.0

    ts_a = pd.date_range("2026-10-20 10:20:00", periods=48, freq="2min", tz="America/New_York").tz_convert("UTC")
    closes_a = [25000.0] * 48
    closes_a[5] = 25026.0  # >= +0.6R (24 pts) on bar 5 close -> moves SL to 25000.25 + 1.5 = 25001.75
    for k in range(6, 48):
        closes_a[k] = 25015.0  # holds cleanly above BE (25001.75) and below fixed VWAP TP (25050.0)
    df_a = pd.DataFrame(
        {
            "timestamp": ts_a,
            "open": closes_a,
            "high": [c + 3.0 for c in closes_a],
            "low": [c - 3.0 for c in closes_a],
            "close": closes_a,
            "volume": [1000.0] * 48,
        }
    )
    bt_a = Backtester(cfg, commission_per_side_usd=0.62, slippage_ticks=1)
    bt_a.strategy.prepare_dataframe = lambda d: d
    bt_a.strategy.evaluate_bar = lambda d, idx=-1: None
    bt_a.strategy.evaluate_mr_b_bar = lambda d, idx=-1, daily_bull_50=None: (
        TradeSignal(
            timestamp=d.iloc[idx]["timestamp"],
            direction="LONG",
            reason="MR-B Test",
            entry_price=25000.0,
            stop_loss=24960.0,
            take_profit=25050.0,
            stop_distance_points=40.0,
            atr=25.0,
            rsi=40.0,
            vwap=25050.0,
            ema_fast=25010.0,
            ema_slow=25020.0,
            strategy_id="MR-B",
        )
        if idx == 0
        else None
    )
    res_a = bt_a.run(df_a)
    assert len(res_a["trades"]) == 1
    tr_a = res_a["trades"][0]
    assert tr_a.strategy_id == "MR-B"
    assert tr_a.contracts == 2
    assert tr_a.take_profit == 25050.0
    assert tr_a.stop_loss == 25001.75
    assert tr_a.exit_reason == "TIME"
    assert tr_a.exit_time == ts_a[45]

    # Scenario B: PREEMPT_V42 + 0 overlaps + v4.2 flat-bar priority
    # Bar 4 drops to hit both TP1 (+2R=24939.75) and Runner TP2 (+5R=24849.75); bar 10 test runs to EOD
    ts_b = list(pd.date_range("2026-10-20 10:20:00", periods=14, freq="2min", tz="America/New_York").tz_convert("UTC")) + [
        pd.Timestamp("2026-10-20 15:56:00", tz="America/New_York").tz_convert("UTC")
    ]
    df_b = pd.DataFrame(
        {
            "timestamp": ts_b,
            "open": [25000.0] * 15,
            "high": [25005.0] * 15,
            "low": [24995.0] * 15,
            "close": [25000.0] * 15,
            "volume": [1000.0] * 15,
        }
    )
    df_b.at[4, "low"] = 24845.0
    df_b.at[4, "close"] = 24850.0

    bt_b = Backtester(cfg, commission_per_side_usd=0.62, slippage_ticks=1)
    bt_b.strategy.prepare_dataframe = lambda d: d
    bt_b.strategy.evaluate_bar = lambda d, idx=-1: (
        TradeSignal(
            timestamp=d.iloc[idx]["timestamp"],
            direction="SHORT",
            reason="v4.2 Test",
            entry_price=25000.0,
            stop_loss=25030.0,
            take_profit=24990.0,
            stop_distance_points=30.0,
            atr=25.0,
            rsi=42.0,
            vwap=25010.0,
            ema_fast=24995.0,
            ema_slow=25005.0,
            strategy_id="v4.2",
        )
        if idx in (2, 10)
        else None
    )
    bt_b.strategy.evaluate_mr_b_bar = lambda d, idx=-1, daily_bull_50=None: (
        TradeSignal(
            timestamp=d.iloc[idx]["timestamp"],
            direction="LONG",
            reason="MR-B Test",
            entry_price=25000.0,
            stop_loss=24960.0,
            take_profit=25050.0,
            stop_distance_points=40.0,
            atr=25.0,
            rsi=40.0,
            vwap=25050.0,
            ema_fast=25010.0,
            ema_slow=25020.0,
            strategy_id="MR-B",
        )
        if idx in (0, 10)
        else None
    )
    res_b = bt_b.run(df_b)
    trades_b = res_b["trades"]
    assert trades_b[0].strategy_id == "MR-B" and trades_b[0].exit_reason == "PREEMPT_V42"
    assert trades_b[1].strategy_id == "v4.2"
    assert trades_b[0].exit_time <= trades_b[1].entry_time  # strict 0-overlap guarantee
    # Flat bar 10 where both v4.2 and MR-B signal: v4.2 takes strict priority
    assert trades_b[-1].strategy_id == "v4.2" and trades_b[-1].entry_time == ts_b[10]

    # Scenario C: Pre-FOMC 13:56 ET Flatten
    ts_fomc = pd.date_range("2026-10-28 13:44:00", periods=10, freq="2min", tz="America/New_York").tz_convert("UTC")
    df_c = pd.DataFrame(
        {
            "timestamp": ts_fomc,
            "open": [25000.0] * 10,
            "high": [25005.0] * 10,
            "low": [24995.0] * 10,
            "close": [25000.0] * 10,
            "volume": [1000.0] * 10,
        }
    )
    bt_c = Backtester(cfg, commission_per_side_usd=0.62, slippage_ticks=1)
    bt_c.strategy.prepare_dataframe = lambda d: d
    bt_c.strategy.evaluate_bar = lambda d, idx=-1: None
    bt_c.strategy.evaluate_mr_b_bar = lambda d, idx=-1, daily_bull_50=None: (
        TradeSignal(
            timestamp=d.iloc[idx]["timestamp"],
            direction="LONG",
            reason="MR-B FOMC Test",
            entry_price=25000.0,
            stop_loss=24960.0,
            take_profit=25050.0,
            stop_distance_points=40.0,
            atr=25.0,
            rsi=40.0,
            vwap=25050.0,
            ema_fast=25010.0,
            ema_slow=25020.0,
            strategy_id="MR-B",
        )
        if idx == 0
        else None
    )
    res_c = bt_c.run(df_c)
    assert len(res_c["trades"]) == 1
    assert res_c["trades"][0].exit_reason == "FOMC_FLAT"
    assert res_c["metrics"]["by_strategy"]["MR-B"]["trades"] == 1


def test_runner_mr_b_bracket_no_scaleout_reanchor_preempt_and_recovery(tmp_path) -> None:
    """Verify live runner MR-B single-bracket, fixed VWAP TP re-anchor, PREEMPT_V42 flatten/abort, and recovery."""
    from types import SimpleNamespace
    from ib_async import Future, LimitOrder, StopOrder
    from ibkr_execution import IBKRNQBotRunner
    from strategy import TradeSignal
    from trade_journal import TradeJournal

    cfg = BotConfig()
    cfg.strategy.enable_mr_b = True
    cfg.risk.fixed_contracts = 2
    cfg.risk.mr_b_contracts = 2
    cfg.risk.enable_partial_tp = True
    cfg.risk.max_daily_loss_usd = 1000.0

    runner = IBKRNQBotRunner(cfg, mode="paper")
    runner.journal = TradeJournal(base_dir=str(tmp_path / "j_mrb"), timezone="America/New_York")
    runner.contract = Future(conId=815824267, symbol="MNQ", localSymbol="MNQZ6", multiplier="2", exchange="CME", currency="USD")
    runner._send_discord_alert = lambda *a, **k: None
    runner._generate_trade_chart = lambda *a, **k: None
    runner._daily_bull_50 = True
    runner._daily_regime_date = pd.Timestamp("2026-10-20").date()

    req_counter = iter(range(7000, 7200))
    placed_orders = []
    open_trades = []
    positions = []

    def _fake_place(c, o):
        placed_orders.append(o)
        if getattr(o, "orderType", "") == "MKT" and positions:
            positions.clear()
            open_trades.clear()
        else:
            open_trades.append(SimpleNamespace(contract=c, order=o, orderStatus=SimpleNamespace(status="Submitted")))
        return SimpleNamespace(contract=c, order=o)

    def _fake_cancel(o):
        open_trades[:] = [t for t in open_trades if getattr(t.order, "orderId", None) != getattr(o, "orderId", None)]

    runner.ib.client = SimpleNamespace(getReqId=lambda: next(req_counter), isConnected=lambda: False)
    runner.ib.placeOrder = _fake_place
    runner.ib.cancelOrder = _fake_cancel
    runner.ib.openTrades = lambda: list(open_trades)
    runner.ib.trades = lambda: list(open_trades)
    runner.ib.positions = lambda: list(positions)

    # 1. Single 3-order bracket for 2 MNQ MR-B + orderRef tag
    sig_mrb = TradeSignal(
        timestamp=pd.Timestamp("2026-10-20 10:20:00-04:00"),
        direction="LONG",
        reason="MR-B Dip",
        entry_price=25000.0,
        stop_loss=24955.0,
        take_profit=25050.0,
        stop_distance_points=45.0,
        atr=30.0,
        rsi=40.0,
        vwap=25050.0,
        ema_fast=25010.0,
        ema_slow=25020.0,
        strategy_id="MR-B",
    )
    runner.place_bracket_order(sig_mrb, quantity=2)
    assert len(placed_orders) == 3  # NO partial scale-out split!
    p_mrb, tp_mrb, sl_mrb = placed_orders
    assert p_mrb.totalQuantity == 2
    assert p_mrb.orderRef.startswith("MR-B:")
    assert tp_mrb.lmtPrice == 25050.0
    assert runner._active_tp1_order is None

    # 2. Entry fill @ 25001.00 re-anchors SL to 24956.00 while preserving fixed VWAP TP = 25050.00!
    positions.append(SimpleNamespace(contract=runner.contract, position=2.0, avgCost=25001.0 * 2.0))
    fill_t = pd.Timestamp("2026-10-20 14:20:05", tz="UTC")
    runner._on_exec_details(
        SimpleNamespace(),
        SimpleNamespace(contract=runner.contract, execution=SimpleNamespace(orderId=p_mrb.orderId, side="BOT", shares=2, price=25001.0, time=fill_t)),
    )
    assert sl_mrb.auxPrice == 24956.0
    assert tp_mrb.lmtPrice == 25050.0  # Fixed VWAP TP MUST stay intact!
    assert sig_mrb.take_profit == 25050.0

    # 3. PREEMPT_V42 happy path + delayed MR-B commissionReport must not corrupt active v4.2 trade
    placed_orders.clear()
    v42_sig = TradeSignal(
        timestamp=pd.Timestamp("2026-10-20 10:24:00-04:00"),
        direction="SHORT",
        reason="Bearish Pullback",
        entry_price=24990.0,
        stop_loss=25020.0,
        take_profit=24900.0,
        stop_distance_points=30.0,
        atr=28.0,
        rsi=42.0,
        vwap=25010.0,
        ema_fast=24995.0,
        ema_slow=25005.0,
        strategy_id="v4.2",
    )
    runner.strategy.prepare_dataframe = lambda d: pd.DataFrame(
        {
            "timestamp": d["timestamp"],
            "open": d["open"],
            "high": d["high"],
            "low": d["low"],
            "close": d["close"],
            "volume": d["volume"],
            "vwap": 25010.0,
            "ema_fast": 24995.0,
            "ema_slow": 25005.0,
            "vwap_z": -0.5,
            "hurst": 0.55,
            "kalman_vel_atr": -0.2,
        }
    )
    runner.strategy.evaluate_bar = lambda d, idx=-1: v42_sig
    bar_df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-10-20 14:22:00", tz="UTC"),
                pd.Timestamp("2026-10-20 14:24:00", tz="UTC"),
            ],
            "open": [25001.0, 24998.0],
            "high": [25003.0, 24999.0],
            "low": [24995.0, 24989.0],
            "close": [24998.0, 24990.0],
            "volume": [1000.0, 1200.0],
        }
    )
    runner._on_bar_update(bar_df, has_new_bar=True)

    # First placed order is MarketOrder flatten for MR-B, followed by 6 orders for 2-ct v4.2 bracket!
    assert placed_orders[0].orderType == "MKT" and placed_orders[0].orderRef == "MR-B:PREEMPT_V42"
    mkt_close_oid = placed_orders[0].orderId
    assert len(placed_orders) == 7
    assert runner._active_strategy_id == "v4.2"
    assert runner._active_signal is v42_sig

    # Now delayed commissionReport arrives for the preempted MR-B MKT order -> goes ONLY to MR-B counters!
    runner._on_commission_report(
        SimpleNamespace(order=placed_orders[0]),
        SimpleNamespace(execution=SimpleNamespace(orderId=mkt_close_oid, shares=2, price=24990.0, time=pd.Timestamp("2026-10-20 14:24:01", tz="UTC"))),
        SimpleNamespace(realizedPNL=-46.48, commission=2.48),
    )
    assert runner.risk.state.mr_b_trades_taken == 1
    assert runner.risk.state.mr_b_losses == 1
    assert runner.risk.state.rth_trades_taken == 0
    assert abs(runner.risk.state.mr_b_realized_pnl_usd - (-46.48)) < 1e-6
    assert runner._active_signal is v42_sig  # Live v4.2 state untouched!

    # 4. PREEMPT_V42 fail-safe abort when IBKR refuses to confirm flat
    runner_abort = IBKRNQBotRunner(cfg, mode="paper")
    runner_abort.contract = runner.contract
    runner_abort._send_discord_alert = lambda *a, **k: None
    runner_abort._active_signal = sig_mrb
    runner_abort._active_strategy_id = "MR-B"
    runner_abort._active_parent_order = p_mrb
    sticky_positions = [SimpleNamespace(contract=runner.contract, position=2.0)]
    abort_placed = []
    runner_abort.ib.positions = lambda: list(sticky_positions)
    runner_abort.ib.openTrades = lambda: []
    runner_abort.ib.client = SimpleNamespace(getReqId=lambda: 8801, isConnected=lambda: False)
    runner_abort.ib.placeOrder = lambda c, o: abort_placed.append(o) or SimpleNamespace(order=o)
    ok_preempt = runner_abort._preempt_mr_b_for_v42(v42_sig, pd.Timestamp("2026-10-20 14:26:00", tz="UTC"), 24990.0, has_pos=True)
    assert ok_preempt is False
    assert len(abort_placed) == 1  # Flatten MarketOrder attempted, ZERO v4.2 brackets submitted!

    # 5. Restart recovery (_recover_active_trade & _restore_daily_risk_state)
    runner_rec = IBKRNQBotRunner(cfg, mode="paper")
    runner_rec.contract = runner.contract
    runner_rec.ib.positions = lambda: [SimpleNamespace(contract=runner.contract, position=2.0, avgCost=25000.0 * 2.0)]
    stp_rec = StopOrder(action="SELL", totalQuantity=2, stopPrice=24955.0)
    stp_rec.orderRef = "MR-B:9001"
    tp_rec = LimitOrder(action="BUY", totalQuantity=2, lmtPrice=25050.0)
    tp_rec.parentId = 9001
    tp_rec.orderRef = "MR-B:9001"
    runner_rec.ib.reqAllOpenOrders = lambda: [
        SimpleNamespace(contract=runner.contract, order=stp_rec),
        SimpleNamespace(contract=runner.contract, order=tp_rec),
    ]
    runner_rec._recover_active_trade()
    assert runner_rec._active_strategy_id == "MR-B"
    assert runner_rec._active_signal.strategy_id == "MR-B"
    assert runner_rec._tp1_filled is False

    now_rth_utc = pd.Timestamp.now(tz="America/New_York").replace(hour=11, minute=0, second=0, microsecond=0).tz_convert("UTC")
    j_rec = TradeJournal(base_dir=str(tmp_path / "j_rec_dual"), timezone="America/New_York")
    j_rec.log("trade_closed", bar_ts=now_rth_utc, strategy_id="MR-B", direction="LONG", net_pnl_usd=-120.0)
    j_rec.log("trade_closed", bar_ts=now_rth_utc, strategy_id="v4.2", direction="SHORT", net_pnl_usd=-90.0)
    runner_state = IBKRNQBotRunner(cfg, mode="paper")
    runner_state.journal = j_rec
    runner_state._restore_daily_risk_state()
    assert runner_state.risk.state.mr_b_trades_taken == 1
    assert runner_state.risk.state.mr_b_losses == 1
    assert runner_state.risk.state.trades_taken == 1
    assert runner_state.risk.state.consecutive_losses == 1
    assert abs(runner_state.risk.state.rth_realized_pnl_usd - (-210.0)) < 1e-6

    # 6. Daily EMA50_d TradingView fallback + fail-closed <50 days check
    hist_dates = pd.bdate_range(end="2026-10-19", periods=60)
    tv_daily_ok = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(d, tz="America/New_York").tz_convert("UTC") for d in hist_dates]
            + [pd.Timestamp("2026-10-20 09:30:00", tz="America/New_York").tz_convert("UTC")],
            "open": [20000.0 + i * 10.0 for i in range(61)],
            "high": [20010.0 + i * 10.0 for i in range(61)],
            "low": [19990.0 + i * 10.0 for i in range(61)],
            "close": [20000.0 + i * 10.0 for i in range(60)] + [15000.0],  # today's uncompleted bar ignored
            "volume": [1000.0] * 61,
        }
    )
    runner_tv = IBKRNQBotRunner(cfg, mode="paper")
    runner_tv.tv_feed = SimpleNamespace(fetch_daily_bars=lambda base_symbol="MNQ", n_bars=150: tv_daily_ok)
    ok_reg = runner_tv._refresh_daily_regime(pd.Timestamp("2026-10-20 10:00:00-04:00"))
    assert ok_reg is True
    assert runner_tv._daily_bull_50 is True

    runner_short = IBKRNQBotRunner(cfg, mode="paper")
    runner_short.tv_feed = SimpleNamespace(fetch_daily_bars=lambda base_symbol="MNQ", n_bars=150: tv_daily_ok.iloc[:10])
    ok_short = runner_short._refresh_daily_regime(pd.Timestamp("2026-10-20 10:00:00-04:00"))
    assert ok_short is False
    assert runner_short._daily_bull_50 is False
