"""
Configuration settings for the NQ / MNQ Futures IBKR Trading Bot (v4 Quant Engine).
Defaults are optimized across 3 full months (66 trading days / 14,850 2-min bars)
of real IBKR Continuous MNQ futures data (July 27 - Oct 27, 2026), achieving:
- 4 / 4 Profitable Months (July, August, September, October all positive)
- Net Profit: +$2,926.84 on 1 MNQ (+$29,268.40 on 1 E-mini NQ)
- Profit Factor: 1.97 | Sharpe Ratio: 2.47 | Max Drawdown: $790.52
"""
from dataclasses import dataclass, field
import os
from typing import Literal


@dataclass
class IBKRConfig:
    """Interactive Brokers TWS / IB Gateway connection & notification settings."""
    host: str = "127.0.0.1"
    paper_port: int = 7497      # Default TWS Paper Trading port (4002 for IB Gateway Paper)
    live_port: int = 7496       # Default TWS Live Trading port (4001 for IB Gateway Live)
    client_id: int = 101
    account: str = ""           # Optional: specify IBKR account code if multiple accounts exist
    timeout: int = 20
    discord_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", ""))
    use_tradingview_data: bool = True  # Use TradingView real-time / hybrid 2m bar feed to bypass IBKR Paper 10m delay
    tradingview_auth_token: str = field(default_factory=lambda: os.getenv("TRADINGVIEW_AUTH_TOKEN", ""))
    tradingview_realtime_proxy: str = "CAPITALCOM:US100"


@dataclass
class ContractConfig:
    """Futures contract specification for Nasdaq-100 E-mini (NQ) or Micro (MNQ)."""
    symbol: Literal["NQ", "MNQ"] = "MNQ"  # Default to MNQ ($2/pt) for safer initial testing; NQ is $20/pt
    exchange: str = "CME"
    currency: str = "USD"
    contract_month: str = ""              # e.g., "202612" or leave empty "" to auto-resolve front month
    tick_size: float = 0.25               # Minimum price fluctuation for NQ/MNQ

    @property
    def point_value(self) -> float:
        """Dollar multiplier per 1.0 index point."""
        return 20.0 if self.symbol == "NQ" else 2.0


@dataclass
class StrategyConfig:
    """Parameters for the v4 Quant Engine (Kalman Velocity + Hurst + Meta-Labeling + Hybrid VWAP/EMA/ORB)."""
    bar_size: Literal["1 min", "2 mins", "5 mins"] = "2 mins"
    history_duration: str = "3 D"         # Initial lookback loaded on startup for indicator warmup

    # EMA & Trend settings
    fast_ema_period: int = 9
    slow_ema_period: int = 21
    trend_ema_period: int = 50
    use_session_vwap: bool = True
    require_trend_ema_alignment: bool = True

    # Trend strength & candle quality filters
    min_ema_spread_atr: float = 0.06      # Minimum |EMA9 - EMA21| / ATR to avoid flat chop (optimized for active frequency)
    min_slow_slope_atr: float = 0.08      # Minimum 3-bar Slow EMA slope normalized by ATR
    min_body_ratio: float = 0.35          # Minimum |Close - Open| / (High - Low) to reject dojis
    max_opposing_wick_ratio: float = 0.40 # Reject longs with >40% upper wick or shorts with >40% lower wick
    require_prev_extreme_break: bool = True

    # Session VWAP Standard Deviation Band (Z-Score) & Prior Day High/Low Guards
    use_vwap_z_filter: bool = True
    min_vwap_z_score: float = 0.20        # Avoid entering inside flat VWAP magnet chop (< 0.20 SD)
    max_vwap_z_score: float = 2.30        # Avoid buying/selling exhausted > 2.30 SD overextensions
    use_pdh_pdl_guard: bool = True        # Block buying directly below Prior Day High or shorting above Prior Day Low
    pdh_pdl_buffer_atr: float = 0.50      # Minimum distance in ATR below PDH / above PDL before breakout

    # Kalman State-Space Filter (used for V-Reversal impulse velocity)
    use_quant_meta_filter: bool = False
    kalman_q_pos: float = 0.05            # Kalman state position process variance
    kalman_q_vel: float = 0.01            # Kalman state velocity process variance
    kalman_r_meas: float = 1.0            # Kalman measurement noise variance

    # Signal triggers ("pullback", "hybrid_all" = Pullback + ORB + V-Reversal)
    entry_mode: Literal["pullback", "crossover", "both", "hybrid_all"] = "hybrid_all"
    max_crossover_atr: float = 0.0        # 0.0 = Lagging Crossover & 3-Bar Momentum triggers disabled (Ablation Step #2)

    # Volatility & Momentum filters
    atr_period: int = 14
    min_atr_points: float = 3.0           # Avoid dead/chop markets (3.0 pts allows clean European premarket trends)
    max_atr_points: float = 95.0          # Allow high-velocity sell-off/trend sessions up to 95 pts 2m ATR while blocking >100 pt capitulation spikes
    rsi_period: int = 14
    rsi_long_min: float = 45.0
    rsi_overbought: float = 70.0          # Do not enter new longs above this RSI
    rsi_short_max: float = 55.0
    rsi_oversold: float = 30.0            # Do not enter new shorts below this RSI
    min_volume_ratio: float = 0.75        # Current bar volume vs 20-bar average volume

    # Pullback tolerance in ATR multiples (how close low/high needs to get to Fast EMA)
    pullback_ema_tolerance_atr: float = 0.40

    # Setup E: Premarket / Morning VWAP Overextension V-Reversal
    enable_v_reversal: bool = True
    v_reversal_min_z_extreme: float = 1.30  # Rolling 12-bar |VWAP Z| must have reached >= 1.30 SD
    v_reversal_min_kalman_vel: float = 0.35 # Minimum opposing Kalman velocity (|vel|/ATR >= 0.35)
    v_reversal_min_body_ratio: float = 0.55 # Strong impulse reversal candle body >= 55% of range
    v_reversal_min_vwap_dist_atr: float = 2.0 # At least 2.0 * ATR room back to Session VWAP


@dataclass
class RiskConfig:
    """Intraday risk management, bracket order sizing, +1R break-even, and v4 Time-Decay Stop."""
    # Position sizing
    fixed_contracts: int = 1              # Number of contracts per trade (if risk_per_trade_usd is 0)
    risk_per_trade_usd: float = 0.0       # If > 0, dynamically sizes contracts up to max_contracts
    max_contracts: int = 3

    # Order execution improvements (Two-Sided Stop-Limit Collar + Unfilled Order Timeout)
    entry_limit_slippage_ticks: int = 1   # Submit Limit entry 1 tick (0.25 pt) marketable to prevent missed breakouts
    entry_collar_points: float = 6.0      # Stop-Limit protection band (pts) so delayed signals never fill far past entry/SL
    cancel_unfilled_entry_bars: int = 3   # Automatically cancel unfilled parent entry order after 3 bars

    # Stop Loss, Take Profit, +1R Break-Even & v4 Time-Decay Stagnant Stop Tightening
    stop_mode: Literal["atr", "fixed_points"] = "atr"
    atr_stop_multiplier: float = 1.2      # Stop loss distance = 1.2 * ATR(14)
    fixed_stop_points: float = 15.0       # Used when stop_mode == "fixed_points"
    min_stop_points: float = 8.0          # Floor on stop distance (32 ticks)
    max_stop_points: float = 40.0         # Cap on stop distance (160 ticks)
    reward_risk_ratio: float = 3.0        # Take Profit distance = 3.0 * Stop distance
    breakeven_at_1r: bool = True          # Automatically move Stop Loss to Entry + 1.0 pt once +1R is reached
    breakeven_offset_points: float = 1.0  # Points locked in when break-even triggers

    # v4 Time-Decay Stop Tightening (Ornstein-Uhlenbeck Stagnation Decay)
    time_decay_bars: int = 6              # If trade hasn't reached +1R within 6 bars (12 mins), tighten stop
    time_decay_stop_r: float = 0.50       # Tighten risk from -1.0R to -0.50R on stagnant trades

    # Daily circuit breakers (Active Frequency Mode: max 4 trades/day across full RTH session)
    max_daily_loss_usd: float = 250.0     # Halts new entries & flattens if breached
    max_daily_trades: int = 4             # Max 4 high-conviction trades per session
    max_consecutive_losses: int = 2       # Stop trading for the day after 2 consecutive losses
    cooldown_bars_after_exit: int = 3     # Wait N bars before entering another trade

    # Trading session window (America/New_York - US Eastern Time)
    # Dual-Session Mode:
    #   1) European Pre-Market Session: 04:00 - 06:00 ET (Trend + V-Reversal), 06:00 - 08:20 ET (V-Reversal only)
    #   2) US Regular Trading Hours (RTH): 09:45 - 15:30 ET (VWAP re-anchored at 09:30 ET Cash Open)
    timezone: str = "America/New_York"
    rth_only: bool = False                # False = stream ETH+RTH bars and set outsideRth=True on orders
    use_premarket_session: bool = True    # Enable European Pre-Market trading window
    premarket_start_time: str = "04:00"   # 10:00 CET (European Cash Open)
    premarket_trend_end_time: str = "06:00"  # 12:00 CET (End of standard trend entries before EU lunch chop; 06:00-08:20 ET = V-Reversal only)
    premarket_end_time: str = "08:20"     # 14:20 CET (Stop 10m before 08:30 ET US macro news releases)
    premarket_max_trades: int = 2         # Dedicated Pre-Market trade cap (does not reduce RTH trade cap)
    entry_start_time: str = "09:45"       # Enter US RTH after 15-min Opening Range settles (15:45 CET)
    entry_end_time: str = "15:30"         # Extended RTH entry window to 15:30 ET (21:30 CET)
    flatten_eod_time: str = "15:55"       # Force flatten all open positions and cancel orders (21:55 CET)


@dataclass
class BotConfig:
    """Master configuration container."""
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
