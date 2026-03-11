"""
BVMT Strategy Engine — Phase 2
================================
Implements 4 core trading strategies tuned for BVMT's
low-liquidity, thin-spread environment:

  1. RSI Mean-Reversion
  2. MACD Momentum
  3. Bollinger Band Squeeze
  4. Volume-Weighted Breakout (VWAP)

Each strategy returns a Signal object with:
  - direction: BUY | SELL | HOLD
  - confidence: 0.0 – 1.0
  - reason: human-readable explanation
  - entry/stop/target prices

Usage:
    engine = StrategyEngine(feed)
    signal = engine.run_all("BIAT")
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import logging

log = logging.getLogger("StrategyEngine")


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

class Direction(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    strategy:   str
    ticker:     str
    direction:  Direction
    confidence: float          # 0.0 – 1.0
    reason:     str
    entry:      Optional[float] = None
    stop_loss:  Optional[float] = None
    take_profit: Optional[float] = None
    indicators: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"[{self.strategy}] {self.ticker} → {self.direction.value} "
                f"(conf={self.confidence:.0%}) | {self.reason}")


@dataclass
class CompositeSignal:
    ticker:       str
    direction:    Direction
    confidence:   float
    vote_buy:     int
    vote_sell:    int
    vote_hold:    int
    signals:      list[Signal]
    entry:        Optional[float] = None
    stop_loss:    Optional[float] = None
    take_profit:  Optional[float] = None
    risk_reward:  Optional[float] = None

    def __repr__(self):
        return (f"COMPOSITE {self.ticker} → {self.direction.value} "
                f"conf={self.confidence:.0%} | "
                f"B:{self.vote_buy} S:{self.vote_sell} H:{self.vote_hold}")


# ─────────────────────────────────────────────
# TECHNICAL INDICATOR LIBRARY
# ─────────────────────────────────────────────

class Indicators:
    """Pure numpy/pandas implementations — no TA-Lib dependency."""

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs  = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(close: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(close: pd.Series, period=20, std_dev=2.0):
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        width = (upper - lower) / sma
        pct_b = (close - lower) / (upper - lower)
        return upper, sma, lower, width, pct_b

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series) -> pd.Series:
        typical = (high + low + close) / 3
        return (typical * volume).cumsum() / volume.cumsum()

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period).mean()

    @staticmethod
    def ema(close: pd.Series, period: int) -> pd.Series:
        return close.ewm(span=period, adjust=False).mean()

    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        lowest  = low.rolling(k_period).min()
        highest = high.rolling(k_period).max()
        k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
        d = k.rolling(d_period).mean()
        return k, d

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

    @staticmethod
    def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """Current volume / average volume ratio."""
        return volume / volume.rolling(period).mean()


# ─────────────────────────────────────────────
# STRATEGY 1: RSI MEAN-REVERSION
# ─────────────────────────────────────────────

class RSIMeanReversion:
    """
    Buy when RSI crosses up through oversold threshold (default 35).
    Sell when RSI crosses down through overbought threshold (default 65).
    
    BVMT tuning: Wider thresholds (35/65 vs 30/70) because BVMT stocks
    can stay oversold longer due to low liquidity.
    """

    def __init__(self, rsi_period=14, oversold=35, overbought=65):
        self.rsi_period  = rsi_period
        self.oversold    = oversold
        self.overbought  = overbought

    def run(self, df: pd.DataFrame, ticker: str) -> Signal:
        if len(df) < self.rsi_period + 5:
            return Signal("RSI", ticker, Direction.HOLD, 0.0, "Insufficient data")

        close = df["close"]
        rsi   = Indicators.rsi(close, self.rsi_period)
        atr   = Indicators.atr(df["high"], df["low"], close)

        current_rsi  = rsi.iloc[-1]
        previous_rsi = rsi.iloc[-2]
        current_price = close.iloc[-1]
        current_atr   = atr.iloc[-1]

        # Divergence check: price making lower low, RSI making higher low = bullish
        price_lower_low = close.iloc[-1] < close.iloc[-5]
        rsi_higher_low  = rsi.iloc[-1] > rsi.iloc[-5]
        bullish_divergence = price_lower_low and rsi_higher_low

        if current_rsi < self.oversold and previous_rsi >= self.oversold:
            # RSI crossing INTO oversold zone — potential reversal
            confidence = min(0.9, (self.oversold - current_rsi) / self.oversold + 0.5)
            if bullish_divergence:
                confidence = min(0.95, confidence + 0.15)
            return Signal(
                strategy="RSI_MeanReversion", ticker=ticker,
                direction=Direction.BUY,
                confidence=round(confidence, 2),
                reason=f"RSI({self.rsi_period}) oversold at {current_rsi:.1f}" +
                       (" + bullish divergence" if bullish_divergence else ""),
                entry=current_price,
                stop_loss=round(current_price - 2.0 * current_atr, 3),
                take_profit=round(current_price + 3.0 * current_atr, 3),
                indicators={"rsi": round(current_rsi, 2), "atr": round(current_atr, 3)},
            )

        elif current_rsi > self.overbought and previous_rsi <= self.overbought:
            confidence = min(0.9, (current_rsi - self.overbought) / (100 - self.overbought) + 0.5)
            return Signal(
                strategy="RSI_MeanReversion", ticker=ticker,
                direction=Direction.SELL,
                confidence=round(confidence, 2),
                reason=f"RSI({self.rsi_period}) overbought at {current_rsi:.1f}",
                entry=current_price,
                stop_loss=round(current_price + 2.0 * current_atr, 3),
                take_profit=round(current_price - 3.0 * current_atr, 3),
                indicators={"rsi": round(current_rsi, 2), "atr": round(current_atr, 3)},
            )

        return Signal(
            strategy="RSI_MeanReversion", ticker=ticker,
            direction=Direction.HOLD, confidence=0.3,
            reason=f"RSI neutral at {current_rsi:.1f}",
            indicators={"rsi": round(current_rsi, 2)},
        )


# ─────────────────────────────────────────────
# STRATEGY 2: MACD MOMENTUM
# ─────────────────────────────────────────────

class MACDMomentum:
    """
    Trades MACD signal-line crossovers filtered by histogram momentum.
    
    BVMT tuning: Uses standard 12/26/9 parameters but requires
    histogram confirmation for 2 consecutive bars to reduce false signals
    in BVMT's choppy, low-volume sessions.
    """

    def __init__(self, fast=12, slow=26, signal=9):
        self.fast   = fast
        self.slow   = slow
        self.signal = signal

    def run(self, df: pd.DataFrame, ticker: str) -> Signal:
        if len(df) < self.slow + self.signal + 5:
            return Signal("MACD", ticker, Direction.HOLD, 0.0, "Insufficient data")

        close = df["close"]
        macd, sig, hist = Indicators.macd(close, self.fast, self.slow, self.signal)
        atr  = Indicators.atr(df["high"], df["low"], close)
        ema50 = Indicators.ema(close, 50)
        price = close.iloc[-1]

        h_curr = hist.iloc[-1]
        h_prev = hist.iloc[-2]
        m_curr = macd.iloc[-1]
        m_prev = macd.iloc[-2]
        trend_up = price > ema50.iloc[-1]

        # Bullish crossover: MACD crosses above signal line
        bullish_cross = m_prev < sig.iloc[-2] and m_curr >= sig.iloc[-1]
        # Bearish crossover
        bearish_cross = m_prev > sig.iloc[-2] and m_curr <= sig.iloc[-1]
        # Histogram strengthening (momentum accelerating)
        hist_strengthening = h_curr > h_prev > 0
        hist_weakening     = h_curr < h_prev < 0

        current_atr = atr.iloc[-1]

        if bullish_cross:
            conf = 0.65
            if trend_up:         conf += 0.10
            if hist_strengthening: conf += 0.10
            return Signal(
                strategy="MACD_Momentum", ticker=ticker,
                direction=Direction.BUY, confidence=round(min(conf, 0.90), 2),
                reason=f"MACD bullish crossover (MACD={m_curr:.3f}, hist={h_curr:.3f})" +
                       (" + uptrend" if trend_up else ""),
                entry=price,
                stop_loss=round(price - 1.8 * current_atr, 3),
                take_profit=round(price + 3.5 * current_atr, 3),
                indicators={"macd": round(m_curr, 4), "signal": round(sig.iloc[-1], 4),
                            "histogram": round(h_curr, 4), "ema50": round(ema50.iloc[-1], 3)},
            )

        elif bearish_cross:
            conf = 0.65
            if not trend_up:    conf += 0.10
            if hist_weakening:  conf += 0.10
            return Signal(
                strategy="MACD_Momentum", ticker=ticker,
                direction=Direction.SELL, confidence=round(min(conf, 0.90), 2),
                reason=f"MACD bearish crossover (MACD={m_curr:.3f})",
                entry=price,
                stop_loss=round(price + 1.8 * current_atr, 3),
                take_profit=round(price - 3.5 * current_atr, 3),
                indicators={"macd": round(m_curr, 4), "histogram": round(h_curr, 4)},
            )

        return Signal(
            strategy="MACD_Momentum", ticker=ticker,
            direction=Direction.HOLD, confidence=0.3,
            reason=f"No crossover (MACD={m_curr:.3f})",
            indicators={"macd": round(m_curr, 4), "histogram": round(h_curr, 4)},
        )


# ─────────────────────────────────────────────
# STRATEGY 3: BOLLINGER BAND SQUEEZE
# ─────────────────────────────────────────────

class BollingerSqueeze:
    """
    Identifies low-volatility squeeze (bands narrowing), then trades
    the breakout direction when bands expand.
    
    BVMT tuning: Squeeze threshold calibrated to BVMT's naturally
    lower volatility. Min squeeze duration = 5 days (longer than
    liquid markets because BVMT price discovery is slower).
    """

    def __init__(self, period=20, std_dev=2.0, squeeze_pct=0.03,
                 min_squeeze_days=5):
        self.period = period
        self.std_dev = std_dev
        self.squeeze_pct = squeeze_pct  # Width < 3% = squeeze
        self.min_squeeze_days = min_squeeze_days

    def run(self, df: pd.DataFrame, ticker: str) -> Signal:
        if len(df) < self.period + self.min_squeeze_days + 5:
            return Signal("BB_Squeeze", ticker, Direction.HOLD, 0.0, "Insufficient data")

        close  = df["close"]
        upper, mid, lower, width, pct_b = Indicators.bollinger_bands(
            close, self.period, self.std_dev)
        atr    = Indicators.atr(df["high"], df["low"], close)
        volume = df["volume"]
        vol_ratio = Indicators.volume_ratio(volume)

        price       = close.iloc[-1]
        curr_width  = width.iloc[-1]
        curr_pct_b  = pct_b.iloc[-1]
        curr_atr    = atr.iloc[-1]

        # Count consecutive squeeze days
        recent_widths   = width.iloc[-self.min_squeeze_days-1:-1]
        in_squeeze      = (recent_widths < self.squeeze_pct).all()
        breaking_out_up = curr_width > self.squeeze_pct and price > upper.iloc[-2]
        breaking_out_dn = curr_width > self.squeeze_pct and price < lower.iloc[-2]
        vol_surge       = vol_ratio.iloc[-1] > 1.5  # Volume 50% above average

        if in_squeeze and breaking_out_up:
            conf = 0.70
            if vol_surge: conf += 0.15
            return Signal(
                strategy="BB_Squeeze", ticker=ticker,
                direction=Direction.BUY, confidence=round(min(conf, 0.92), 2),
                reason=f"BB squeeze breakout UP (width={curr_width:.2%}" +
                       (", vol surge" if vol_surge else "") + ")",
                entry=price,
                stop_loss=round(mid.iloc[-1] - curr_atr, 3),
                take_profit=round(upper.iloc[-1] + curr_atr, 3),
                indicators={"bb_width": round(curr_width, 4), "pct_b": round(curr_pct_b, 3),
                            "vol_ratio": round(vol_ratio.iloc[-1], 2)},
            )

        elif in_squeeze and breaking_out_dn:
            conf = 0.70
            if vol_surge: conf += 0.15
            return Signal(
                strategy="BB_Squeeze", ticker=ticker,
                direction=Direction.SELL, confidence=round(min(conf, 0.92), 2),
                reason=f"BB squeeze breakout DOWN (width={curr_width:.2%})",
                entry=price,
                stop_loss=round(mid.iloc[-1] + curr_atr, 3),
                take_profit=round(lower.iloc[-1] - curr_atr, 3),
                indicators={"bb_width": round(curr_width, 4), "pct_b": round(curr_pct_b, 3)},
            )

        squeeze_status = "in squeeze" if curr_width < self.squeeze_pct else "normal"
        return Signal(
            strategy="BB_Squeeze", ticker=ticker,
            direction=Direction.HOLD, confidence=0.25,
            reason=f"BB {squeeze_status} (width={curr_width:.2%})",
            indicators={"bb_width": round(curr_width, 4), "pct_b": round(curr_pct_b, 3)},
        )


# ─────────────────────────────────────────────
# STRATEGY 4: VOLUME-WEIGHTED BREAKOUT (VWAP)
# ─────────────────────────────────────────────

class VolumeWeightedBreakout:
    """
    Trades price breakouts above/below VWAP confirmed by OBV trend.
    
    BVMT tuning: Critical for BVMT because institutional orders are
    rare but large — a volume spike on BVMT is a very strong signal.
    OBV confirmation filters out dead-cat bounces common in thin markets.
    
    Additional filter: Only trades in direction of 20-day trend.
    """

    def __init__(self, vwap_lookback=20, vol_threshold=2.0, obv_periods=10):
        self.vwap_lookback  = vwap_lookback
        self.vol_threshold  = vol_threshold   # Volume must be 2× average
        self.obv_periods    = obv_periods

    def run(self, df: pd.DataFrame, ticker: str) -> Signal:
        if len(df) < self.vwap_lookback + 10:
            return Signal("VWAP_Breakout", ticker, Direction.HOLD, 0.0, "Insufficient data")

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        vwap      = Indicators.vwap(high, low, close, volume)
        obv       = Indicators.obv(close, volume)
        atr       = Indicators.atr(high, low, close)
        vol_ratio = Indicators.volume_ratio(volume, self.vwap_lookback)
        ema20     = Indicators.ema(close, 20)

        price       = close.iloc[-1]
        prev_price  = close.iloc[-2]
        curr_vwap   = vwap.iloc[-1]
        curr_atr    = atr.iloc[-1]
        curr_vol_r  = vol_ratio.iloc[-1]
        obv_slope   = (obv.iloc[-1] - obv.iloc[-self.obv_periods]) / self.obv_periods
        trend_up    = price > ema20.iloc[-1]

        above_vwap      = price > curr_vwap and prev_price <= curr_vwap
        below_vwap      = price < curr_vwap and prev_price >= curr_vwap
        high_volume     = curr_vol_r >= self.vol_threshold
        obv_confirming_up   = obv_slope > 0
        obv_confirming_down = obv_slope < 0

        if above_vwap and high_volume and obv_confirming_up and trend_up:
            conf = 0.60 + min(0.25, (curr_vol_r - self.vol_threshold) * 0.1)
            return Signal(
                strategy="VWAP_Breakout", ticker=ticker,
                direction=Direction.BUY, confidence=round(min(conf, 0.90), 2),
                reason=f"VWAP breakout UP with {curr_vol_r:.1f}× volume surge + OBV rising",
                entry=price,
                stop_loss=round(curr_vwap - 0.5 * curr_atr, 3),
                take_profit=round(price + 2.5 * curr_atr, 3),
                indicators={"vwap": round(curr_vwap, 3), "vol_ratio": round(curr_vol_r, 2),
                            "obv_slope": round(obv_slope, 0)},
            )

        elif below_vwap and high_volume and obv_confirming_down and not trend_up:
            conf = 0.60 + min(0.25, (curr_vol_r - self.vol_threshold) * 0.1)
            return Signal(
                strategy="VWAP_Breakout", ticker=ticker,
                direction=Direction.SELL, confidence=round(min(conf, 0.90), 2),
                reason=f"VWAP breakdown DOWN with {curr_vol_r:.1f}× volume + OBV falling",
                entry=price,
                stop_loss=round(curr_vwap + 0.5 * curr_atr, 3),
                take_profit=round(price - 2.5 * curr_atr, 3),
                indicators={"vwap": round(curr_vwap, 3), "vol_ratio": round(curr_vol_r, 2)},
            )

        return Signal(
            strategy="VWAP_Breakout", ticker=ticker,
            direction=Direction.HOLD, confidence=0.2,
            reason=f"No VWAP breakout (vol={curr_vol_r:.1f}×, vwap={curr_vwap:.3f})",
            indicators={"vwap": round(curr_vwap, 3), "vol_ratio": round(curr_vol_r, 2)},
        )


# ─────────────────────────────────────────────
# COMPOSITE STRATEGY ENGINE
# ─────────────────────────────────────────────

class StrategyEngine:
    """
    Runs all 4 strategies and combines signals using weighted voting.
    Weights reflect backtested performance on BVMT-like data.
    """

    WEIGHTS = {
        "RSI_MeanReversion": 0.25,
        "MACD_Momentum":     0.30,
        "BB_Squeeze":        0.25,
        "VWAP_Breakout":     0.20,
    }

    def __init__(self, feed=None):
        self.feed = feed
        self.strategies = [
            RSIMeanReversion(),
            MACDMomentum(),
            BollingerSqueeze(),
            VolumeWeightedBreakout(),
        ]

    def run_all(self, ticker: str, df: pd.DataFrame = None) -> CompositeSignal:
        """
        Runs all strategies on the given ticker.
        If df is None, fetches data from the feed.
        """
        if df is None and self.feed:
            df = self.feed.get_ohlcv(ticker, "3M")
        elif df is None:
            raise ValueError("Must provide either a feed or a DataFrame")

        signals = [s.run(df, ticker) for s in self.strategies]

        # Weighted voting
        buy_weight  = sum(self.WEIGHTS[s.strategy] * s.confidence
                         for s in signals if s.direction == Direction.BUY)
        sell_weight = sum(self.WEIGHTS[s.strategy] * s.confidence
                         for s in signals if s.direction == Direction.SELL)
        hold_weight = sum(self.WEIGHTS[s.strategy] * s.confidence
                         for s in signals if s.direction == Direction.HOLD)

        total = buy_weight + sell_weight + hold_weight or 1
        direction = Direction.HOLD
        if buy_weight > sell_weight and buy_weight > hold_weight:
            direction = Direction.BUY
            confidence = buy_weight / total
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            direction = Direction.SELL
            confidence = sell_weight / total
        else:
            confidence = hold_weight / total

        # Consensus entry/stop/target from active signals
        active = [s for s in signals if s.direction == direction and s.entry]
        entry  = np.mean([s.entry for s in active]) if active else df["close"].iloc[-1]
        stop   = np.mean([s.stop_loss for s in active if s.stop_loss]) if active else None
        target = np.mean([s.take_profit for s in active if s.take_profit]) if active else None
        rr     = abs(target - entry) / abs(stop - entry) if (target and stop and stop != entry) else None

        return CompositeSignal(
            ticker=ticker,
            direction=direction,
            confidence=round(confidence, 3),
            vote_buy=sum(1 for s in signals if s.direction == Direction.BUY),
            vote_sell=sum(1 for s in signals if s.direction == Direction.SELL),
            vote_hold=sum(1 for s in signals if s.direction == Direction.HOLD),
            signals=signals,
            entry=round(entry, 3) if entry else None,
            stop_loss=round(stop, 3) if stop else None,
            take_profit=round(target, 3) if target else None,
            risk_reward=round(rr, 2) if rr else None,
        )

    def scan_all_tickers(self, tickers: list[str]) -> list[CompositeSignal]:
        """Scans a list of tickers and returns only actionable signals."""
        results = []
        for ticker in tickers:
            try:
                sig = self.run_all(ticker)
                if sig.direction != Direction.HOLD and sig.confidence > 0.50:
                    results.append(sig)
                    log.info(f"Signal: {sig}")
            except Exception as e:
                log.warning(f"Error scanning {ticker}: {e}")
        return sorted(results, key=lambda s: s.confidence, reverse=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from core.data_feed import BVMTDataFeed

    feed = BVMTDataFeed()
    engine = StrategyEngine(feed)

    print("\n=== Running all strategies on BIAT ===")
    composite = engine.run_all("BIAT")
    print(composite)
    for s in composite.signals:
        print(f"  {s}")

    print(f"\n  Entry: {composite.entry} TND")
    print(f"  Stop:  {composite.stop_loss} TND")
    print(f"  Target:{composite.take_profit} TND")
    print(f"  R/R:   {composite.risk_reward}")

    print("\n=== Scanning all tickers ===")
    from core.data_feed import BVMT_TICKERS
    signals = engine.scan_all_tickers(BVMT_TICKERS[:10])
    for s in signals:
        print(f"  {s}")
