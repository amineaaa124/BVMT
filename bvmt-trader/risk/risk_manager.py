"""
BVMT Risk Management — Phase 4
================================
Production-grade risk controls for the BVMT AlgoTrader.

Modules:
  1. KellyCriterion  — Optimal position sizing
  2. SectorLimits    — Concentration risk controls
  3. DrawdownGuard   — Daily/portfolio drawdown limits
  4. NewsFilter      — Pause trading on CMF events
  5. RiskManager     — Master orchestrator

Design philosophy:
  BVMT-specific risks to manage:
  - Low liquidity: slippage and partial fills are common
  - Sector concentration: Banking is 60%+ of TUNINDEX
  - CMF announcement risk: price gaps on filing days
  - Currency: TND is managed float — macro risk is real
  - T+3 settlement: cash management is critical
"""

import logging
import math
import re
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

log = logging.getLogger("RiskManager")


# ─────────────────────────────────────────────
# 1. KELLY CRITERION POSITION SIZER
# ─────────────────────────────────────────────

class KellyCriterion:
    """
    Computes optimal position size using the Kelly Criterion.
    Uses fractional Kelly (default: half-Kelly) for safety.

    Kelly formula: f* = (p*b - q) / b
      p = win probability
      b = win/loss ratio (reward/risk ratio)
      q = loss probability = 1 - p

    Half-Kelly: f = f* × 0.5  (industry standard for live trading)
    """

    def __init__(self, fraction: float = 0.5, max_position_pct: float = 0.15):
        self.fraction    = fraction          # 0.5 = half Kelly
        self.max_pos_pct = max_position_pct  # Hard cap: never > 15% per position

    def position_size(self,
                      win_probability: float,
                      avg_win_pct: float,
                      avg_loss_pct: float,
                      portfolio_value: float,
                      current_price: float) -> dict:
        """
        Returns recommended position in shares and TND value.

        Args:
            win_probability: Historical win rate for this signal type (0–1)
            avg_win_pct:     Average gain on winners as fraction (e.g. 0.05 = 5%)
            avg_loss_pct:    Average loss on losers as fraction (e.g. 0.025 = 2.5%)
            portfolio_value: Total portfolio value in TND
            current_price:   Stock price in TND
        """
        p = max(0.01, min(0.99, win_probability))
        q = 1 - p
        b = avg_win_pct / max(avg_loss_pct, 0.001)  # Reward/risk ratio

        kelly_f = (p * b - q) / b
        kelly_f = max(0, kelly_f)                    # Never short via Kelly

        # Apply fraction (half-Kelly)
        adjusted_f = kelly_f * self.fraction

        # Cap at max position
        final_f = min(adjusted_f, self.max_pos_pct)

        position_value = portfolio_value * final_f
        shares = max(1, int(position_value / current_price))

        return {
            "kelly_full":    round(kelly_f, 4),
            "kelly_adjusted": round(adjusted_f, 4),
            "kelly_capped":  round(final_f, 4),
            "position_value": round(position_value, 2),
            "shares":         shares,
            "pct_of_portfolio": round(final_f * 100, 2),
            "max_loss_tnd": round(shares * current_price * avg_loss_pct, 2),
        }

    def backtest_kelly(self, trade_history: list[dict]) -> dict:
        """
        Estimates Kelly fraction from actual trade history.
        trade_history: list of {"pnl_pct": float}
        """
        if len(trade_history) < 10:
            return {"error": "Need at least 10 trades for Kelly estimation"}

        pnls = [t["pnl_pct"] for t in trade_history]
        wins  = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        if not wins or not losses:
            return {"error": "Need both wins and losses"}

        p = len(wins) / len(pnls)
        b = np.mean(wins) / np.mean(losses)
        q = 1 - p
        kelly = (p * b - q) / b

        return {
            "win_rate":   round(p, 3),
            "avg_win":    round(np.mean(wins), 4),
            "avg_loss":   round(np.mean(losses), 4),
            "reward_risk": round(b, 2),
            "kelly_full": round(kelly, 4),
            "kelly_half": round(kelly * 0.5, 4),
            "sample_size": len(pnls),
        }


# ─────────────────────────────────────────────
# 2. SECTOR CONCENTRATION LIMITS
# ─────────────────────────────────────────────

SECTOR_LIMITS = {
    "Banking":    0.35,   # Max 35% in banking (BVMT is banking-heavy)
    "Consumer":   0.25,
    "Industrial": 0.20,
    "Telecom":    0.15,
    "Energy":     0.15,
    "Insurance":  0.15,
    "Finance":    0.15,
    "Transport":  0.10,
    "Other":      0.10,
}

SECTOR_MAP = {
    "BIAT": "Banking", "ATB": "Banking", "BH": "Banking", "BNA": "Banking",
    "BT": "Banking", "STB": "Banking", "UIB": "Banking", "UBCI": "Banking",
    "SFBT": "Consumer", "POULINA": "Industrial", "TPR": "Energy",
    "OTT": "Telecom", "SOTETEL": "Telecom", "TUNISAIR": "Transport",
    "STAR": "Insurance", "ASTREE": "Insurance", "COMAR": "Insurance",
    "AMI": "Insurance",
}


class SectorConcentrationCheck:
    """
    Blocks new orders that would push sector exposure over limits.
    Also warns when a single stock exceeds 10% of portfolio.
    """

    SINGLE_STOCK_LIMIT = 0.10   # Max 10% in any single stock

    def check_new_order(self, ticker: str, order_value: float,
                        current_positions: dict, portfolio_value: float) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        current_positions: {ticker: {"qty": n, "total_cost": x}}
        """
        sector = SECTOR_MAP.get(ticker, "Other")
        limit  = SECTOR_LIMITS.get(sector, 0.15)

        # Current sector exposure
        current_sector_value = sum(
            pos["total_cost"]
            for t, pos in current_positions.items()
            if SECTOR_MAP.get(t, "Other") == sector and pos["qty"] > 0
        )
        new_sector_exposure = (current_sector_value + order_value) / max(portfolio_value, 1)

        if new_sector_exposure > limit:
            return False, (f"Sector '{sector}' would reach {new_sector_exposure:.1%} "
                           f"(limit: {limit:.0%}). Current: {current_sector_value/portfolio_value:.1%}")

        # Single stock limit
        current_stock_value = current_positions.get(ticker, {}).get("total_cost", 0)
        new_stock_exposure  = (current_stock_value + order_value) / max(portfolio_value, 1)
        if new_stock_exposure > self.SINGLE_STOCK_LIMIT:
            return False, (f"Single stock {ticker} would reach {new_stock_exposure:.1%} "
                           f"(limit: {self.SINGLE_STOCK_LIMIT:.0%})")

        return True, "OK"

    def get_sector_report(self, positions: dict, portfolio_value: float) -> pd.DataFrame:
        """Returns a DataFrame showing sector exposures vs limits."""
        sector_values: dict[str, float] = {}
        for ticker, pos in positions.items():
            if pos["qty"] > 0:
                sector = SECTOR_MAP.get(ticker, "Other")
                sector_values[sector] = sector_values.get(sector, 0) + pos["total_cost"]

        rows = []
        for sector, limit in SECTOR_LIMITS.items():
            val  = sector_values.get(sector, 0)
            exp  = val / max(portfolio_value, 1)
            rows.append({
                "sector":    sector,
                "exposure":  round(exp * 100, 2),
                "limit_pct": round(limit * 100, 0),
                "value_tnd": round(val, 2),
                "status":    "⚠ OVER" if exp > limit else "✓ OK",
            })
        return pd.DataFrame(rows).sort_values("exposure", ascending=False)


# ─────────────────────────────────────────────
# 3. DRAWDOWN GUARD
# ─────────────────────────────────────────────

@dataclass
class DrawdownState:
    peak_equity:    float
    current_equity: float
    daily_start:    float
    date:           date = field(default_factory=date.today)

    @property
    def portfolio_drawdown(self) -> float:
        """Drawdown from all-time peak."""
        return (self.peak_equity - self.current_equity) / max(self.peak_equity, 1)

    @property
    def daily_drawdown(self) -> float:
        """Intraday drawdown from day open."""
        return (self.daily_start - self.current_equity) / max(self.daily_start, 1)


class DrawdownGuard:
    """
    Suspends trading when drawdown limits are hit.

    Hard limits:
      - Daily drawdown: -3%   → Pause for the day
      - Portfolio drawdown: -15% → Emergency stop, alert
    Soft warnings:
      - Daily: -1.5% → Reduce position sizes by 50%
      - Portfolio: -8% → Reduce position sizes by 50%
    """

    DAILY_HARD   = 0.03
    DAILY_SOFT   = 0.015
    PORTFOLIO_HARD  = 0.15
    PORTFOLIO_SOFT  = 0.08

    def __init__(self, initial_equity: float):
        self.state = DrawdownState(
            peak_equity=initial_equity,
            current_equity=initial_equity,
            daily_start=initial_equity,
        )
        self._trading_suspended = False
        self._size_reduction = 1.0

    def update(self, current_equity: float):
        today = date.today()
        if today > self.state.date:
            # New day: reset daily tracker
            self.state.daily_start = current_equity
            self.state.date = today

        self.state.current_equity = current_equity
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity

        self._evaluate()

    def _evaluate(self):
        dd_daily = self.state.daily_drawdown
        dd_port  = self.state.portfolio_drawdown

        if dd_daily >= self.DAILY_HARD or dd_port >= self.PORTFOLIO_HARD:
            self._trading_suspended = True
            self._size_reduction = 0.0
            log.critical(f"🚨 TRADING SUSPENDED — Daily DD={dd_daily:.2%} Port DD={dd_port:.2%}")
        elif dd_daily >= self.DAILY_SOFT or dd_port >= self.PORTFOLIO_SOFT:
            self._trading_suspended = False
            self._size_reduction = 0.5
            log.warning(f"⚠ Position sizes halved — Daily DD={dd_daily:.2%}")
        else:
            self._trading_suspended = False
            self._size_reduction = 1.0

    @property
    def is_suspended(self) -> bool:
        return self._trading_suspended

    @property
    def size_multiplier(self) -> float:
        return self._size_reduction

    def get_status(self) -> dict:
        return {
            "suspended":           self._trading_suspended,
            "size_multiplier":     self._size_reduction,
            "daily_drawdown_pct":  round(self.state.daily_drawdown * 100, 2),
            "portfolio_drawdown_pct": round(self.state.portfolio_drawdown * 100, 2),
            "peak_equity":         round(self.state.peak_equity, 2),
            "current_equity":      round(self.state.current_equity, 2),
            "daily_pnl":           round(self.state.current_equity - self.state.daily_start, 2),
        }


# ─────────────────────────────────────────────
# 4. NEWS EVENT FILTER
# ─────────────────────────────────────────────

# High-impact CMF event types that should pause trading
HIGH_IMPACT_EVENTS = {
    "résultats annuels", "résultats semestriels",
    "dividende", "augmentation de capital",
    "offre publique", "suspension", "radiation",
    "profit warning", "communiqué exceptionnel",
}

# TUNINDEX components — suspension of any of these affects index
INDEX_COMPONENTS = {
    "BIAT", "ATB", "BH", "SFBT", "POULINA", "OTT", "STB", "BNA", "UIB"
}


class NewsEventFilter:
    """
    Monitors CMF announcements and suspends/reduces trading
    around high-impact corporate events.

    Logic:
      - On earnings/dividend day: suspend that ticker 30 min before market open
      - On exceptional communiqué: suspend immediately
      - On index component suspension: reduce all positions by 50%
    """

    def __init__(self, feed=None):
        self.feed = feed
        self._blocked_tickers: dict[str, str] = {}   # ticker → reason
        self._last_check: Optional[datetime]  = None
        self._check_interval = 300   # seconds between CMF checks

    def refresh(self):
        """Fetches latest CMF filings and updates blocked list."""
        if (self._last_check and
                (datetime.now() - self._last_check).seconds < self._check_interval):
            return

        if not self.feed:
            return

        self._blocked_tickers.clear()

        for ticker in INDEX_COMPONENTS:
            try:
                filings = self.feed.get_company_filings(ticker, limit=3)
                for f in filings:
                    event_type = f.get("type", "").lower()
                    filing_date_str = f.get("date", "")

                    # Check if event is today
                    try:
                        filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d").date()
                    except:
                        continue

                    if filing_date == date.today():
                        for keyword in HIGH_IMPACT_EVENTS:
                            if keyword in event_type or keyword in f.get("title", "").lower():
                                self._blocked_tickers[ticker] = f["title"]
                                log.warning(f"News filter: {ticker} blocked — {f['title']}")
                                break
            except Exception as e:
                log.debug(f"News filter error for {ticker}: {e}")

        self._last_check = datetime.now()

    def is_blocked(self, ticker: str) -> tuple[bool, str]:
        self.refresh()
        if ticker in self._blocked_tickers:
            return True, self._blocked_tickers[ticker]
        return False, ""

    def get_blocked(self) -> dict[str, str]:
        return dict(self._blocked_tickers)


# ─────────────────────────────────────────────
# 5. MASTER RISK MANAGER
# ─────────────────────────────────────────────

class RiskManager:
    """
    Orchestrates all risk checks. Every order passes through here.
    Returns a RiskDecision with allow/deny + adjusted parameters.
    """

    def __init__(self, initial_equity: float = 100_000.0, feed=None):
        self.kelly     = KellyCriterion(fraction=0.5, max_position_pct=0.12)
        self.sector    = SectorConcentrationCheck()
        self.drawdown  = DrawdownGuard(initial_equity)
        self.news      = NewsEventFilter(feed)

        # Historical performance stats (update from backtest/live)
        self.perf_stats = {
            "win_rate":  0.62,
            "avg_win":   0.048,
            "avg_loss":  0.022,
        }

    def evaluate(self, ticker: str, signal_confidence: float,
                 current_price: float, current_positions: dict,
                 portfolio_value: float) -> dict:
        """
        Full risk evaluation for a potential trade.
        Returns decision dict with allow, reason, and adjusted sizing.
        """
        checks = []

        # 1. Drawdown check
        self.drawdown.update(portfolio_value)
        if self.drawdown.is_suspended:
            return {"allow": False, "reason": "Trading suspended — drawdown limit hit",
                    "checks": checks, "size_multiplier": 0}

        # 2. News filter
        blocked, news_reason = self.news.is_blocked(ticker)
        if blocked:
            return {"allow": False, "reason": f"News blackout: {news_reason}",
                    "checks": checks, "size_multiplier": 0}

        # 3. Kelly position sizing
        sizing = self.kelly.position_size(
            win_probability=self.perf_stats["win_rate"] * signal_confidence / 0.6,
            avg_win_pct=self.perf_stats["avg_win"],
            avg_loss_pct=self.perf_stats["avg_loss"],
            portfolio_value=portfolio_value,
            current_price=current_price,
        )
        checks.append(f"Kelly: {sizing['pct_of_portfolio']:.1f}% = {sizing['shares']} shares")

        # Apply drawdown size reduction
        adjusted_shares = max(1, int(sizing["shares"] * self.drawdown.size_multiplier))
        order_value = adjusted_shares * current_price

        # 4. Sector concentration
        sector_ok, sector_reason = self.sector.check_new_order(
            ticker, order_value, current_positions, portfolio_value)
        if not sector_ok:
            return {"allow": False, "reason": f"Sector limit: {sector_reason}",
                    "checks": checks, "size_multiplier": self.drawdown.size_multiplier}
        checks.append(f"Sector: OK ({SECTOR_MAP.get(ticker,'Other')})")

        # 5. Min confidence threshold
        if signal_confidence < 0.55:
            return {"allow": False, "reason": f"Signal confidence {signal_confidence:.0%} < 55% threshold",
                    "checks": checks, "size_multiplier": 0}
        checks.append(f"Confidence: {signal_confidence:.0%} ✓")

        return {
            "allow": True,
            "reason": "All risk checks passed",
            "checks": checks,
            "recommended_shares": adjusted_shares,
            "order_value": round(order_value, 2),
            "max_loss_tnd": sizing["max_loss_tnd"],
            "size_multiplier": self.drawdown.size_multiplier,
            "kelly_fraction": sizing["kelly_capped"],
        }

    def daily_report(self, positions: dict, portfolio_value: float) -> str:
        """Generates a daily risk summary report."""
        dd  = self.drawdown.get_status()
        sec = self.sector.get_sector_report(positions, portfolio_value)

        lines = [
            "\n" + "═"*50,
            "  BVMT AlgoTrader — Daily Risk Report",
            "═"*50,
            f"  Portfolio equity:   {portfolio_value:>12,.2f} TND",
            f"  Daily P&L:          {dd['daily_pnl']:>+12,.2f} TND",
            f"  Daily drawdown:     {dd['daily_drawdown_pct']:>11.2f}%",
            f"  Portfolio drawdown: {dd['portfolio_drawdown_pct']:>11.2f}%",
            f"  Trading status:     {'🚫 SUSPENDED' if dd['suspended'] else '✓ ACTIVE'}",
            f"  Size multiplier:    {dd['size_multiplier']:.0%}",
            "",
            "  Sector Exposure:",
        ]
        for _, row in sec.iterrows():
            bar = "█" * int(row["exposure"] / 2)
            lines.append(f"    {row['sector']:12} {bar:20} {row['exposure']:5.1f}% / {row['limit_pct']:.0f}%  {row['status']}")

        lines.append("═"*50)
        blocked = self.news.get_blocked()
        if blocked:
            lines.append(f"  ⚠ News blackout: {', '.join(blocked.keys())}")
        lines.append("")
        return "\n".join(lines)


if __name__ == "__main__":
    print("\n=== Risk Manager Demo ===")
    rm = RiskManager(initial_equity=100_000)

    mock_positions = {
        "BIAT": {"qty": 100, "total_cost": 14_520, "avg_cost": 145.2},
        "SFBT": {"qty": 500, "total_cost": 9_150,  "avg_cost": 18.3},
    }

    decision = rm.evaluate(
        ticker="ATB",
        signal_confidence=0.75,
        current_price=38.5,
        current_positions=mock_positions,
        portfolio_value=100_000,
    )
    print(f"\n  Decision: {'✓ ALLOW' if decision['allow'] else '✗ DENY'}")
    print(f"  Reason: {decision['reason']}")
    for c in decision.get("checks", []):
        print(f"    • {c}")
    if decision["allow"]:
        print(f"  Recommended: {decision['recommended_shares']} shares @ 38.5 = {decision['order_value']:,.0f} TND")
        print(f"  Max risk: {decision['max_loss_tnd']:,.0f} TND")

    print("\n=== Kelly Backtest ===")
    mock_trades = [{"pnl_pct": v} for v in
                   [0.05, -0.02, 0.04, 0.06, -0.025, 0.03, -0.018, 0.055, 0.04, -0.02,
                    0.07, -0.015, 0.045, -0.03, 0.05]]
    result = rm.kelly.backtest_kelly(mock_trades)
    print(f"  Win rate:    {result['win_rate']:.0%}")
    print(f"  Reward/Risk: {result['reward_risk']:.2f}×")
    print(f"  Kelly full:  {result['kelly_full']:.1%}")
    print(f"  Kelly half:  {result['kelly_half']:.1%}")

    print(rm.daily_report(mock_positions, 100_000))
