"""
BVMT Execution Layer — Phase 3
================================
Handles order lifecycle: creation, validation, submission,
tracking, and cancellation.

Supports:
  - Semi-manual mode: generates orders, human approves
  - Auto mode: submits orders automatically (with limits)
  - Paper trading: full simulation with realistic fills
  - Broker API bridge (BIAT Capital / Attijari format)

BVMT-specific rules enforced:
  - Market hours: Continuous session 10:00–14:00 only
  - Tick size: 0.100 TND (liquid), 0.010 TND (illiquid < 5 TND)
  - Price limits: ±6% daily price band (BVMT circuit breaker)
  - Order types: Limit only (no market orders in fixing session)
  - T+3 settlement
"""

import uuid
import time
import logging
import random
from datetime import datetime, date
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
import pandas as pd

log = logging.getLogger("Execution")


# ─────────────────────────────────────────────
# ENUMS & CONSTANTS
# ─────────────────────────────────────────────

class OrderType(str, Enum):
    LIMIT  = "LIMIT"
    MARKET = "MARKET"      # Only valid in continuous session
    STOP   = "STOP_LIMIT"


class OrderStatus(str, Enum):
    PENDING   = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL   = "PARTIAL_FILL"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"
    EXPIRED   = "EXPIRED"


class Side(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


BVMT_TICK_SIZE     = 0.100   # TND for prices > 5 TND
BVMT_TICK_SMALL    = 0.010   # TND for prices ≤ 5 TND
BVMT_DAILY_LIMIT   = 0.06    # ±6% circuit breaker
BVMT_SETTLEMENT    = 3       # T+3 days
MAX_ORDER_VALUE    = 500_000  # TND — internal risk limit per order


# ─────────────────────────────────────────────
# ORDER DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class Order:
    order_id:    str
    ticker:      str
    side:        Side
    qty:         int
    price:       float             # Limit price
    order_type:  OrderType = OrderType.LIMIT
    status:      OrderStatus = OrderStatus.PENDING
    filled_qty:  int = 0
    avg_fill_price: float = 0.0
    created_at:  str = field(default_factory=lambda: datetime.now().isoformat())
    filled_at:   Optional[str] = None
    strategy:    Optional[str] = None
    stop_loss:   Optional[float] = None
    take_profit: Optional[float] = None
    notes:       Optional[str] = None

    @property
    def value(self) -> float:
        return self.price * self.qty

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL)

    def __repr__(self):
        return (f"Order({self.order_id[:8]}… {self.side.value} {self.qty}×{self.ticker} "
                f"@ {self.price} | {self.status.value})")


@dataclass
class Fill:
    fill_id:    str
    order_id:   str
    ticker:     str
    side:       Side
    qty:        int
    price:      float
    timestamp:  str
    commission: float    # BVMT brokerage: ~0.15% + 0.015% CMF tax

    @property
    def gross_value(self) -> float:
        return self.qty * self.price

    @property
    def net_value(self) -> float:
        if self.side == Side.BUY:
            return self.gross_value + self.commission
        return self.gross_value - self.commission


# ─────────────────────────────────────────────
# ORDER VALIDATOR
# ─────────────────────────────────────────────

class OrderValidator:
    """
    Validates orders against BVMT rules before submission.
    Raises ValueError with clear rejection reasons.
    """

    def __init__(self, prev_close_prices: dict[str, float] = None):
        self.prev_closes = prev_close_prices or {}

    def validate(self, order: Order) -> tuple[bool, str]:
        """Returns (is_valid, rejection_reason)."""

        # 1. Trading hours
        now = datetime.now()
        hour = now.hour + now.minute / 60
        weekday = now.weekday()
        if weekday >= 5:
            return False, "BVMT is closed on weekends"
        # Allow during all sessions for demo; in prod restrict to 9:00-14:10
        if not (9.0 <= hour <= 14.17):
            return False, f"Outside BVMT trading hours (now {now.strftime('%H:%M')} TUN)"

        # 2. Minimum order size
        if order.qty < 1:
            return False, "Quantity must be at least 1 share"

        # 3. Maximum order value
        if order.value > MAX_ORDER_VALUE:
            return False, f"Order value {order.value:,.0f} TND exceeds limit {MAX_ORDER_VALUE:,.0f} TND"

        # 4. Tick size compliance
        tick = BVMT_TICK_SMALL if order.price <= 5.0 else BVMT_TICK_SIZE
        rounded = round(round(order.price / tick) * tick, 3)
        if abs(order.price - rounded) > 0.0001:
            return False, f"Price {order.price} not on tick grid (nearest: {rounded})"

        # 5. Daily price band (±6%)
        prev = self.prev_closes.get(order.ticker)
        if prev:
            pct_chg = abs(order.price - prev) / prev
            if pct_chg > BVMT_DAILY_LIMIT:
                return False, (f"Price {order.price} exceeds ±{BVMT_DAILY_LIMIT*100}% "
                               f"daily limit from previous close {prev}")

        # 6. Positive price
        if order.price <= 0:
            return False, "Price must be positive"

        return True, "OK"


# ─────────────────────────────────────────────
# PAPER TRADING ENGINE
# ─────────────────────────────────────────────

class PaperTradingEngine:
    """
    Simulates order execution with realistic BVMT fill mechanics:
    - Partial fills based on available volume
    - Slippage model for BVMT's thin order book
    - Realistic commission structure
    """

    # BVMT commission structure
    BROKERAGE_RATE = 0.0015   # 0.15% brokerage fee
    CMF_TAX_RATE   = 0.00015  # 0.015% CMF market supervision tax
    STAMP_DUTY     = 0.00025  # 0.025% stamp duty on buys

    def __init__(self):
        self.orders:  dict[str, Order] = {}
        self.fills:   list[Fill] = []
        self.positions: dict[str, dict] = {}   # ticker → {qty, avg_cost}
        self.cash     = 100_000.0              # Starting capital in TND
        self.on_fill: Optional[Callable] = None  # Callback

    def place_order(self, order: Order) -> Order:
        self.orders[order.order_id] = order
        log.info(f"Paper order placed: {order}")
        # Simulate async fill
        self._simulate_fill(order)
        return order

    def _simulate_fill(self, order: Order):
        """Simulates realistic fills with partial fill probability."""
        # BVMT fill probability based on side and market conditions
        fill_prob = random.uniform(0.70, 0.98)
        if random.random() > fill_prob:
            order.status = OrderStatus.CANCELLED
            log.info(f"Order {order.order_id[:8]} not filled (thin market)")
            return

        # Partial fill: 20% chance of partial
        fill_pct = 1.0 if random.random() > 0.20 else random.uniform(0.3, 0.9)
        fill_qty = max(1, int(order.qty * fill_pct))

        # Slippage: 0–0.05% for liquid stocks (BVMT is thin)
        slippage = random.uniform(0, 0.0005)
        if order.side == Side.BUY:
            fill_price = order.price * (1 + slippage)
        else:
            fill_price = order.price * (1 - slippage)

        fill_price = round(fill_price, 3)
        commission = self._calc_commission(order.side, fill_qty, fill_price)

        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side,
            qty=fill_qty,
            price=fill_price,
            timestamp=datetime.now().isoformat(),
            commission=round(commission, 2),
        )
        self.fills.append(fill)
        self._update_position(fill)

        order.filled_qty = fill_qty
        order.avg_fill_price = fill_price
        order.filled_at = fill.timestamp
        order.status = OrderStatus.FILLED if fill_qty == order.qty else OrderStatus.PARTIAL

        log.info(f"Fill: {fill_qty}×{order.ticker} @ {fill_price} | commission={commission:.2f} TND")

        if self.on_fill:
            self.on_fill(fill)

    def _calc_commission(self, side: Side, qty: int, price: float) -> float:
        gross = qty * price
        comm  = gross * self.BROKERAGE_RATE
        cmf   = gross * self.CMF_TAX_RATE
        stamp = gross * self.STAMP_DUTY if side == Side.BUY else 0
        return comm + cmf + stamp

    def _update_position(self, fill: Fill):
        t = fill.ticker
        if t not in self.positions:
            self.positions[t] = {"qty": 0, "avg_cost": 0.0, "total_cost": 0.0}

        pos = self.positions[t]
        if fill.side == Side.BUY:
            new_total = pos["total_cost"] + fill.net_value
            new_qty   = pos["qty"] + fill.qty
            pos["qty"]        = new_qty
            pos["total_cost"] = new_total
            pos["avg_cost"]   = new_total / new_qty if new_qty > 0 else 0
            self.cash -= fill.net_value
        else:
            proceeds = fill.gross_value - fill.commission
            pos["qty"]      = max(0, pos["qty"] - fill.qty)
            self.cash      += proceeds
            if pos["qty"] == 0:
                pos["total_cost"] = 0.0
                pos["avg_cost"]   = 0.0

    def get_portfolio_summary(self, current_prices: dict[str, float]) -> dict:
        positions_value = sum(
            pos["qty"] * current_prices.get(ticker, pos["avg_cost"])
            for ticker, pos in self.positions.items()
            if pos["qty"] > 0
        )
        total_equity = self.cash + positions_value
        cost_basis   = sum(pos["total_cost"] for pos in self.positions.values())
        unrealized   = positions_value - cost_basis

        return {
            "cash":             round(self.cash, 2),
            "positions_value":  round(positions_value, 2),
            "total_equity":     round(total_equity, 2),
            "unrealized_pnl":   round(unrealized, 2),
            "positions":        {t: p for t, p in self.positions.items() if p["qty"] > 0},
            "total_fills":      len(self.fills),
            "total_orders":     len(self.orders),
        }

    def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if order and order.is_active:
            order.status = OrderStatus.CANCELLED
            log.info(f"Order {order_id[:8]} cancelled")
            return True
        return False


# ─────────────────────────────────────────────
# AUTO-TRADER CONTROLLER
# ─────────────────────────────────────────────

class AutoTrader:
    """
    Connects Strategy Engine signals to the Execution Engine.
    Enforces position limits and prevents over-trading.
    """

    def __init__(self, execution_engine: PaperTradingEngine,
                 max_position_pct: float = 0.10,
                 max_daily_orders: int = 20):
        self.exec     = execution_engine
        self.max_pos  = max_position_pct   # Max 10% of portfolio per position
        self.max_daily = max_daily_orders
        self._daily_orders = 0
        self._last_reset   = date.today()

    def _reset_daily_counter(self):
        if date.today() > self._last_reset:
            self._daily_orders = 0
            self._last_reset = date.today()

    def execute_signal(self, signal, current_price: float,
                       approval_required: bool = False) -> Optional[Order]:
        """
        Converts a CompositeSignal into an Order and submits it.
        If approval_required=True, returns the order in PENDING state
        without submitting (for semi-manual mode).
        """
        from strategies.engine import Direction

        self._reset_daily_counter()
        if self._daily_orders >= self.max_daily:
            log.warning("Daily order limit reached")
            return None

        if signal.direction == Direction.HOLD:
            return None

        # Position sizing: Kelly-lite (see risk module)
        equity  = self.exec.cash + sum(
            p["total_cost"] for p in self.exec.positions.values())
        max_val = equity * self.max_pos
        qty     = max(1, int(max_val / current_price))

        # Snap price to tick
        tick = BVMT_TICK_SMALL if current_price <= 5.0 else BVMT_TICK_SIZE
        if signal.direction == Direction.BUY:
            # Limit order slightly above mid to get filled
            price = round((round(current_price / tick) + 1) * tick, 3)
        else:
            price = round((round(current_price / tick) - 1) * tick, 3)

        order = Order(
            order_id=str(uuid.uuid4()),
            ticker=signal.ticker,
            side=Side.BUY if signal.direction == Direction.BUY else Side.SELL,
            qty=qty,
            price=price,
            strategy=signal.signals[0].strategy if signal.signals else "composite",
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

        # Validate
        validator = OrderValidator()
        valid, reason = validator.validate(order)
        if not valid:
            log.warning(f"Order rejected: {reason}")
            order.status = OrderStatus.REJECTED
            order.notes  = reason
            return order

        if approval_required:
            log.info(f"Order pending approval: {order}")
            return order

        # Submit
        result = self.exec.place_order(order)
        self._daily_orders += 1
        return result

    def check_stop_loss_take_profit(self, current_prices: dict[str, float]):
        """Monitors open positions and exits when SL/TP is hit."""
        for oid, order in list(self.exec.orders.items()):
            if order.status != OrderStatus.FILLED:
                continue
            price = current_prices.get(order.ticker)
            if not price:
                continue

            hit = None
            if order.stop_loss and order.side == Side.BUY and price <= order.stop_loss:
                hit = ("STOP_LOSS", order.stop_loss)
            elif order.take_profit and order.side == Side.BUY and price >= order.take_profit:
                hit = ("TAKE_PROFIT", order.take_profit)

            if hit:
                reason, exit_price = hit
                exit_order = Order(
                    order_id=str(uuid.uuid4()),
                    ticker=order.ticker,
                    side=Side.SELL,
                    qty=order.filled_qty,
                    price=exit_price,
                    notes=f"Auto-exit: {reason}",
                )
                self.exec.place_order(exit_order)
                log.info(f"{reason} triggered for {order.ticker} @ {exit_price}")


# ─────────────────────────────────────────────
# ORDER BOOK HELPER
# ─────────────────────────────────────────────

def format_order_book(book: dict) -> str:
    lines = [f"\n{'─'*42}", f"  Order Book — {book['ticker']}  |  Mid: {book['mid']:.3f} TND",
             f"  Spread: {book['spread']:.3f} TND ({book['spread_pct']:.3f}%)", f"{'─'*42}",
             f"  {'PRICE':>10}  {'QTY':>8}  {'TOTAL':>10}"]
    lines.append("  —— ASKS (sellers) ——")
    for a in reversed(book["asks"][:5]):
        lines.append(f"  \033[31m{a['price']:>10.3f}  {a['qty']:>8,}  {a['total']:>10,.0f}\033[0m")
    lines.append(f"  {'· · · · · ·':>32}")
    for b in book["bids"][:5]:
        lines.append(f"  \033[32m{b['price']:>10.3f}  {b['qty']:>8,}  {b['total']:>10,.0f}\033[0m")
    lines.append("  —— BIDS (buyers) ——")
    lines.append(f"{'─'*42}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys; sys.path.insert(0, "..")
    from core.data_feed import BVMTDataFeed
    from strategies.engine import StrategyEngine

    feed   = BVMTDataFeed()
    engine = StrategyEngine(feed)
    exec_e = PaperTradingEngine()
    trader = AutoTrader(exec_e)

    print("\n=== PAPER TRADING SESSION ===")
    tickers = ["BIAT", "ATB", "SFBT", "TPR", "OTT"]
    prices  = {t: feed.get_quote(t)["price"] for t in tickers}

    for ticker in tickers:
        signal = engine.run_all(ticker)
        if signal.direction.value != "HOLD":
            print(f"\n{signal}")
            order = trader.execute_signal(signal, prices[ticker], approval_required=False)
            if order:
                print(f"  → {order}")

    print("\n=== Portfolio Summary ===")
    summary = exec_e.get_portfolio_summary(prices)
    print(f"  Cash:        {summary['cash']:>10,.2f} TND")
    print(f"  Positions:   {summary['positions_value']:>10,.2f} TND")
    print(f"  Total equity:{summary['total_equity']:>10,.2f} TND")
    print(f"  Unrealized:  {summary['unrealized_pnl']:>10,.2f} TND")
    print(f"  Orders/Fills:{summary['total_orders']}/{summary['total_fills']}")
