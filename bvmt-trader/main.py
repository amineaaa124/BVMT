"""
BVMT AlgoTrader — Main Orchestrator
=====================================
Ties all 4 phases together into a single runnable system.

  python main.py --mode paper     # Paper trading simulation
  python main.py --mode scan      # One-shot signal scan
  python main.py --mode backtest  # Backtest on historical data
  python main.py --mode risk      # Risk report only
"""

import sys
import time
import argparse
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("BVMT-Main")


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════╗
║       BVMT AlgoTrader — Bourse de Tunis                 ║
║  ──────────────────────────────────────────────────────  ║
║  Phase 1: Data Feed (BVMT/CMF scraping + synthetic)     ║
║  Phase 2: Strategy Engine (RSI/MACD/BB/VWAP)           ║
║  Phase 3: Execution (Paper + AutoTrader)                ║
║  Phase 4: Risk Management (Kelly/Sector/DD/News)        ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_paper_trading(cycles: int = 10, sleep: float = 2.0):
    """Runs a full paper trading session."""
    from core.data_feed import BVMTDataFeed, BVMT_TICKERS
    from strategies.engine import StrategyEngine, Direction
    from core.execution import PaperTradingEngine, AutoTrader
    from risk.risk_manager import RiskManager

    feed   = BVMTDataFeed()
    engine = StrategyEngine(feed)
    exec_e = PaperTradingEngine()
    trader = AutoTrader(exec_e, max_position_pct=0.10)
    risk   = RiskManager(initial_equity=100_000.0, feed=feed)

    tickers = BVMT_TICKERS[:8]

    log.info(f"Starting paper trading session | {len(tickers)} tickers | {cycles} cycles")
    log.info(f"Initial capital: {exec_e.cash:,.2f} TND")

    for cycle in range(1, cycles + 1):
        print(f"\n{'─'*55}")
        print(f"  Cycle {cycle}/{cycles}  |  {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'─'*55}")

        # Get current prices
        prices = {t: feed.get_quote(t)["price"] for t in tickers}

        # Check stop-loss / take-profit
        trader.check_stop_loss_take_profit(prices)

        # Run strategy scan
        signals = engine.scan_all_tickers(tickers)
        print(f"  Actionable signals: {len(signals)}")

        for sig in signals:
            price = prices.get(sig.ticker, 0)

            # Risk check before any order
            portfolio_val = exec_e.get_portfolio_summary(prices)["total_equity"]
            decision = risk.evaluate(
                ticker=sig.ticker,
                signal_confidence=sig.confidence,
                current_price=price,
                current_positions=exec_e.positions,
                portfolio_value=portfolio_val,
            )

            if decision["allow"]:
                order = trader.execute_signal(sig, price, approval_required=False)
                if order:
                    print(f"  ✓ Executed: {order}")
            else:
                print(f"  ✗ Blocked  [{sig.ticker}]: {decision['reason']}")

        # Portfolio summary
        summary = exec_e.get_portfolio_summary(prices)
        print(f"\n  Portfolio:")
        print(f"    Cash:     {summary['cash']:>10,.2f} TND")
        print(f"    Positions:{summary['positions_value']:>10,.2f} TND")
        print(f"    Equity:   {summary['total_equity']:>10,.2f} TND")
        print(f"    Unreal.:  {summary['unrealized_pnl']:>+10,.2f} TND")

        if cycle < cycles:
            time.sleep(sleep)

    # Final risk report
    summary = exec_e.get_portfolio_summary(prices)
    print(risk.daily_report(exec_e.positions, summary["total_equity"]))


def run_scan():
    """One-shot signal scan across all BVMT tickers."""
    from core.data_feed import BVMTDataFeed, BVMT_TICKERS
    from strategies.engine import StrategyEngine

    feed   = BVMTDataFeed()
    engine = StrategyEngine(feed)

    print(f"\n  Scanning {len(BVMT_TICKERS[:15])} BVMT tickers...\n")
    signals = engine.scan_all_tickers(BVMT_TICKERS[:15])

    if not signals:
        print("  No actionable signals at this time.")
        return

    print(f"  {'Ticker':<8} {'Direction':<6} {'Confidence':>10} {'Votes':>8} {'R/R':>6}")
    print(f"  {'─'*8} {'─'*6} {'─'*10} {'─'*8} {'─'*6}")
    for s in signals:
        icon = {"BUY":"▲","SELL":"▼","HOLD":"◆"}[s.direction.value]
        print(f"  {s.ticker:<8} {icon} {s.direction.value:<5} "
              f"{s.confidence:>9.0%}  "
              f"B{s.vote_buy}S{s.vote_sell}H{s.vote_hold}  "
              f"{s.risk_reward or 0:>5.2f}×")


def run_backtest():
    """Simple backtest of the composite strategy."""
    from core.data_feed import BVMTDataFeed
    from strategies.engine import StrategyEngine, Direction
    from risk.risk_manager import KellyCriterion

    feed   = BVMTDataFeed()
    engine = StrategyEngine(feed)
    kelly  = KellyCriterion()

    ticker  = "BIAT"
    df      = feed.get_ohlcv(ticker, "1Y")
    capital = 100_000.0
    trades  = []

    print(f"\n  Backtesting: {ticker} | {len(df)} trading days\n")

    window = 60
    for i in range(window, len(df)):
        slice_df = df.iloc[i-window:i]
        sig = engine.run_all(ticker, df=slice_df)

        if sig.direction == Direction.BUY and sig.confidence > 0.55:
            entry  = df["close"].iloc[i]
            future = df["close"].iloc[min(i+5, len(df)-1)]   # 5-day exit
            pnl    = (future - entry) / entry
            trades.append({"direction":"BUY","pnl_pct":pnl,"confidence":sig.confidence})

        elif sig.direction == Direction.SELL and sig.confidence > 0.55:
            entry  = df["close"].iloc[i]
            future = df["close"].iloc[min(i+5, len(df)-1)]
            pnl    = (entry - future) / entry
            trades.append({"direction":"SELL","pnl_pct":pnl,"confidence":sig.confidence})

    if not trades:
        print("  No trades generated.")
        return

    wins    = [t for t in trades if t["pnl_pct"] > 0]
    losses  = [t for t in trades if t["pnl_pct"] <= 0]
    total   = sum(t["pnl_pct"] for t in trades)
    wr      = len(wins)/len(trades)

    import numpy as np
    avg_win  = np.mean([t["pnl_pct"] for t in wins])  if wins   else 0
    avg_loss = abs(np.mean([t["pnl_pct"] for t in losses])) if losses else 0.001

    kelly_result = kelly.backtest_kelly(trades)

    print(f"  Results ({len(trades)} trades):")
    print(f"    Win rate:       {wr:.1%}")
    print(f"    Avg win:        {avg_win:.2%}")
    print(f"    Avg loss:       {avg_loss:.2%}")
    print(f"    Total return:   {total:.2%}")
    print(f"    Kelly (half):   {kelly_result.get('kelly_half',0):.1%}")
    print(f"    Reward/Risk:    {avg_win/avg_loss:.2f}×")


def run_risk_report():
    from risk.risk_manager import RiskManager

    rm = RiskManager(initial_equity=127_450.0)
    mock_positions = {
        "BIAT":    {"qty":120, "total_cost":17_424, "avg_cost":145.2},
        "SFBT":    {"qty":500, "total_cost": 9_150, "avg_cost": 18.3},
        "TPR":     {"qty":800, "total_cost": 4_720, "avg_cost":  5.9},
        "ATB":     {"qty":200, "total_cost": 7_700, "avg_cost": 38.5},
        "UIB":     {"qty":150, "total_cost": 3_270, "avg_cost": 21.8},
    }
    print(rm.daily_report(mock_positions, 127_450))


if __name__ == "__main__":
    print_banner()

    parser = argparse.ArgumentParser(description="BVMT AlgoTrader")
    parser.add_argument("--mode", choices=["paper","scan","backtest","risk"],
                        default="scan", help="Execution mode")
    parser.add_argument("--cycles", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "paper":
        run_paper_trading(cycles=args.cycles)
    elif args.mode == "scan":
        run_scan()
    elif args.mode == "backtest":
        run_backtest()
    elif args.mode == "risk":
        run_risk_report()
