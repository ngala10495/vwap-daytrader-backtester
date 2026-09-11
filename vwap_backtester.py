"""
VWAP Trend-Pullback Day Trading Backtester — Gold
------------------------------------------------------
This is the committed pick for a genuine day-trading strategy: VWAP is the
actual benchmark institutional desks execute against intraday on liquid
futures markets (including Gold) — not a retail indicator layered on top
of price action. This is deliberately different from ORB, SMC, CRT, and
the swing systems (MA Crossover, Turtle) built earlier in this project.

Logic:
    1. VWAP resets every session (never carried overnight — this is
       standard practice; VWAP is a same-day benchmark by definition).
    2. Day's bias: price above VWAP + rising short EMA = bullish;
       below VWAP + falling EMA = bearish. Only trade WITH this bias.
    3. Entry: price pulls back and touches/crosses VWAP, then the next
       close moves back beyond VWAP in the bias direction — this is
       "buying the dip to fair value in an uptrend" (or the mirror short).
    4. Stop: just beyond VWAP on the wrong side.
    5. Target: 2x the risk (fixed R:R, consistent and auditable).
    6. One trade per day maximum. Forced flat by session close — zero
       overnight exposure, same day-trading discipline as the ORB script.

Usage:
    pip install yfinance pandas numpy --break-system-packages
    python3 vwap_backtester.py

Requires internet access (yfinance) to pull real OHLCV data.
NOTE: 15m data is capped at 60 days by Yahoo Finance.
NOTE: this strategy needs real intraday VOLUME data (GC=F futures should
have it; if a day shows zero volume, that day is skipped as unreliable).
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import time

# ----------------------------- CONFIG ----------------------------------

TICKER = "GC=F"
PERIOD = "60d"
INTERVAL = "15m"

TREND_EMA = 20                    # short EMA used to confirm trend direction
PULLBACK_BUFFER_PCT = 0.0005       # how close to VWAP counts as a "touch"
STOP_BUFFER_PCT = 0.0008           # stop placed this far beyond VWAP
MIN_RR = 2.0                       # fixed target: 2x risk

SESSION_START = time(7, 0)         # London open — start looking for trades
SESSION_CLOSE = time(20, 0)        # force-flatten by this time, no exceptions


# --------------------------- CORE LOGIC ---------------------------------

def compute_session_vwap(day_df: pd.DataFrame) -> pd.Series:
    typical_price = (day_df["High"] + day_df["Low"] + day_df["Close"]) / 3
    cum_vol = day_df["Volume"].cumsum()
    cum_vol_price = (typical_price * day_df["Volume"]).cumsum()
    return cum_vol_price / cum_vol.replace(0, np.nan)


def run_vwap_strategy(ticker: str) -> dict:
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False)
    if df.empty:
        return {"error": "no data returned"}
    df = df.droplevel(1, axis=1) if isinstance(df.columns, pd.MultiIndex) else df
    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        return {"error": "no volume data available for this ticker — VWAP requires real volume"}

    df["ema"] = df["Close"].ewm(span=TREND_EMA, adjust=False).mean()
    df["date"] = df.index.date
    df["t"] = df.index.time

    trades = []

    for d, day_df in df.groupby("date"):
        if day_df["Volume"].sum() == 0:
            continue   # unreliable day, skip

        day_df = day_df.copy()
        day_df["vwap"] = compute_session_vwap(day_df)
        day_df["ema_slope"] = day_df["ema"].diff(5)   # precomputed, not sliced in the loop
        session = day_df[(day_df["t"] >= SESSION_START) & (day_df["t"] <= SESSION_CLOSE)]
        if len(session) < TREND_EMA:
            continue

        position = None
        already_traded_today = False
        prev_close = None
        prev_vwap = None

        for ts, bar in session.iterrows():
            vwap = bar["vwap"]
            if pd.isna(vwap):
                prev_close, prev_vwap = bar["Close"], vwap
                continue

            # --- Manage open position: stop / target / eod flatten ---
            if position is not None:
                direction = position["direction"]
                if direction == "long":
                    if bar["Low"] <= position["stop"]:
                        trades.append({**position, "exit": position["stop"], "exit_time": ts,
                                        "outcome": "loss", "reason": "stop"})
                        position = None
                    elif bar["High"] >= position["target"]:
                        trades.append({**position, "exit": position["target"], "exit_time": ts,
                                        "outcome": "win", "reason": "target"})
                        position = None
                else:
                    if bar["High"] >= position["stop"]:
                        trades.append({**position, "exit": position["stop"], "exit_time": ts,
                                        "outcome": "loss", "reason": "stop"})
                        position = None
                    elif bar["Low"] <= position["target"]:
                        trades.append({**position, "exit": position["target"], "exit_time": ts,
                                        "outcome": "win", "reason": "target"})
                        position = None

            # --- Look for a new entry (only if flat, only once per day) ---
            if position is None and not already_traded_today and prev_close is not None:
                bias_long = bar["Close"] > vwap and bar["ema_slope"] > 0
                bias_short = bar["Close"] < vwap and bar["ema_slope"] < 0

                touched_vwap = abs(prev_close - prev_vwap) / prev_vwap <= PULLBACK_BUFFER_PCT \
                    if prev_vwap and prev_vwap > 0 else False

                if touched_vwap and bias_long and bar["Close"] > vwap:
                    stop = vwap * (1 - STOP_BUFFER_PCT)
                    risk = bar["Close"] - stop
                    if risk > 0:
                        position = {"direction": "long", "entry": bar["Close"], "entry_time": ts,
                                     "stop": stop, "target": bar["Close"] + risk * MIN_RR}
                        already_traded_today = True
                elif touched_vwap and bias_short and bar["Close"] < vwap:
                    stop = vwap * (1 + STOP_BUFFER_PCT)
                    risk = stop - bar["Close"]
                    if risk > 0:
                        position = {"direction": "short", "entry": bar["Close"], "entry_time": ts,
                                     "stop": stop, "target": bar["Close"] - risk * MIN_RR}
                        already_traded_today = True

            prev_close, prev_vwap = bar["Close"], vwap

        # Force-flatten if still open at session close
        if position is not None:
            last_bar = session.iloc[-1]
            exit_price = last_bar["Close"]
            pnl_direction = 1 if position["direction"] == "long" else -1
            outcome = "win" if (exit_price - position["entry"]) * pnl_direction > 0 else "loss"
            trades.append({**position, "exit": exit_price, "exit_time": session.index[-1],
                            "outcome": outcome, "reason": "eod_flatten"})

    if not trades:
        return {"trades": 0, "win_rate": None, "total_R": 0, "trade_log": []}

    processed = []
    for t in trades:
        risk = abs(t["entry"] - t["stop"])
        pnl = (t["exit"] - t["entry"]) if t["direction"] == "long" else (t["entry"] - t["exit"])
        r_multiple = pnl / risk if risk > 0 else 0
        processed.append({
            "date": t["entry_time"].date(),
            "direction": t["direction"],
            "entry_time": t["entry_time"],
            "exit_time": t["exit_time"],
            "outcome": t["outcome"],
            "reason": t["reason"],
            "rr": round(r_multiple, 2),
        })

    wins = sum(1 for t in processed if t["outcome"] == "win")
    total_R = round(sum(t["rr"] for t in processed), 2)
    reason_counts = pd.Series([t["reason"] for t in processed]).value_counts().to_dict()

    return {
        "trades": len(processed),
        "wins": wins,
        "losses": len(processed) - wins,
        "win_rate": round(100 * wins / len(processed), 1),
        "total_R": total_R,
        "exit_reason_breakdown": reason_counts,
        "trade_log": processed,
    }


if __name__ == "__main__":
    print(f"Running VWAP Trend-Pullback day trading system on {TICKER} ({PERIOD} of {INTERVAL} data)...")
    result = run_vwap_strategy(TICKER)

    print("\n=== VWAP DAY TRADING BACKTEST RESULTS ===")
    summary = {k: v for k, v in result.items() if k != "trade_log"}
    print(summary)

    trades = result.get("trade_log", [])
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv("vwap_trade_log.csv", index=False)
        print(f"\nSaved {len(trades_df)} trades to vwap_trade_log.csv")
    else:
        print("\nNo trades to export.")
