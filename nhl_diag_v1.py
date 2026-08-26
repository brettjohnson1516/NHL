"""
nhl_diag_v1.py

Post-mortem on a backtest, from the bets CSV nhl_wp_backtest_v8.py already
wrote. No model rerun, no Kalshi reload.

Ported from nba_diag.py, with two views added to separate the two candidate
explanations for road favourites losing:

  1. "pregame skill is overpriced on road favs" -- if so, the error should be
     worst early, when the pregame line dominates the baseline, and should
     shrink as game state takes over. See `road fav by period` and
     `road fav by pregame strength`.
  2. "the model is biased toward road favs relative to the market" -- if so,
     model_err is large and positive while market_err sits near zero. That is
     the model being wrong, not the market.

Read model_err and market_err together:
  model_err  = mean model probability - actual win rate
  market_err = mean execution price   - actual win rate
A positive model_err with a market_err near zero means the model is the one
that is wrong.

Caveat: market_err is biased slightly positive by construction, because both
sides of a Kalshi print sum to 101c rather than 100c. Roughly half a point.

Usage (PowerShell):
  python nhl_diag_v1.py --suffix v10
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

DEFAULT_ROOT = os.environ.get(
    "NHL_ROOT", r"C:\Users\saint\OneDrive\Documents\NHL_AUG_2026\data"
)


def table(df, by, label, order=None):
    if by not in df.columns:
        print(f"\n--- {label} --- (column '{by}' not in the bets file, skipped)")
        return None
    g = df.groupby(by, dropna=False, observed=True)
    t = pd.DataFrame({
        "bets": g.size(),
        "games": g["game_id"].nunique(),
        "staked": g["stake"].sum(),
        "pnl": g["pnl"].sum(),
        "model_p": g["model_p"].mean(),
        "market_p": g["price_cents"].mean() / 100.0,
        "actual": g["won"].mean(),
    })
    t["roi_pct"] = 100.0 * t["pnl"] / t["staked"].replace(0, np.nan)
    t["model_err"] = t["model_p"] - t["actual"]
    t["market_err"] = t["market_p"] - t["actual"]
    if order is not None:
        t = t.reindex([o for o in order if o in t.index])
    print(f"\n--- {label} ---")
    print(t.reset_index().to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    return t


def concentration(df, label, top):
    per = df.groupby("game_id").agg(bets=("pnl", "size"), pnl=("pnl", "sum"),
                                    staked=("stake", "sum")).sort_values("pnl")
    print(f"\n--- {label}: concentration ---")
    print(f"  games: {len(per):,}")
    print(f"  total pnl {per['pnl'].sum():,.0f}")
    print(f"  worst {top} games contribute {per['pnl'].head(top).sum():,.0f}")
    print(f"  best  {top} games contribute {per['pnl'].tail(top).sum():,.0f}")
    print(f"  median game pnl {per['pnl'].median():,.0f}; "
          f"share of games losing {100.0 * (per['pnl'] < 0).mean():.1f}%")


ROLE_ORDER = ["home fav", "home dog", "road fav", "road dog"]
SCORE_ORDER = ["trail 4+", "trail 3", "trail 2", "trail 1", "tied",
               "lead 1", "lead 2", "lead 3", "lead 4+"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--suffix", default="v10",
                    help="the --out-suffix used on the backtest run")
    ap.add_argument("--role", default="road fav",
                    help="which role to drill into")
    ap.add_argument("--top-games", type=int, default=15)
    args = ap.parse_args()

    bdir = os.path.join(args.root, "backtests")
    f = os.path.join(bdir, f"nhl_backtest_bets_{args.suffix}.csv")
    if not os.path.exists(f):
        avail = sorted(os.path.basename(p) for p in
                       glob.glob(os.path.join(bdir, "nhl_backtest_bets_*.csv")))
        raise SystemExit(f"ERROR: {f} not found.\nAvailable: {avail}")

    b = pd.read_csv(f)
    b["game_date"] = pd.to_datetime(b["game_date"], errors="coerce")
    if "month" not in b.columns:
        b["month"] = b["game_date"].dt.strftime("%Y-%m")
    b["period_view"] = np.where(b["period"] >= 4, "OT", b["period"].astype(int).astype(str))

    # seconds left in regulation, for the decay test
    b["secs_left_reg"] = np.where(
        b["period"] <= 3,
        (3 - b["period"]) * 1200.0 + (1200.0 - b["period_seconds"]),
        0.0).clip(0, 3600)
    b["time_bucket"] = pd.cut(
        b["secs_left_reg"], [-1, 120, 300, 600, 1200, 2400, 3601],
        labels=["0-2m", "2-5m", "5-10m", "10-20m", "20-40m", "40-60m"])

    # how strong a pregame favourite/underdog was the side we bet?
    ph = b["p_home_close"].to_numpy("float64")
    side_close = np.where(b["side"].to_numpy() == "home", ph, 1.0 - ph)
    b["pregame_strength"] = pd.cut(
        side_close, [0, 0.35, 0.45, 0.5, 0.55, 0.65, 1.0],
        labels=["<35%", "35-45%", "45-50%", "50-55%", "55-65%", "65%+"])

    print(f"  {len(b):,} bets / {b['game_id'].nunique():,} games from {os.path.basename(f)}")
    print(f"  total staked {b['stake'].sum():,.0f}  pnl {b['pnl'].sum():,.0f}  "
          f"roi {100 * b['pnl'].sum() / b['stake'].sum():.2f}%")

    print("\n================ ROLE OVERVIEW ================")
    table(b, "role", "every role: is the model or the market wrong?", ROLE_ORDER)

    sel = b[b["role"] == args.role].copy()
    if sel.empty:
        raise SystemExit(f"no bets with role '{args.role}'")
    print(f"\n================ {args.role.upper()} ({len(sel):,} bets) ================")

    # --- hypothesis 1: pregame skill overpriced ---------------------------
    # If the pregame line is the culprit, the error should be biggest when the
    # baseline still leans on it (early) and should fade as the clock runs.
    table(sel, "period_view", f"{args.role} by period", ["1", "2", "3", "OT"])
    table(sel, "time_bucket", f"{args.role} by time left in regulation "
                              f"(pregame decay test)")
    table(sel, "pregame_strength", f"{args.role} by pregame strength of the "
                                   f"side bet")

    # --- hypothesis 2: general bias vs the market -------------------------
    table(sel, "price_bucket", f"{args.role} by execution price")
    table(sel, "score_bucket", f"{args.role} by score differential", SCORE_ORDER)
    table(sel, "edge_bucket", f"{args.role} by edge bucket")
    table(sel, "month", f"{args.role} by month")
    if "game_end" in sel.columns:
        table(sel, "game_end", f"{args.role} by how the game finished",
              ["regulation", "overtime", "shootout"])
    concentration(sel, args.role, args.top_games)

    print("\n================ MONTHS ================")
    table(b, "month", "all bets by month")
    worst = b.groupby("month")["pnl"].sum().idxmin()
    w = b[b["month"] == worst]
    print(f"\n================ WORST MONTH ({worst}) ================")
    table(w, "role", "worst month by role", ROLE_ORDER)
    table(w, "price_bucket", "worst month by execution price")
    table(w, "period_view", "worst month by period", ["1", "2", "3", "OT"])
    concentration(w, f"worst month {worst}", args.top_games)

    print("\n================ READING THIS ================")
    r = table(b, "role", "roles again, for reference", ROLE_ORDER)
    if r is not None and args.role in r.index:
        me = r.at[args.role, "model_err"]
        ke = r.at[args.role, "market_err"]
        print(f"\n  {args.role}: model_err {me:+.4f}, market_err {ke:+.4f}")
        if abs(me) > 2 * abs(ke):
            print("  -> the MODEL is the one that is wrong here, not the market.")
        elif abs(ke) > 2 * abs(me):
            print("  -> the MARKET is mispriced here and the model is closer.")
        else:
            print("  -> neither is clearly wrong; check the concentration numbers "
                  "for whether a few games drive this.")
        print("  If model_err shrinks across the time-left buckets as the clock "
              "runs down, the pregame line is the culprit.")
        print("  If it is flat across time, it is a general bias, not a pregame "
              "one.")


if __name__ == "__main__":
    main()
