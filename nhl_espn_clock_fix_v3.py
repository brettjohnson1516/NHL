#!/usr/bin/env python
"""
nhl_espn_clock_fix_v3.py

Repair the inverted ESPN game clock, deciding PER FILE from the data.

WHY v2 IS NOT SAFE HERE
-----------------------
Two problems show up once nhl_espn_pbp_v6.py has been run over a cache built
by v5:

1. v6 reuses cached shards. `fetch_game` returns early if the shard exists, so
   v6's corrected clock parsing never touches a game v5 already fetched. The
   cache ends up MIXED: v5 games inverted, v6 games correct, and neither
   carries a marker because v6 does not write `clock_fixed` either.

2. v2 repairs anything without that marker. On a mixed cache it therefore
   inverts the correct v6 games while fixing the v5 ones.

v2's verification is also wrong. It checks that the smallest seconds_elapsed in
period 1 is 0. Under the bug the clock counts DOWN from 1200 to 0 across the
period, so the minimum is still ~0 -- at the last event rather than the first.
That test passes on inverted data.

THE TEST THAT ACTUALLY WORKS
----------------------------
Within one game, seconds_elapsed must RISE with event order. Correct files
correlate near +1, inverted files near -1. It needs no wallclock, no marker,
and no assumption about which fetcher wrote the file.

Repairs are applied per game, so a mixed shard directory comes out uniformly
correct. Files already correct are left untouched.

Usage
-----
  python nhl_espn_clock_fix_v3.py --outdir data --dry-run
  python nhl_espn_clock_fix_v3.py --outdir data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REGULATION_PERIODS = 3
PERIOD_SECONDS = 1200
REG_OT_SECONDS = 300
PLAYOFF_OT_SECONDS = 1200
REGULATION_SECONDS = REGULATION_PERIODS * PERIOD_SECONDS
SEASON_TYPE_POST = 3


def period_geometry(period, season_type):
    p = pd.to_numeric(period, errors="coerce")
    post = pd.to_numeric(season_type, errors="coerce") == SEASON_TYPE_POST
    length = np.where(p <= REGULATION_PERIODS, PERIOD_SECONDS,
                      np.where(post, PLAYOFF_OT_SECONDS,
                               np.where(p == 4, REG_OT_SECONDS, np.nan)))
    start = np.where(p <= REGULATION_PERIODS, (p - 1) * PERIOD_SECONDS,
                     np.where(post,
                              REGULATION_SECONDS + (p - 4) * PLAYOFF_OT_SECONDS,
                              np.where(p == 4, REGULATION_SECONDS, np.nan)))
    return start, length


def direction(df):
    """Median per-game correlation of seconds_elapsed with event order.

    Near +1 the clock counts up and the file is correct. Near -1 it counts
    down and the file is inverted. NaN when there is nothing to judge on.
    """
    if df.empty or "seconds_elapsed" not in df.columns:
        return float("nan")
    d = df
    if "is_shootout" in d.columns:
        d = d[~d["is_shootout"].astype(bool)]
    order_col = None
    for c in ("sequence_number", "play_id", "event_idx", "sort_order"):
        if c in d.columns:
            order_col = c
            break
    d = d.dropna(subset=["seconds_elapsed"])
    if d.empty or "game_id" not in d.columns:
        return float("nan")

    cors = []
    for _, g in d.groupby("game_id"):
        if len(g) < 20:
            continue
        y = pd.to_numeric(g["seconds_elapsed"], errors="coerce").to_numpy()
        if order_col is not None:
            x = pd.to_numeric(g[order_col], errors="coerce").to_numpy()
            if not np.isfinite(x).all():
                x = np.arange(len(g), dtype="float64")
        else:
            x = np.arange(len(g), dtype="float64")
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 20 or np.std(y[ok]) == 0 or np.std(x[ok]) == 0:
            continue
        cors.append(np.corrcoef(x[ok], y[ok])[0, 1])
    return float(np.median(cors)) if cors else float("nan")


def invert(df):
    """Apply the correction to every row. Caller decides whether to."""
    start, length = period_geometry(df["period"], df["season_type"])
    stored = pd.to_numeric(df["pc_seconds_left"], errors="coerce")
    df["seconds_elapsed"] = start + stored
    df["pc_seconds_left"] = length - stored
    df["seconds_left_reg"] = REGULATION_SECONDS - df["seconds_elapsed"]
    df["clock_fixed"] = True
    return df, int(stored.notna().sum())


def repair_by_game(df, threshold):
    """Invert only the games whose clock runs backwards."""
    if df.empty or "pc_seconds_left" not in df.columns:
        return df, 0, 0, 0
    if "game_id" not in df.columns:
        d = direction(df)
        if np.isfinite(d) and d < threshold:
            df, n = invert(df)
            return df, n, 1, 0
        return df, 0, 0, 1

    parts, rows, bad, good = [], 0, 0, 0
    for _, g in df.groupby("game_id", sort=False):
        d = direction(g)
        if np.isfinite(d) and d < threshold:
            g, n = invert(g.copy())
            rows += n
            bad += 1
        else:
            good += 1
        parts.append(g)
    return pd.concat(parts).sort_index(), rows, bad, good


def process(path, threshold, dry):
    df = pd.read_parquet(path)
    before = direction(df)
    df, rows, bad, good = repair_by_game(df, threshold)
    after = direction(df)
    if rows and not dry:
        df.to_parquet(path, index=False)
    return before, after, rows, bad, good


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="a game is inverted when its order/clock correlation "
                         "is below this. Correct games sit near +1, inverted "
                         "near -1, so 0 separates them cleanly.")
    ap.add_argument("--skip-shards", action="store_true",
                    help="season files only; leave the per-game cache alone")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    season_dir = args.outdir / "pbp_espn_nhl"
    raw_dir = args.outdir / "espn_nhl_raw"
    season_files = sorted(season_dir.glob("pbp_*.parquet"))
    if not season_files:
        print(f"No season files under {season_dir}", file=sys.stderr)
        return 1

    print(f"{len(season_files)} season files "
          f"(corr near +1 = clock counts up = correct)")
    print(f"  {'file':<20} {'corr before':>12} {'corr after':>11} "
          f"{'games fixed':>12} {'already ok':>11}")
    tot_rows = tot_bad = 0
    for f in season_files:
        b, a, rows, bad, good = process(f, args.threshold, args.dry_run)
        tot_rows += rows
        tot_bad += bad
        bs = "  n/a" if np.isnan(b) else f"{b:12.4f}"
        as_ = "  n/a" if np.isnan(a) else f"{a:11.4f}"
        print(f"  {f.name:<20} {bs} {as_} {bad:>12,} {good:>11,}")

    if not args.skip_shards:
        print("\nper-game shards (these feed any later rebuild):")
        for d in sorted(raw_dir.glob("espn_games_*")):
            shards = sorted(d.glob("*.parquet"))
            bad = good = 0
            for sp in shards:
                try:
                    _, _, rows, b, g = process(sp, args.threshold, args.dry_run)
                except Exception as e:
                    print(f"    {sp.name}: {type(e).__name__} {e}")
                    continue
                tot_rows += rows
                bad += b
                good += g
            print(f"  {d.name}: {bad:,} inverted and fixed, {good:,} already ok, "
                  f"{len(shards):,} shards")
            tot_bad += bad

    print(f"\n{tot_rows:,} rows corrected across {tot_bad:,} games")
    if args.dry_run:
        print("--dry-run: nothing written.")
    else:
        print("Done. Re-run nhl_pbp_merge_v1.py -- the merge reads these files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
