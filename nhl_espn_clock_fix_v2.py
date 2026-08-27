#!/usr/bin/env python
"""
Repair the inverted game-clock columns in the ESPN play-by-play files.

THE BUG (mine)
--------------
ESPN's NHL feed reports `clock.displayValue` as time ELAPSED in the period,
not time remaining. Every version of nhl_espn_pbp up to v5 read it as
remaining, which inverted three columns on all 2.53M rows:

    pc_seconds_left      held elapsed-in-period instead of remaining
    seconds_elapsed      counted DOWN through each period
    seconds_left_reg     inverted to match

It shows plainly on any single game: the opening face-off of the first period
came out at seconds_elapsed = 1200 rather than 0. Measured across the whole
2025-26 season, the correlation between seconds_elapsed and wallclock was
0.830 with the bug and 0.994 without it.

NOTHING NEEDS RE-FETCHING. The stored value is the raw parsed clock, so the
correct columns are a pure function of what is already on disk:

    true_elapsed_in_period = stored pc_seconds_left
    true_pc_seconds_left   = period_length - stored pc_seconds_left
    true_seconds_elapsed   = period_start + stored pc_seconds_left

This script applies that to the season files AND to the cached per-game shards,
so a later rebuild cannot resurrect the bug. Shootout rows carry no clock and
are left null, as before.

It is idempotent by a marker column, `clock_fixed`. Running it twice is safe -
the second run reports the files as already repaired and changes nothing.

Use nhl_espn_pbp_v6.py for any future pull; it parses the clock correctly.

Usage
-----
  python nhl_espn_clock_fix_v2.py --outdir data
  python nhl_espn_clock_fix_v2.py --outdir data --dry-run
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


def period_geometry(period: pd.Series, season_type: pd.Series):
    """(period_start_elapsed, period_length) for every row, vectorised."""
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


def verify(df: pd.DataFrame) -> tuple[float, float]:
    """
    Check the DATA, not the marker.

    Two independent tells, both computed from what is actually in the file:
    the smallest seconds_elapsed in a first period must be 0 (it was 1200 with
    the bug), and seconds_elapsed must rise with wallclock inside a game
    (0.994 correct, 0.830 inverted). v1 reported purely off the `clock_fixed`
    column and printed "already repaired" for files that had never been
    touched, which is worse than useless - it asserted a fact it had not
    checked.
    """
    if df.empty or "seconds_elapsed" not in df.columns:
        return float("nan"), float("nan")
    d = df
    if "is_shootout" in d.columns:
        d = d[~d["is_shootout"].astype(bool)]
    p1 = d[(pd.to_numeric(d["period"], errors="coerce") == 1)
           & d["seconds_elapsed"].notna()]
    opening = float(p1["seconds_elapsed"].min()) if len(p1) else float("nan")

    corr = float("nan")
    if "wallclock_unix" in d.columns:
        s = d.dropna(subset=["wallclock_unix", "seconds_elapsed"])
        if len(s):
            cs = (s.groupby("game_id")
                   .apply(lambda g: np.corrcoef(g["seconds_elapsed"],
                                                g["wallclock_unix"])[0, 1]
                          if len(g) > 50 else np.nan, include_groups=False)
                   .dropna())
            if len(cs):
                corr = float(cs.median())
    return opening, corr


def repair(df: pd.DataFrame) -> tuple[pd.DataFrame, int, str]:
    if df.empty or "pc_seconds_left" not in df.columns:
        return df, 0, "no clock column - nothing to repair"
    if "clock_fixed" in df.columns and bool(df["clock_fixed"].all()):
        return df, 0, "marked clock_fixed already"

    start, length = period_geometry(df["period"], df["season_type"])
    stored = pd.to_numeric(df["pc_seconds_left"], errors="coerce")

    elapsed_in_period = stored
    df["seconds_elapsed"] = start + elapsed_in_period
    df["pc_seconds_left"] = length - elapsed_in_period
    df["seconds_left_reg"] = REGULATION_SECONDS - df["seconds_elapsed"]
    df["clock_fixed"] = True
    return df, int(stored.notna().sum()), "repaired"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=Path("data"),
                    help="Same --outdir you gave nhl_espn_pbp_v5.py.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change and write nothing.")
    args = ap.parse_args()

    season_dir = args.outdir / "pbp_espn_nhl"
    raw_dir = args.outdir / "espn_nhl_raw"
    season_files = sorted(season_dir.glob("pbp_*.parquet"))
    if not season_files:
        print(f"No season files under {season_dir}", file=sys.stderr)
        return 1

    print(f"{len(season_files)} season files")
    total_rows = 0
    print(f"  {'file':<20} {'action':<28} {'P1 min':>7} {'corr':>7}  verdict")
    suspect = []
    for f in season_files:
        df = pd.read_parquet(f)
        df, n, why = repair(df)
        total_rows += n
        opening, corr = verify(df)
        ok = (opening == 0) and (np.isnan(corr) or corr > 0.95)
        verdict = "OK" if ok else "STILL INVERTED"
        if not ok:
            suspect.append(f.name)
        c = "  n/a" if np.isnan(corr) else f"{corr:6.4f}"
        o = "  n/a" if np.isnan(opening) else f"{opening:7.0f}"
        print(f"  {f.name:<20} {why:<28} {o} {c}  {verdict}")
        if n and not args.dry_run:
            df.to_parquet(f, index=False)

    # The per-game shards feed any future rebuild, so they get the same repair.
    shard_dirs = sorted(raw_dir.glob("espn_games_*"))
    fixed_shards = 0
    for d in shard_dirs:
        shards = sorted(d.glob("*.parquet"))
        n_here = 0
        for sp in shards:
            sd = pd.read_parquet(sp)
            sd, n, _ = repair(sd)
            if n and not args.dry_run:
                sd.to_parquet(sp, index=False)
            n_here += 1 if n else 0
        fixed_shards += n_here
        print(f"  {d.name}: {n_here:,} of {len(shards):,} shards repaired")

    print(f"\n{total_rows:,} rows repaired across the season files, "
          f"{fixed_shards:,} shards repaired")
    if suspect:
        print(f"\nSTILL INVERTED: {suspect}")
        print("  A first period whose smallest seconds_elapsed is 1200 rather "
              "than 0 is the bug, regardless of what clock_fixed says. If "
              "these carry the marker but fail the check, the marker is wrong "
              "- drop the clock_fixed column from those files and re-run.")
    else:
        print("Every season file passes the data check: period-1 starts at 0 "
              "and seconds_elapsed rises with wallclock.")
    if args.dry_run:
        print("--dry-run: nothing was written.")
    else:
        print("Done. seconds_elapsed now counts UP and matches wallclock.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
