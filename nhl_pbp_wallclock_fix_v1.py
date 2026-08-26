#!/usr/bin/env python
"""
nhl_pbp_wallclock_fix_v1.py

Rebuild `wallclock_unix` on the merged pbp by matching each NHL row to its ESPN
row directly, instead of interpolating against the game clock.

THE BUG
-------
nhl_pbp_merge_v1 builds anchor pairs (game_seconds -> wallclock), collapses
them with `groupby("gs").median()`, and interpolates every NHL row against the
GAME CLOCK.

Two things break:

  * The game clock freezes on a whistle, so many wall-clock seconds map to one
    game second. A goal and the faceoff that follows it share a game second but
    sit ~25 wall seconds apart. The median blends them and drags the goal late.
  * `np.maximum.accumulate` then forces the blended anchors non-decreasing,
    which pushes neighbours later still.

Measured on 2025-26: the merged goal timestamp runs a median of 23s LATE
against ESPN's own wallclock, on 93.6% of goals. Anchoring the Kalshi price
curve on the merged timestamp shows 70% of the price move already done before
t=0; anchoring on ESPN's raw wallclock shows 2%. The market moves when the goal
happens -- our timestamp was simply late.

THE FIX
-------
Match NHL rows to ESPN rows one-to-one within (game, period, event type), in
game-clock order, and take ESPN's wallclock directly. Rows that do not match
are interpolated against EVENT ORDER, which -- unlike the game clock -- is
strictly increasing in wall time and does not freeze on a whistle.

Idempotent by a `wallclock_source` column. Writes a .bak beside each file the
first time it touches it.

Usage
-----
  python nhl_pbp_wallclock_fix_v1.py --root <NHL_AUG_2026\\data> --espn-root <data>
  python nhl_pbp_wallclock_fix_v1.py --root ... --espn-root ... --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

TYPE_MAP = {
    "Goal": "GOAL", "Shot": "SHOT", "Missed": "MISSED_SHOT",
    "Blocked": "BLOCKED_SHOT", "Hit": "HIT", "Face Off": "FACEOFF",
    "Giveaway": "GIVEAWAY", "Takeaway": "TAKEAWAY", "Stoppage": "STOP",
    "Period Start": "PERIOD_START", "Period End": "PERIOD_END",
    "End of Game": "GAME_END",
}


def one_to_one(nhl_gs, espn_gs, tol):
    """Greedy nearest match in game-clock order, each ESPN row used once."""
    order_n = np.argsort(nhl_gs, kind="stable")
    order_e = np.argsort(espn_gs, kind="stable")
    out = np.full(len(nhl_gs), -1, dtype="int64")
    i = j = 0
    while i < len(order_n) and j < len(order_e):
        a, b = order_n[i], order_e[j]
        d = espn_gs[b] - nhl_gs[a]
        if abs(d) <= tol:
            out[a] = b
            i += 1
            j += 1
        elif d < 0:
            j += 1
        else:
            i += 1
    return out


def fix_game(nhl_g, espn_g, tol):
    """Wallclock per NHL row: matched ESPN value, else interpolated on order."""
    n = len(nhl_g)
    wc = np.full(n, np.nan)
    src = np.full(n, "interp", dtype=object)

    gs = nhl_g["game_seconds"].to_numpy("float64")
    et = nhl_g["event_type"].to_numpy(object)
    e_gs = espn_g["seconds_elapsed"].to_numpy("float64")
    e_wc = espn_g["wallclock_unix"].to_numpy("float64")
    e_et = espn_g["etype"].to_numpy(object)

    for t in pd.unique(et):
        ni = np.flatnonzero(et == t)
        ei = np.flatnonzero(e_et == t)
        if len(ni) == 0 or len(ei) == 0:
            continue
        m = one_to_one(gs[ni], e_gs[ei], tol)
        hit = m >= 0
        wc[ni[hit]] = e_wc[ei[m[hit]]]
        src[ni[hit]] = "espn"

    # A wall clock cannot run backwards. Enforce it on EVENT ORDER, which is
    # the ordering that is actually monotone in wall time.
    known = np.flatnonzero(np.isfinite(wc))
    if len(known) >= 2:
        wc[known] = np.maximum.accumulate(wc[known])
        idx = np.arange(n)
        miss = ~np.isfinite(wc)
        wc[miss] = np.interp(idx[miss], known, wc[known])
        src[miss & (idx < known[0])] = "extrap"
        src[miss & (idx > known[-1])] = "extrap"
    return wc, src, int(np.isfinite(wc).sum()), len(known)


def load_espn(espn_root, season):
    fp = Path(espn_root) / "pbp_espn_nhl" / f"pbp_{season}.parquet"
    if not fp.exists():
        return None
    d = pd.read_parquet(fp, columns=["game_id", "period", "seconds_elapsed",
                                     "wallclock_unix", "type_text",
                                     "is_shootout"])
    d = d[~d["is_shootout"].astype(bool)]
    d = d.dropna(subset=["seconds_elapsed", "wallclock_unix"])
    d["etype"] = d["type_text"].map(TYPE_MAP)
    d = d[d["etype"].notna()].copy()
    d["game_id"] = d["game_id"].astype(str)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="the data root holding pbp_merged")
    ap.add_argument("--espn-root", required=True,
                    help="the --outdir given to the ESPN fetcher (holds "
                         "pbp_espn_nhl)")
    ap.add_argument("--seasons", nargs="+", type=int,
                    default=[2021, 2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--tol", type=float, default=4.0,
                    help="seconds of game clock allowed when matching a row")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pbp_dir = Path(args.root) / "pbp_merged"
    if not pbp_dir.is_dir():
        raise SystemExit(f"ERROR: {pbp_dir} not found")

    for season in args.seasons:
        fp = pbp_dir / f"pbp_{season}.parquet"
        if not fp.exists():
            print(f"[{season}] no merged file, skipped")
            continue
        nhl = pd.read_parquet(fp)
        if "wallclock_source" in nhl.columns:
            print(f"[{season}] already rebuilt, skipped")
            continue
        espn = load_espn(args.espn_root, season)
        if espn is None:
            print(f"[{season}] no ESPN file, skipped")
            continue

        nhl["game_id"] = nhl["game_id"].astype("int64")
        nhl["espn_game_id"] = nhl["espn_game_id"].astype(str)
        nhl = nhl.sort_values(["game_id", "event_idx"], kind="stable").reset_index(drop=True)

        old = nhl["wallclock_unix"].to_numpy("float64").copy()
        new = np.full(len(nhl), np.nan)
        src = np.full(len(nhl), "none", dtype=object)

        espn_by = {g: d for g, d in espn.groupby("game_id", sort=False)}
        pos = nhl.groupby("game_id", sort=False).indices
        matched = anchors = games = 0
        for gid, idx in pos.items():
            sub = nhl.iloc[idx]
            eid = sub["espn_game_id"].iloc[0]
            eg = espn_by.get(eid)
            if eg is None or len(eg) < 3:
                continue
            w, s, nfin, nanch = fix_game(sub, eg, args.tol)
            new[idx] = w
            src[idx] = s
            matched += int((s == "espn").sum())
            anchors += nanch
            games += 1

        have = np.isfinite(new)
        delta = new[have & np.isfinite(old)] - old[have & np.isfinite(old)]
        goal = (nhl["event_type"].to_numpy(object) == "GOAL") & have & np.isfinite(old)
        gd = new[goal] - old[goal]
        print(f"[{season}] {games:,} games | rows with a wallclock "
              f"{have.sum():,}/{len(nhl):,} | direct ESPN matches {matched:,} "
              f"({matched / max(have.sum(), 1):.1%})")
        print(f"          shift new-minus-old: median {np.median(delta):+.1f}s, "
              f"mean {np.mean(delta):+.1f}s")
        if goal.sum():
            print(f"          on GOAL rows only : median {np.median(gd):+.1f}s "
                  f"over {goal.sum():,} goals")

        nhl["wallclock_unix"] = new
        nhl["wallclock_source"] = src
        nhl["wallclock_extrapolated"] = (src == "extrap")
        if not args.dry_run:
            bak = fp.with_suffix(".parquet.bak")
            if not bak.exists():
                shutil.copy2(fp, bak)
            nhl.to_parquet(fp, index=False)

    if args.dry_run:
        print("\n--dry-run: nothing written.")
    else:
        print("\nDone. Rebuild features, then re-run the backtest.")


if __name__ == "__main__":
    main()
