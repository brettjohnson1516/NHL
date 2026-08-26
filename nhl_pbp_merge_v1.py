#!/usr/bin/env python
"""
Put ESPN's wall clock onto the NHL feed's play-by-play, so one table has both
the UTC timestamp and the on-ice data.

WHY NOT JUST SWITCH SOURCES
---------------------------
There is no free NHL play-by-play feed that carries both. I looked:

  * api-web.nhle.com/v1/gamecenter/{id}/play-by-play - the current official
    feed, and the one `nhl_pbp_full` is built from. Its plays carry
    timeInPeriod, secondsRemaining, situationCode and sortOrder. No wall-clock
    timestamp of any kind.
  * statsapi.web.nhl.com - the OLD official feed - did carry `about.dateTime`
    per play, which was a real UTC wall clock. It was retired and the host no
    longer resolves.
  * ESPN's summary endpoint carries `wallclock` on every play, and no on-ice
    skaters, no line changes and no time-on-ice.

So the two feeds are complementary and neither is redundant. This script joins
them instead of choosing.

HOW THE JOIN WORKS
------------------
Games first. The ESPN table and the NHL table are matched on the team pair
plus the date. ESPN's game_date is the UTC date of puck drop, so a 7pm ET game
lands on the FOLLOWING calendar day - matching on the date alone found only
469 of 1,384 games in 2025-26. Allowing the NHL local date to be the ESPN date
or the day before matched all 1,384, with no NHL game claimed twice.

Events second. Both feeds carry period and seconds-elapsed for the same
official scoresheet, so events are anchored on (period, event type, second)
within a tolerance. Anchors are then used to interpolate a wall clock onto
EVERY NHL row - including the CHANGE events ESPN does not have at all, which
is the whole point, since those are what carry the on-ice sets.

Interpolation is monotone by construction: anchors are sorted by game time,
forced non-decreasing in wall clock, deduplicated, and interpolated linearly
between. Rows outside the anchor range are extrapolated from the nearest pair
and flagged `wallclock_extrapolated` rather than silently invented.

REQUIRES THE CLOCK FIX
----------------------
ESPN reports its NHL clock as time ELAPSED in the period, and nhl_espn_pbp up
to v5 read it as remaining. With that bug the anchors line up backwards and
this join produces garbage. Run nhl_espn_clock_fix_v1.py first - this script
checks for its `clock_fixed` marker and refuses to run without it.

INPUTS   <outdir>/pbp_espn_nhl/pbp_{season}.parquet    (nhl_espn_pbp_v6)
         <outdir>/nhl_pbp_raw/play_by_play_{season}.parquet  (nhl_ytd_stats_v3)
         <outdir>/nhl_schedules_raw/nhl_schedule_{season}.parquet (fetched here)

OUTPUT   <outdir>/pbp_merged/pbp_{season}.parquet
         the full NHL feed, plus:
           espn_game_id            the ESPN id, so this joins to your odds
                                   and Kalshi tables
           wallclock_unix          UTC epoch seconds
           wallclock_extrapolated  outside the anchor range
           n_anchors               anchors behind this game's interpolation
         <outdir>/pbp_merged/game_crosswalk.parquet
           nhl_game_id <-> espn_game_id <-> date and teams

Seasons before 2021 have no ESPN file and are skipped; the NHL feed still
stands alone for those.

Usage
-----
  python nhl_pbp_merge_v1.py --seasons 2021 2026 --outdir data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

SCHED_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases"
             "/download/nhl_schedules/nhl_schedule_{season}.parquet")

# ESPN abbreviation -> NHL feed abbreviation. Only the ones that differ.
ESPN_TO_NHL = {"TB": "TBL", "LA": "LAK", "NJ": "NJD", "SJ": "SJS",
               "UTAH": "UTA", "WAS": "WSH", "MON": "MTL", "CLS": "CBJ",
               "VEG": "VGK", "WIN": "WPG", "CAL": "CGY", "ANH": "ANA",
               "NAS": "NSH", "ARI": "ARI"}

# ESPN type_text -> NHL event_type. Only types precise enough to anchor on.
TYPE_MAP = {
    "Goal": "GOAL", "Shot": "SHOT", "Missed": "MISSED_SHOT",
    "Blocked": "BLOCKED_SHOT", "Hit": "HIT", "Face Off": "FACEOFF",
    "Giveaway": "GIVEAWAY", "Takeaway": "TAKEAWAY", "Stoppage": "STOP",
    "Period Start": "PERIOD_START", "Period End": "PERIOD_END",
    "End of Game": "GAME_END",
}


def nhl_abbr(code) -> Optional[str]:
    if not isinstance(code, str) or not code:
        return None
    c = code.upper()
    return ESPN_TO_NHL.get(c, c)


def get_schedule(season: int, raw_dir: Path) -> Optional[pd.DataFrame]:
    """NHL game_id with its LOCAL game date - the bridge to the ESPN date."""
    path = raw_dir / f"nhl_schedule_{season}.parquet"
    if not path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        r = requests.get(SCHED_URL.format(season=season), timeout=180)
        if r.status_code != 200:
            print(f"  schedule {season}: HTTP {r.status_code}")
            return None
        path.write_bytes(r.content)
    df = pd.read_parquet(path)
    cols = {c.lower(): c for c in df.columns}
    need = ("game_id", "game_date", "home_team_abbr", "away_team_abbr")
    if not all(n in cols for n in need):
        print(f"  schedule {season}: unexpected columns {list(df.columns)[:12]}")
        return None
    out = df[[cols[n] for n in need]].copy()
    out.columns = ["nhl_game_id", "game_date", "home_abbr", "away_abbr"]
    out["nhl_game_id"] = pd.to_numeric(out["nhl_game_id"],
                                       errors="coerce").astype("Int64")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    return out.dropna(subset=["nhl_game_id", "game_date"])


def crosswalk(espn: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """
    Match ESPN games to NHL games on teams plus a date that may be off by one.

    ESPN stamps a game with the UTC date of puck drop, so a night game in North
    America is filed on the next calendar day. Trying same-day only matched 469
    of 1,384 games in 2025-26; allowing same-day OR one day earlier matched all
    of them, and no NHL game was claimed twice.
    """
    e = espn.drop_duplicates("game_id")[["game_id", "game_date", "home_team",
                                         "away_team"]].copy()
    e["h"] = e["home_team"].map(nhl_abbr)
    e["a"] = e["away_team"].map(nhl_abbr)
    e["d"] = pd.to_datetime(e["game_date"], errors="coerce")
    e = e.dropna(subset=["h", "a", "d"])

    s = sched.rename(columns={"home_abbr": "h", "away_abbr": "a",
                              "game_date": "d"})
    hits = []
    for shift in (0, -1, 1):
        t = e.copy()
        t["d"] = t["d"] + pd.Timedelta(days=shift)
        m = t.merge(s, on=["d", "h", "a"], how="inner")
        m["date_shift"] = shift
        hits.append(m)
    out = pd.concat(hits, ignore_index=True)
    # Prefer the smallest shift for any ESPN game matched more than once.
    out["absshift"] = out["date_shift"].abs()
    out = (out.sort_values("absshift")
              .drop_duplicates("game_id", keep="first")
              .drop_duplicates("nhl_game_id", keep="first"))
    return out.rename(columns={"game_id": "espn_game_id", "d": "nhl_date"})[
        ["espn_game_id", "nhl_game_id", "nhl_date", "h", "a", "date_shift"]]


def anchors_for_game(e: pd.DataFrame, n: pd.DataFrame,
                     tol: float) -> pd.DataFrame:
    """
    Anchor pairs (game_seconds, wallclock) for one game.

    Matching is on period and event type, then nearest second within `tol`.
    Only types that exist in both feeds and mean the same thing are used - a
    generic ESPN "Penalty" covers a dozen NHL sub-types and would anchor to the
    wrong second, so penalties are left out entirely. Shots, faceoffs, hits and
    stoppages are plentiful enough on their own.
    """
    rows = []
    e = e.dropna(subset=["seconds_elapsed", "wallclock_unix", "etype"])
    for (period, etype), grp in n.groupby(["period", "event_type"], sort=False):
        cand = e[(e["period"] == period) & (e["etype"] == etype)]
        if cand.empty:
            continue
        cs = cand["seconds_elapsed"].to_numpy()
        cw = cand["wallclock_unix"].to_numpy()
        order = np.argsort(cs)
        cs, cw = cs[order], cw[order]
        for gs in grp["game_seconds"].to_numpy():
            i = int(np.abs(cs - gs).argmin())
            if abs(cs[i] - gs) <= tol:
                rows.append((gs, cw[i]))
    if not rows:
        return pd.DataFrame(columns=["gs", "wc"])
    a = pd.DataFrame(rows, columns=["gs", "wc"]).groupby("gs", as_index=False)["wc"].median()
    a = a.sort_values("gs")
    # A wall clock cannot run backwards. Forcing the anchors non-decreasing
    # removes the handful of mismatched pairs without discarding the game.
    a["wc"] = np.maximum.accumulate(a["wc"].to_numpy())
    return a


def merge_season(season: int, outdir: Path, tol: float,
                 report: dict) -> Optional[pd.DataFrame]:
    espn_path = outdir / "pbp_espn_nhl" / f"pbp_{season}.parquet"
    nhl_path = outdir / "nhl_pbp_raw" / f"play_by_play_{season}.parquet"
    if not espn_path.exists():
        print(f"  no ESPN file for {season} - skipped")
        return None
    if not nhl_path.exists():
        print(f"  no NHL file for {season} - run nhl_ytd_stats_v3.py first")
        return None

    espn = pd.read_parquet(espn_path, columns=[
        "game_id", "game_date", "home_team", "away_team", "period",
        "seconds_elapsed", "wallclock_unix", "type_text", "is_shootout",
        "clock_fixed"] if "clock_fixed" in pd.read_parquet(
            espn_path, columns=[]).columns else None)
    if "clock_fixed" not in espn.columns or not bool(espn["clock_fixed"].all()):
        print(f"\n  {espn_path.name} has not been through "
              f"nhl_espn_clock_fix_v1.py.")
        print("  Its seconds_elapsed counts DOWN, the anchors would line up "
              "backwards, and every wall clock this produced would be wrong. "
              "Refusing to merge.")
        report["unfixed"] += 1
        return None

    espn = espn[~espn["is_shootout"].astype(bool)].copy()
    espn["etype"] = espn["type_text"].map(TYPE_MAP)
    espn["game_id"] = espn["game_id"].astype(str)

    sched = get_schedule(season, outdir / "nhl_schedules_raw")
    if sched is None:
        return None
    xw = crosswalk(espn, sched)
    n_espn = espn["game_id"].nunique()
    print(f"  crosswalk: {len(xw):,} of {n_espn:,} ESPN games matched "
          f"({(xw['date_shift'] == -1).sum():,} needed the day-before shift)")
    report["games_matched"] += len(xw)
    report["games_espn"] += n_espn

    nhl = pd.read_parquet(nhl_path)
    nhl = nhl.dropna(subset=["game_id"])
    nhl["game_id"] = pd.to_numeric(nhl["game_id"], errors="coerce").astype("Int64")
    nhl = nhl.merge(xw[["espn_game_id", "nhl_game_id"]],
                    left_on="game_id", right_on="nhl_game_id", how="left")
    nhl = nhl.drop(columns=["nhl_game_id"])

    espn_by_game = {g: d for g, d in espn.groupby("game_id", sort=False)}

    wc = np.full(len(nhl), np.nan)
    extrap = np.zeros(len(nhl), dtype=bool)
    nanch = np.zeros(len(nhl), dtype="int32")
    pos = {gid: idx for gid, idx in nhl.groupby("game_id", sort=False).indices.items()}

    done = 0
    for gid, idx in pos.items():
        sub = nhl.iloc[idx]
        eid = sub["espn_game_id"].iloc[0]
        if not isinstance(eid, str) or eid not in espn_by_game:
            continue
        a = anchors_for_game(espn_by_game[eid], sub, tol)
        if len(a) < 3:
            report["too_few_anchors"] += 1
            continue
        gs = sub["game_seconds"].to_numpy(dtype="float64")
        w = np.interp(gs, a["gs"].to_numpy(), a["wc"].to_numpy())
        wc[idx] = w
        extrap[idx] = (gs < a["gs"].iloc[0]) | (gs > a["gs"].iloc[-1])
        nanch[idx] = len(a)
        done += 1
        if done % 200 == 0:
            print(f"    {done:,} games interpolated")

    nhl["wallclock_unix"] = wc
    nhl["wallclock_extrapolated"] = extrap
    nhl["n_anchors"] = nanch
    report["games_merged"] += done
    return nhl


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    default=[2021, 2026],
                    help="ESPN season ENDING year. ESPN coverage here starts "
                         "at 2021.")
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="Seconds of game clock allowed between two feeds' "
                         "versions of the same event. Default 2.")
    args = ap.parse_args()

    out_dir = args.outdir / "pbp_merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"games_espn": 0, "games_matched": 0, "games_merged": 0,
              "too_few_anchors": 0, "unfixed": 0}

    first, last = args.seasons
    xws, summary = [], []
    for season in range(first, last + 1):
        print(f"\n=== season {season} ===")
        merged = merge_season(season, args.outdir, args.tolerance, report)
        if merged is None:
            continue
        path = out_dir / f"pbp_{season}.parquet"
        merged.to_parquet(path, index=False)

        have = merged["wallclock_unix"].notna()
        ex = merged["wallclock_extrapolated"] & have
        summary.append((season, len(merged), int(have.sum()), int(ex.sum())))
        print(f"  wrote {path.name}: {len(merged):,} rows, "
              f"{have.mean():.1%} with a wall clock, "
              f"{ex.sum() / max(int(have.sum()), 1):.1%} of those extrapolated")

        xw = merged[["game_id", "espn_game_id"]].dropna().drop_duplicates()
        xw["season"] = season
        xws.append(xw)

    if report["unfixed"]:
        print(f"\n{report['unfixed']} season(s) refused. Run:")
        print(f"  python nhl_espn_clock_fix_v1.py --outdir {args.outdir}")
        return 1
    if not summary:
        print("Nothing merged.", file=sys.stderr)
        return 1

    if xws:
        cw = pd.concat(xws, ignore_index=True)
        cw.to_parquet(out_dir / "game_crosswalk.parquet", index=False)
        print(f"\nwrote game_crosswalk.parquet: {len(cw):,} games "
              f"(nhl_game_id <-> espn_game_id)")

    print("\nPer season:")
    print(f"  {'season':>7} {'rows':>11} {'with wallclock':>15} {'extrapolated':>13}")
    for s, r, h, e in summary:
        print(f"  {s:>7} {r:>11,} {h:>15,} {e:>13,}")

    print("\nCoverage:")
    print(f"  ESPN games seen            : {report['games_espn']:,}")
    print(f"  matched to an NHL game     : {report['games_matched']:,}")
    print(f"  interpolated onto NHL rows : {report['games_merged']:,}")
    print(f"  games with under 3 anchors : {report['too_few_anchors']:,}")

    # The one check that matters: a wall clock that disagrees with the game
    # clock is worse than no wall clock at all.
    last_path = out_dir / f"pbp_{summary[-1][0]}.parquet"
    d = pd.read_parquet(last_path, columns=["game_id", "game_seconds",
                                            "wallclock_unix"]).dropna()
    if len(d):
        c = (d.groupby("game_id")
               .apply(lambda g: np.corrcoef(g["game_seconds"],
                                            g["wallclock_unix"])[0, 1]
                      if len(g) > 50 else np.nan, include_groups=False)
               .dropna())
        print(f"\nSanity on {last_path.name} - correlation between game_seconds "
              f"and the interpolated wallclock:")
        print(f"  median {c.median():.4f}   5th pct {c.quantile(0.05):.4f}   "
              f"min {c.min():.4f}")
        print("  Anything well below 0.99 means the anchors for that game did "
              "not line up; check n_anchors on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
