#!/usr/bin/env python
"""
Stage 3 (NHL) - pull Pinnacle CLOSING MONEYLINES and GAME TOTALS from The Odds
API historical snapshots, ESPN seasons 2021-2026.

This is your two NBA scripts merged into one pass. The NBA build fetched h2h in
nba_pinnacle_odds.py and then totals in nba_fetch_totals.py, which meant two
runs over the same snapshot grid. Here `markets=h2h,totals` rides one request,
so the moneyline and the total are guaranteed to come from THE SAME snapshot -
no chance of the total being priced at a different moment than the close it is
supposed to sit next to. Two parquet files come out, matching your NBA layout:

    <outdir>/odds/closing_lines_pinnacle.parquet     moneyline
    <outdir>/odds/closing_totals_pinnacle.parquet    over/under

THE GAME INDEX COMES FROM YOUR OWN NHL DATA
-------------------------------------------
There is no espn_nhl_schedules release to read tip-off times from - the same
gap that forced nhl_espn_pbp_v5.py to scrape. So the game index is built from
what that script already wrote and cached:

    <outdir>/pbp_espn_nhl/pbp_{season}.parquet   game_id, teams, season_type
    <outdir>/espn_nhl_raw/espn_gameids_{season}.json   the ISO tip-off time

Both are already on disk. No extra ESPN calls, and espn_game_id stays the join
key, so this drops straight onto the pbp table.

Run nhl_espn_pbp_v5.py first. If either file is missing for a season, that
season is reported and skipped rather than silently half-loaded.

TEAM ABBREVIATIONS ARE RESOLVED, NOT ASSUMED
--------------------------------------------
The Odds API uses full names; your pbp stores ESPN abbreviations, and I have
not seen your files, so I am not going to hardcode a guess at whether ESPN
writes LA or LAK, TB or TBL, UTA or UTAH. Each franchise below carries the
plausible codes, and the script picks whichever one ACTUALLY APPEARS in your
pbp data. Anything it cannot resolve is printed with a count, never dropped in
silence. Franchises that changed name inside the window - Arizona -> Utah
Hockey Club -> Utah Mammoth - carry every name the API may have used.

THE SNAPSHOT GRID
-----------------
Same as the NBA script: historical odds are periodic snapshots, not a feed, and
a request returns the closest snapshot AT OR BEFORE the timestamp asked for.
A true one-minute-before-puck-drop price does not exist. The grid spacing and
its phase are MEASURED per season from `previous_timestamp` / `next_timestamp`
rather than assumed, because an assumed :00/:10 grid asks for timestamps that
never existed and quietly lands you a full step staler than necessary.
Per-game staleness is recorded in `snapshot_lag_seconds`.

QUOTA - READ BEFORE RUNNING
---------------------------
Historical requests are metered per markets x regions. Two markets on one
region costs twice what one market costs, so this run is the same total spend
as fetching h2h and totals separately - it just does it in one pass.

  * `--dry-run` plans everything and prints the request count per season
    without spending anything.
  * `--probe` makes exactly ONE request, dumps the JSON shape and the quota
    headers, and stops.
  * `--max-requests N` is a hard stop.
  * Remaining and used quota print as the run goes.

Regions defaults to `eu` because that is where Pinnacle sits. The NBA moneyline
script asked for `us,eu`, which doubles the meter for a book that is only in
one of them.

Every snapshot is cached to disk as raw JSON, keyed by markets+regions, so a
re-run costs nothing and an interrupted run resumes. Both output files are
MERGED with what is already there, so running one season does not wipe another.

Setup
-----
  pip install pandas pyarrow requests
  $env:ODDS_API_KEY = "<your-key>"

Usage
-----
  python nhl_pinnacle_odds_v2.py --seasons 2021 2026 --outdir data --dry-run
  python nhl_pinnacle_odds_v2.py --seasons 2021 2026 --outdir data --probe
  python nhl_pinnacle_odds_v2.py --seasons 2021 2026 --outdir data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "icehockey_nhl"

# The Odds API full name -> the ESPN abbreviations it might appear as in your
# pbp files. The script picks whichever candidate is actually present.
TEAM_CANDIDATES = {
    "Anaheim Ducks": ("ANA",),
    "Arizona Coyotes": ("ARI", "ARZ", "PHX"),
    "Boston Bruins": ("BOS",),
    "Buffalo Sabres": ("BUF",),
    "Calgary Flames": ("CGY", "CAL"),
    "Carolina Hurricanes": ("CAR",),
    "Chicago Blackhawks": ("CHI",),
    "Colorado Avalanche": ("COL",),
    "Columbus Blue Jackets": ("CBJ", "CLS", "CLB"),
    "Dallas Stars": ("DAL",),
    "Detroit Red Wings": ("DET",),
    "Edmonton Oilers": ("EDM",),
    "Florida Panthers": ("FLA", "FLO"),
    "Los Angeles Kings": ("LA", "LAK"),
    "Minnesota Wild": ("MIN",),
    "Montreal Canadiens": ("MTL", "MON"),
    "Montréal Canadiens": ("MTL", "MON"),
    "Nashville Predators": ("NSH", "NAS"),
    "New Jersey Devils": ("NJ", "NJD"),
    "New York Islanders": ("NYI",),
    "New York Rangers": ("NYR",),
    "Ottawa Senators": ("OTT",),
    "Philadelphia Flyers": ("PHI",),
    "Pittsburgh Penguins": ("PIT",),
    "San Jose Sharks": ("SJ", "SJS"),
    "Seattle Kraken": ("SEA",),
    "St Louis Blues": ("STL",),
    "St. Louis Blues": ("STL",),
    "Tampa Bay Lightning": ("TB", "TBL"),
    "Toronto Maple Leafs": ("TOR",),
    "Utah Hockey Club": ("UTA", "UTAH"),
    "Utah Mammoth": ("UTA", "UTAH"),
    "Utah HC": ("UTA", "UTAH"),
    "Vancouver Canucks": ("VAN",),
    "Vegas Golden Knights": ("VGK", "VEG", "LV"),
    "Washington Capitals": ("WSH", "WAS"),
    "Winnipeg Jets": ("WPG", "WIN"),
}

DEFAULT_STEP = 300


def _build_franchise_maps():
    """
    Canonical franchise key for every API name and every ESPN abbreviation.

    v1 collapsed each API name to ONE abbreviation - the first candidate that
    appeared anywhere in the pbp - and matched on that string. That silently
    lost all 82 Utah games in 2024-25: ESPN wrote UTAH that season and UTA the
    next, both codes were present in the pooled data, so the resolver picked
    UTA and every UTAH row failed to match. The event was in the snapshot the
    whole time; the join key was wrong.

    Matching now happens on a franchise key instead, so a franchise whose ESPN
    abbreviation changes mid-window still lines up. Arizona and Utah stay
    SEPARATE keys - same franchise historically, but they never coexist in a
    season and the API names distinguish them, so keeping them apart costs
    nothing and prevents a cross-era mismatch.
    """
    name_to_fr, code_to_fr = {}, {}
    for name, codes in TEAM_CANDIDATES.items():
        fr = codes[0]
        name_to_fr[name] = fr
        for c in codes:
            if code_to_fr.get(c, fr) != fr:
                raise ValueError(
                    f"abbreviation {c!r} claimed by two franchises: "
                    f"{code_to_fr[c]!r} and {fr!r}")
            code_to_fr[c] = fr
    return name_to_fr, code_to_fr


NAME_TO_FRANCHISE, CODE_TO_FRANCHISE = _build_franchise_maps()


# -- small helpers ------------------------------------------------------------

def floor_to_grid(when: datetime, step: int, phase: int) -> datetime:
    epoch = int(when.timestamp())
    return datetime.fromtimestamp(epoch - ((epoch - phase) % step),
                                  tz=timezone.utc)


def american_to_prob(o) -> float:
    o = float(o)
    return (-o) / ((-o) + 100.0) if o < 0 else 100.0 / (o + 100.0)


# -- game index ---------------------------------------------------------------

def load_games(season: int, outdir: Path, season_types: set[int],
               report: dict) -> Optional[pd.DataFrame]:
    """
    One season's games: espn_game_id, teams, and the UTC tip-off.

    Teams and season_type come from the pbp parquet; the tip-off time comes
    from the cached scoreboard JSON, because the pbp table only kept the date.
    Both were written by nhl_espn_pbp_v5.py.
    """
    pbp_path = outdir / "pbp_espn_nhl" / f"pbp_{season}.parquet"
    ids_path = outdir / "espn_nhl_raw" / f"espn_gameids_{season}.json"
    if not pbp_path.exists():
        print(f"  season {season}: missing {pbp_path} - run nhl_espn_pbp_v5.py")
        return None
    if not ids_path.exists():
        print(f"  season {season}: missing {ids_path} - run nhl_espn_pbp_v5.py")
        return None

    pbp = pd.read_parquet(pbp_path, columns=["game_id", "season", "season_type",
                                             "game_date", "home_team",
                                             "away_team"])
    g = pbp.drop_duplicates("game_id").copy()
    g["game_id"] = g["game_id"].astype(str)

    ids = json.loads(ids_path.read_text())
    times = {str(r["game_id"]): r.get("date") for r in ids if r.get("date")}
    g["commence_time"] = pd.to_datetime(g["game_id"].map(times), utc=True,
                                        errors="coerce")

    n_no_time = int(g["commence_time"].isna().sum())
    report["no_commence_time"] += n_no_time
    g = g[g["commence_time"].notna()]

    before = len(g)
    g = g[g["season_type"].isin(season_types)]
    report["excluded_season_type"] += before - len(g)

    g["season"] = season
    return g


def resolve_team_map(all_abbrs: set[str], report: dict) -> dict[str, str]:
    """
    Odds API full name -> the abbreviation your pbp actually uses.

    Any franchise whose candidates are all absent is reported. That happens
    legitimately for Arizona if the window has no Coyotes seasons, and
    illegitimately if ESPN uses a code I did not anticipate - the report says
    which, because it lists the unresolved names next to the codes present.
    """
    out = {}
    for name, cands in TEAM_CANDIDATES.items():
        for c in cands:
            if c in all_abbrs:
                out[name] = c
                break
        else:
            report["unresolved_franchises"].add(name)
    return out


# -- http ---------------------------------------------------------------------

def fetch_snapshot(when: datetime, api_key: str, cache_dir: Path,
                   session: requests.Session, markets: str, regions: str,
                   stats: dict) -> Optional[dict]:
    """One historical snapshot, cached to disk so a re-run is free."""
    # The cache key carries markets and regions. A snapshot fetched for h2h
    # alone does not contain totals, and reusing it under a different markets
    # string would silently produce a file with no totals in it.
    tag = f"{markets.replace(',', '-')}_{regions.replace(',', '-')}"
    cache = cache_dir / f"{when.strftime('%Y%m%dT%H%M%SZ')}_{tag}.json"
    if cache.exists() and cache.stat().st_size > 0:
        stats["cached"] += 1
        try:
            return json.loads(cache.read_text())
        except json.JSONDecodeError:
            cache.unlink()

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
        "bookmakers": "pinnacle",
        "date": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for attempt in range(5):
        try:
            r = session.get(f"{API_BASE}/historical/sports/{SPORT}/odds",
                            params=params, timeout=60)
        except requests.RequestException as e:
            stats["errors"] += 1
            time.sleep(min(30.0, 2 ** attempt))
            if attempt == 4:
                print(f"  request failed for {when}: {e}")
            continue

        stats["requests"] += 1
        rem = r.headers.get("x-requests-remaining")
        if rem is not None:
            stats["remaining"] = rem
        used = r.headers.get("x-requests-used")
        if used is not None:
            stats["used"] = used

        if r.status_code == 401:
            print(f"  HTTP 401 - the key was rejected: {r.text[:200]}")
            return None
        if r.status_code == 422:
            print(f"  HTTP 422 for {when}: {r.text[:200]}")
            return None
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(30.0, 2 ** attempt))
            continue
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} for {when}: {r.text[:200]}")
            return None

        try:
            payload = r.json()
        except ValueError:
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))
        return payload
    return None


def calibrate_grid(sample: datetime, api_key: str, cache_dir: Path,
                   session: requests.Session, markets: str, regions: str,
                   stats: dict) -> tuple[int, int]:
    """
    Discover (step_seconds, phase_seconds) for the snapshot grid near `sample`.

    `previous_timestamp` and `next_timestamp` bracket the served snapshot, so
    the gap between them is the real spacing and the served timestamp modulo
    that spacing is the phase.
    """
    payload = fetch_snapshot(sample, api_key, cache_dir, session, markets,
                             regions, stats)
    if payload is None:
        return DEFAULT_STEP, 0
    ts = payload.get("timestamp")
    if not ts:
        return DEFAULT_STEP, 0
    t = pd.Timestamp(ts).to_pydatetime()

    step = None
    for other in (payload.get("previous_timestamp"),
                  payload.get("next_timestamp")):
        if other:
            d = abs(int((pd.Timestamp(other).to_pydatetime() - t).total_seconds()))
            if d > 0:
                step = d if step is None else min(step, d)
    if not step:
        step = DEFAULT_STEP

    # Snap to the nearest real cadence. Measured gaps come back a second or two
    # off - 298 rather than 300 - and an unsnapped value makes the derived
    # phase drift between runs, which changes every requested timestamp and
    # therefore misses the entire snapshot cache. A re-run then costs full
    # price instead of nothing.
    step = min((300, 600), key=lambda c: abs(c - step))
    phase = int(t.timestamp()) % step
    phase = int(round(phase / 60.0) * 60) % step
    return step, phase


# -- extraction ---------------------------------------------------------------

def extract_pinnacle(payload: dict, team_map: dict[str, str],
                     report: dict) -> list[dict]:
    """See extract_pinnacle_fr - kept only so older helper scripts import."""
    return extract_pinnacle_fr(payload, report)


def extract_pinnacle_fr(payload: dict, report: dict) -> list[dict]:
    """
    Pull Pinnacle h2h and totals out of one snapshot.

    Both markets are read from the same event, so a row carries the moneyline
    and the total priced at the same instant. Either can be missing
    independently - Pinnacle sometimes has one market up and not the other -
    and a missing one is left null and counted rather than dropping the game.
    """
    snap_ts = payload.get("timestamp")
    out = []
    for ev in payload.get("data") or []:
        home_raw, away_raw = ev.get("home_team"), ev.get("away_team")
        home = NAME_TO_FRANCHISE.get(home_raw)
        away = NAME_TO_FRANCHISE.get(away_raw)
        if home is None or away is None:
            report["unmapped_names"].add(home_raw if home is None else away_raw)
            continue

        ml: dict[str, float] = {}
        total = over = under = None
        for bk in ev.get("bookmakers") or []:
            if bk.get("key") != "pinnacle":
                continue
            for mk in bk.get("markets") or []:
                key = mk.get("key")
                if key == "h2h":
                    for oc in mk.get("outcomes") or []:
                        nm = NAME_TO_FRANCHISE.get(oc.get("name"))
                        if nm is not None and oc.get("price") is not None:
                            ml[nm] = float(oc["price"])
                elif key == "totals":
                    for oc in mk.get("outcomes") or []:
                        nm = oc.get("name")
                        if nm == "Over":
                            over = oc.get("price")
                            total = oc.get("point", total)
                        elif nm == "Under":
                            under = oc.get("price")
                            total = total if total is not None else oc.get("point")

        has_ml = home in ml and away in ml
        has_total = total is not None and over is not None and under is not None
        if not has_ml:
            report["no_pinnacle_ml"] += 1
        if not has_total:
            report["no_pinnacle_total"] += 1
        if not has_ml and not has_total:
            continue

        out.append({
            "snapshot_time": snap_ts,
            "commence_time_api": ev.get("commence_time"),
            "odds_api_event_id": ev.get("id"),
            "home_fr": home,
            "away_fr": away,
            "home_close_ml": ml.get(home) if has_ml else np.nan,
            "away_close_ml": ml.get(away) if has_ml else np.nan,
            "total_close": float(total) if has_total else np.nan,
            "over_price": float(over) if has_total else np.nan,
            "under_price": float(under) if has_total else np.nan,
        })
    return out


def merge_out(new: pd.DataFrame, path: Path, key: str = "espn_game_id"
              ) -> pd.DataFrame:
    """Merge with whatever is already on disk instead of overwriting it."""
    if not path.exists():
        return new
    old = pd.read_parquet(path)
    old[key] = old[key].astype(str)
    merged = (pd.concat([old, new], ignore_index=True)
                .drop_duplicates(key, keep="last"))
    print(f"  merging {path.name}: {len(old):,} existing -> {len(merged):,}")
    return merged


# -- main ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key", default=None,
                    help="Falls back to the ODDS_API_KEY env var.")
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    default=[2021, 2026],
                    help="ESPN season numbers (ENDING year), inclusive. NHL "
                         "historical odds start late 2020, so 2021 is the "
                         "earliest fully covered season.")
    ap.add_argument("--outdir", type=Path, default=Path("data"),
                    help="Same --outdir you gave nhl_espn_pbp_v5.py.")
    ap.add_argument("--season-types", type=int, nargs="+", default=[2, 3],
                    help="ESPN season types to price. 1 preseason, 2 regular, "
                         "3 postseason, 4 all-star. Default 2 3.")
    ap.add_argument("--regions", default="eu",
                    help="Pinnacle is an eu book. Adding regions multiplies "
                         "the quota cost for no extra data.")
    ap.add_argument("--markets", default="h2h,totals")
    ap.add_argument("--lead-seconds", type=int, default=60,
                    help="Ask for the snapshot this many seconds before puck "
                         "drop. Default 60.")
    ap.add_argument("--max-lag-seconds", type=int, default=1800,
                    help="Reject a matched snapshot staler than this.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan the pull and print the request count per "
                         "season. Spends nothing.")
    ap.add_argument("--probe", action="store_true",
                    help="Make exactly ONE request, print the raw JSON shape "
                         "and quota headers, then stop.")
    ap.add_argument("--max-requests", type=int, default=0,
                    help="Hard stop after N requests. 0 means no cap.")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("ODDS_API_KEY", "")
    if not api_key and not args.dry_run:
        print("Need --api-key or the ODDS_API_KEY env var.", file=sys.stderr)
        return 1

    cache_dir = args.outdir / "odds_raw" / "nhl_pinnacle_snapshots"
    out_dir = args.outdir / "odds"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"unmapped_names": set(), "unresolved_franchises": set(),
              "no_pinnacle_ml": 0, "no_pinnacle_total": 0, "no_snapshot": 0,
              "too_stale": 0, "no_snapshot_before_tip": 0,
              "no_commence_time": 0, "excluded_season_type": 0}

    first, last = args.seasons
    season_types = set(args.season_types)

    # -- game index ----------------------------------------------------------
    frames = []
    for season in range(first, last + 1):
        g = load_games(season, args.outdir, season_types, report)
        if g is not None and len(g):
            frames.append(g)
    if not frames:
        print("No games loaded. Run nhl_espn_pbp_v5.py first.", file=sys.stderr)
        return 1
    sched = pd.concat(frames, ignore_index=True)

    abbrs = set(sched["home_team"]) | set(sched["away_team"])
    unknown = sorted(a for a in abbrs if a not in CODE_TO_FRANCHISE)
    sched["home_fr"] = sched["home_team"].map(CODE_TO_FRANCHISE)
    sched["away_fr"] = sched["away_team"].map(CODE_TO_FRANCHISE)
    before = len(sched)
    sched = sched[sched["home_fr"].notna() & sched["away_fr"].notna()].copy()
    print(f"{len(sched):,} games, {len(abbrs)} distinct team codes, "
          f"{sched['home_fr'].nunique()} franchises")
    if unknown:
        print(f"  ESPN codes not in TEAM_CANDIDATES: {unknown} - "
              f"{before - len(sched):,} games skipped. Add them.")

    sched["target"] = (sched["commence_time"]
                       - pd.Timedelta(seconds=args.lead_seconds))

    session = requests.Session()
    stats = {"requests": 0, "cached": 0, "errors": 0,
             "remaining": "?", "used": "?"}

    grids: dict[int, tuple[int, int]] = {}
    if args.dry_run or not api_key:
        for season in sorted(sched["season"].unique()):
            grids[int(season)] = (DEFAULT_STEP, 0)
        print(f"Dry run: assuming a {DEFAULT_STEP}s grid with no phase. The "
              f"real grid is measured from the API when actually fetching, so "
              f"live counts can differ from these.")
    else:
        print("Calibrating the snapshot grid (one request per season):")
        for season in sorted(sched["season"].unique()):
            sample = sched.loc[sched["season"] == season, "target"].min()
            step, phase = calibrate_grid(
                pd.Timestamp(sample).to_pydatetime(), api_key, cache_dir,
                session, args.markets, args.regions, stats)
            grids[int(season)] = (step, phase)
            print(f"  {int(season)}: {step}s spacing, phase +{phase}s "
                  f"({phase // 60}m past the step)")

    sched["snapshot_req"] = [
        floor_to_grid(pd.Timestamp(t).to_pydatetime(), *grids[int(s)])
        for t, s in zip(sched["target"], sched["season"])
    ]

    plan = sched.groupby("season")["snapshot_req"].nunique()
    total_snaps = sched["snapshot_req"].nunique()
    print("\nPlan - distinct snapshots to request (games sharing a puck-drop "
          "time share one request):")
    print(f"  {'season':>7} {'games':>7} {'snapshots':>10}")
    for season, n in plan.items():
        print(f"  {int(season):>7} "
              f"{int((sched['season'] == season).sum()):>7,} {int(n):>10,}")
    print(f"  {'TOTAL':>7} {len(sched):>7,} {total_snaps:>10,}")
    print(f"\nThat is the REQUEST count, not the credit cost. Each request is "
          f"metered per market x region: {args.markets} x {args.regions}.")

    if args.dry_run:
        return 0

    # -- probe ---------------------------------------------------------------
    if args.probe:
        when = pd.Timestamp(sorted(sched["snapshot_req"].unique())[0]).to_pydatetime()
        print(f"\nProbe: one request at {when}")
        payload = fetch_snapshot(when, api_key, cache_dir, session,
                                 args.markets, args.regions, stats)
        if payload is None:
            print("  no payload returned")
            return 1
        print(f"  top-level keys : {list(payload.keys())}")
        print(f"  timestamp      : {payload.get('timestamp')}")
        print(f"  previous       : {payload.get('previous_timestamp')}")
        print(f"  next           : {payload.get('next_timestamp')}")
        data = payload.get("data") or []
        print(f"  games in data  : {len(data)}")
        if data:
            print("\n  first event:")
            print(json.dumps(data[0], indent=2)[:2500])
        print(f"\n  x-requests-remaining : {stats['remaining']}")
        print(f"  x-requests-used      : {stats['used']}")
        return 0

    # -- fetch ---------------------------------------------------------------
    targets = sorted(pd.Timestamp(t).to_pydatetime()
                     for t in sched["snapshot_req"].unique())
    rows: list[dict] = []
    for i, when in enumerate(targets, 1):
        if args.max_requests and stats["requests"] >= args.max_requests:
            print(f"\nStopping: hit --max-requests {args.max_requests}. "
                  f"Re-run to continue; fetched snapshots are cached.")
            break
        payload = fetch_snapshot(when, api_key, cache_dir, session,
                                 args.markets, args.regions, stats)
        if payload is None:
            report["no_snapshot"] += 1
            continue
        rows.extend(extract_pinnacle_fr(payload, report))
        if i % 50 == 0 or i == len(targets):
            print(f"  [{i}/{len(targets)}] requests={stats['requests']:,} "
                  f"cached={stats['cached']:,} rows={len(rows):,} "
                  f"remaining={stats['remaining']}")
        time.sleep(args.sleep)

    if not rows:
        print("No Pinnacle prices extracted.", file=sys.stderr)
        return 1

    odds = pd.DataFrame(rows)
    odds["snapshot_time"] = pd.to_datetime(odds["snapshot_time"], utc=True)

    # A snapshot contains every game live at that moment, including ones that
    # start hours later, so the same game appears in many snapshots. Keep, per
    # game, the LATEST snapshot at or before its own puck drop.
    merged = sched.merge(odds, on=["home_fr", "away_fr"], how="left")
    merged = merged[merged["snapshot_time"].notna()]
    merged["lag"] = (merged["commence_time"]
                     - merged["snapshot_time"]).dt.total_seconds()
    merged = merged[merged["lag"] >= 0]

    report["no_snapshot_before_tip"] = int(
        sched["game_id"].nunique() - merged["game_id"].nunique())
    merged = merged.sort_values("lag").groupby("game_id", as_index=False).first()

    stale = merged["lag"] > args.max_lag_seconds
    report["too_stale"] = int(stale.sum())
    merged = merged[~stale].copy()

    merged["game_date"] = pd.to_datetime(merged["game_date"]).dt.date
    merged = merged.rename(columns={"game_id": "espn_game_id",
                                    "lag": "snapshot_lag_seconds"})

    # -- moneyline file ------------------------------------------------------
    ml = merged[merged["home_close_ml"].notna()
                & merged["away_close_ml"].notna()].copy()
    p_home = ml["home_close_ml"].map(american_to_prob)
    p_away = ml["away_close_ml"].map(american_to_prob)
    tot = p_home + p_away
    ml["p_home_close"] = np.where(tot > 0, p_home / tot, np.nan)
    ml["overround_close"] = tot - 1.0
    ml["line_source"] = "pinnacle"
    ml_out = ml[["season", "game_date", "home_team", "away_team",
                 "espn_game_id", "home_close_ml", "away_close_ml",
                 "p_home_close", "overround_close", "snapshot_time",
                 "commence_time", "snapshot_lag_seconds", "line_source"]]
    ml_path = out_dir / "closing_lines_pinnacle.parquet"
    ml_final = merge_out(ml_out, ml_path).sort_values(["season", "game_date"])
    ml_final.to_parquet(ml_path, index=False)

    # -- totals file ---------------------------------------------------------
    tt = merged[merged["total_close"].notna()].copy()
    p_over = tt["over_price"].map(american_to_prob)
    p_under = tt["under_price"].map(american_to_prob)
    tsum = p_over + p_under
    tt["p_over_close"] = np.where(tsum > 0, p_over / tsum, np.nan)
    tt["overround_total"] = tsum - 1.0
    tt["book"] = "pinnacle"
    tt_out = tt[["season", "game_date", "home_team", "away_team",
                 "espn_game_id", "total_close", "over_price", "under_price",
                 "p_over_close", "overround_total", "snapshot_time",
                 "commence_time", "snapshot_lag_seconds", "book"]]
    tt_path = out_dir / "closing_totals_pinnacle.parquet"
    tt_final = merge_out(tt_out, tt_path).sort_values(["season", "game_date"])
    tt_final.to_parquet(tt_path, index=False)

    # -- report --------------------------------------------------------------
    print(f"\nWrote {ml_path}: {len(ml_final):,} games")
    print(f"Wrote {tt_path}: {len(tt_final):,} games")
    print(f"  requests made        : {stats['requests']:,}")
    print(f"  snapshots from cache : {stats['cached']:,}")
    print(f"  quota remaining      : {stats['remaining']}")
    print(f"  quota used           : {stats['used']}")

    print("\nCoverage against the ESPN game index:")
    print(f"  {'season':>7} {'games':>7} {'ml':>7} {'ml%':>7} "
          f"{'totals':>7} {'tot%':>7}")
    for season in sorted(sched["season"].unique()):
        n_all = int((sched["season"] == season).sum())
        n_ml = int((ml_out["season"] == season).sum())
        n_tt = int((tt_out["season"] == season).sum())
        print(f"  {int(season):>7} {n_all:>7,} {n_ml:>7,} "
              f"{n_ml / max(1, n_all):>6.1%} {n_tt:>7,} "
              f"{n_tt / max(1, n_all):>6.1%}")

    print("\nSnapshot staleness (puck drop minus snapshot, seconds):")
    print(merged["snapshot_lag_seconds"].describe(
        percentiles=[0.5, 0.9, 0.99]).round(0).to_string())
    print("\nMoneyline overround:")
    print(ml["overround_close"].describe(
        percentiles=[0.05, 0.5, 0.95]).round(4).to_string())
    print(f"Mean devigged P(home): {ml['p_home_close'].mean():.4f}")
    print("\nTotal line:")
    print(tt["total_close"].describe(
        percentiles=[0.05, 0.5, 0.95]).round(2).to_string())
    print(f"Mean devigged P(over): {tt['p_over_close'].mean():.4f}")

    print("\nIssues:")
    print(f"  games with no tip-off time in the id cache   : "
          f"{report['no_commence_time']:,}")
    print(f"  games excluded by --season-types             : "
          f"{report['excluded_season_type']:,}")
    print(f"  games matched to no snapshot at or before tip: "
          f"{report['no_snapshot_before_tip']:,}")
    print(f"  dropped for a snapshot staler than {args.max_lag_seconds}s : "
          f"{report['too_stale']:,}")
    print(f"  event-snapshots with no Pinnacle moneyline   : "
          f"{report['no_pinnacle_ml']:,}")
    print(f"  event-snapshots with no Pinnacle total       : "
          f"{report['no_pinnacle_total']:,}")
    if report["unmapped_names"]:
        print(f"  UNMAPPED TEAM NAMES: {sorted(report['unmapped_names'])}")
        print("  Add these to TEAM_CANDIDATES - games involving them were "
              "skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
