#!/usr/bin/env python
"""
Stage 2 (ESPN) - build the NHL play-by-play table from ESPN, with wall clock.

READ THIS FIRST - THIS IS NOT A PORT OF nba_espn_pbp.py
-------------------------------------------------------
The NBA script downloads one prebuilt parquet per season from the
sportsdataverse release store. THERE IS NO EQUIVALENT FOR ESPN NHL. I checked
the release store directly: it publishes espn_nba_pbp, espn_wnba_pbp,
espn_cfb_pbp, espn_mens/womens_college_basketball_pbp - and for hockey only
`nhl_pbp_full` / `nhl_pbp_lite`, which are fastRhockey/NHL-API data, NOT ESPN.
I pulled nhl_pbp_full for 2024 and checked its 94 columns: there is no
wallclock column and no per-event timestamp of any kind. It cannot answer the
question you asked.

So this script SCRAPES ESPN per game, which is what the NBA script was written
to avoid. Consequences you should know before running it:

  * ~7,700 games across seasons 2021-2026, one HTTP call each, plus ~1,170
    scoreboard calls to find the game ids. With the default 4 workers this is
    a long run - budget an hour or more, and longer if ESPN throttles.
  * Every game is cached as its own parquet shard, so an interrupted run
    resumes instead of restarting. Re-running after a crash only fetches what
    is missing.
  * ESPN can rate-limit. On 429/5xx the fetch backs off and retries; a game
    that still fails is COUNTED AND LISTED at the end, never silently dropped.

WHAT I VERIFIED AND WHAT I DID NOT
----------------------------------
Verified by download: the sportsdataverse NHL release store contents, the
nhl_pbp_full schema (no wallclock), and the nhl_schedules season files this
script uses for its date list (2021: 952 games / 167 dates, 2026: 1,394 games
/ 211 dates).

Verified from the fastRhockey documentation of the same ESPN endpoint, not by
calling it: the ESPN NHL play row carries `wallclock` (ISO 8601), plus
period_number, clock_display_value, home/away score, team_id, athlete_id_1..3,
coordinate_x/y, strength_text and shot_info_text. I could not reach
site.api.espn.com from where I built this, so the JSON key paths in
parse_summary() are written from that documented shape and are UNTESTED against
a live response. Run one season first (--seasons 2026 2026) and check the
report before committing to all six.

SEASON NUMBERING
----------------
ESPN names a season by its ENDING year, same as the NBA script. `--seasons
2021 2026` therefore means the 2020-21 season through the 2025-26 season.
2021 is the shortened COVID season: 952 games, 2021-01-13 to 2021-07-07.

NHL CLOCK
---------
Three periods of 1200 seconds. Overtime length depends on season type, which is
why season_type is threaded through the clock math:
  regular season (type 2)  period 4 = 300s, period 5 = SHOOTOUT
  postseason     (type 3)  period 4+ = 1200s each, no shootout
Shootout rows have no meaningful game clock. They are flagged `is_shootout` and
their clock columns are left NULL rather than filled with a made-up value.

WALL CLOCK
----------
`wallclock` is kept as the raw ISO string and also as `wallclock_unix`, UTC
epoch seconds. That is the column that joins this table to a Kalshi trade tape.
Rows where ESPN omits wallclock are counted in the report.

SCORES
------
Running scores are forced non-decreasing and the corrections counted, same
guard as the NBA script. `home_final` / `away_final` / `home_won` come from the
ESPN header competitors, not from the last play-by-play row; disagreements
between the two are counted.

Output: one parquet per season under <outdir>/pbp_espn_nhl/pbp_{season}.parquet

Setup
-----
  pip install pandas pyarrow requests

Usage
-----
  python nhl_espn_pbp_v6.py --seasons 2026 2026 --outdir data
  python nhl_espn_pbp_v6.py --seasons 2021 2026 --outdir data
  python nhl_espn_pbp_v6.py --seasons 2026 2026 --outdir data --audit-types
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

SCHEDULE_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/"
                "releases/download/nhl_schedules/nhl_schedule_{season}.parquet")
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary"

REGULATION_PERIODS = 3
PERIOD_SECONDS = 1200
REG_OT_SECONDS = 300
PLAYOFF_OT_SECONDS = 1200
REGULATION_SECONDS = REGULATION_PERIODS * PERIOD_SECONDS  # 3600

SEASON_TYPE_POST = 3

# Every per-game counter. Workers fill a local copy; main merges them.
COUNTER_KEYS = (
    "no_plays", "no_competitors", "no_final_score", "no_season_type",
    "no_period", "no_clock", "no_wallclock", "score_corrections",
)

# Score-integrity keys. These are recomputed from the ASSEMBLED season table,
# not counted during the fetch, so they are correct on a resumed run where most
# games came from cache instead of from ESPN.
SCORE_KEYS = (
    "shootout_games", "shootout_score_unexplained",
    "final_score_disagreement", "winner_would_have_flipped",
)

_local = threading.local()


def session() -> requests.Session:
    """
    One requests.Session per worker thread; Sessions are not thread-safe.

    NO User-Agent override. v1 set "Mozilla/5.0" here and ESPN answered every
    single request with 403 Access Denied - a browser UA on a non-browser
    request is exactly what their bot filter looks for. The requests default
    UA works. Do not "fix" this by adding one back.
    """
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        _local.s = s
    return s


# -- clock --------------------------------------------------------------------

def is_shootout(period: int, season_type: int, period_display: str) -> bool:
    if isinstance(period_display, str) and "shootout" in period_display.lower():
        return True
    return season_type != SEASON_TYPE_POST and period >= 5


def period_length(period: int, season_type: int) -> Optional[int]:
    if period <= REGULATION_PERIODS:
        return PERIOD_SECONDS
    if season_type == SEASON_TYPE_POST:
        return PLAYOFF_OT_SECONDS
    if period == 4:
        return REG_OT_SECONDS
    return None  # shootout


def period_start_elapsed(period: int, season_type: int) -> Optional[int]:
    if period <= REGULATION_PERIODS:
        return (period - 1) * PERIOD_SECONDS
    if season_type == SEASON_TYPE_POST:
        return REGULATION_SECONDS + (period - 4) * PLAYOFF_OT_SECONDS
    if period == 4:
        return REGULATION_SECONDS
    return None  # shootout


def parse_clock(v) -> Optional[float]:
    """'14:32' -> 872.0 seconds left in the period. '32.4' -> 32.4."""
    if v is None:
        return None
    if not isinstance(v, str):
        return None
    v = v.strip()
    if not v:
        return None
    try:
        if ":" in v:
            mm, ss = v.split(":", 1)
            return int(mm) * 60 + float(ss)
        return float(v)
    except ValueError:
        return None


def parse_wallclock(v) -> Optional[float]:
    """ESPN ISO 8601 -> UTC epoch seconds."""
    if not isinstance(v, str) or not v:
        return None
    try:
        return pd.Timestamp(v).tz_convert("UTC").timestamp()
    except (ValueError, TypeError):
        try:
            return pd.Timestamp(v).tz_localize("UTC").timestamp()
        except Exception:
            return None


# -- http ---------------------------------------------------------------------

_http_errors_shown = 0
_http_lock = threading.Lock()


def _report_http(msg: str) -> None:
    """
    Print the first few HTTP failures instead of swallowing them.

    v1 returned None on every failure with no message, so a blanket 403 looked
    identical to "ESPN had no games that day" and the run printed
    "50/167 dates, 0 games" while telling you nothing. A wrong answer that
    looks like an empty one is the worst kind.
    """
    global _http_errors_shown
    with _http_lock:
        if _http_errors_shown < 5:
            _http_errors_shown += 1
            print(f"  HTTP ERROR: {msg}")
            if _http_errors_shown == 5:
                print("  (further HTTP errors suppressed; see the report at the end)")


def get_json(url: str, params: dict, retries: int = 5) -> Optional[dict]:
    for attempt in range(retries):
        try:
            r = session().get(url, params=params, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(30.0, 1.0 * (2 ** attempt)))
                continue
            if r.status_code >= 400:
                # 401/403 and friends do not get better on retry. Say so once
                # and give up on this request immediately.
                _report_http(f"{r.status_code} {r.url} :: {r.text[:120]!r}")
                return None
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                _report_http(f"{type(e).__name__} {url} :: {e}")
            time.sleep(min(30.0, 1.0 * (2 ** attempt)))
        except json.JSONDecodeError as e:
            _report_http(f"bad JSON from {url} :: {e}")
            return None
    return None


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=180) as r:
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}  {url}")
                return False
            tmp = dest.with_suffix(dest.suffix + ".part")
            n = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    n += len(chunk)
            tmp.replace(dest)
        print(f"  fetched {dest.name}  ({n / 1e6:.1f} MB)")
        return True
    except requests.RequestException as e:
        print(f"  FAILED  {url}: {e}")
        return False


# -- game discovery -----------------------------------------------------------

def season_dates(season: int, raw_dir: Path) -> list[str]:
    """
    Every date the NHL played in this season, as YYYYMMDD.

    Taken from the sportsdataverse nhl_schedules release rather than from ESPN,
    because it is one small file per season instead of a blind date sweep, and
    because it is the same file for every season back to 2021. It is only used
    to decide WHICH DATES to ask ESPN about - no game data comes from it.
    """
    path = raw_dir / f"nhl_schedule_{season}.parquet"
    if not download(SCHEDULE_URL.format(season=season), path):
        return []
    sch = pd.read_parquet(path, columns=["game_date"])
    dates = sorted({str(d).replace("-", "")[:8]
                    for d in sch["game_date"].dropna().unique()})
    return dates


def espn_games_on(date_yyyymmdd: str) -> list[dict]:
    """ESPN event ids for one date. Empty list means ESPN had nothing."""
    data = get_json(SCOREBOARD_URL, {"dates": date_yyyymmdd, "limit": 200})
    if not data:
        return []
    out = []
    for ev in data.get("events") or []:
        gid = ev.get("id")
        if not gid:
            continue
        season = ev.get("season") or {}
        out.append({
            "game_id": str(gid),
            "season": season.get("year"),
            "season_type": season.get("type"),
            "date": ev.get("date"),
        })
    return out


# -- summary parsing ----------------------------------------------------------

def _pid(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _competitors(summary: dict) -> tuple[Optional[dict], Optional[dict]]:
    header = summary.get("header") or {}
    comps = header.get("competitions") or []
    if not comps:
        return None, None
    home = away = None
    for c in comps[0].get("competitors") or []:
        if c.get("homeAway") == "home":
            home = c
        elif c.get("homeAway") == "away":
            away = c
    return home, away


def parse_summary(game_id: str, summary: dict, report: dict,
                  type_audit: Optional[set]) -> Optional[pd.DataFrame]:
    plays = summary.get("plays") or []
    if not plays:
        report["no_plays"] += 1
        return None

    header = summary.get("header") or {}
    season_block = header.get("season") or {}
    season = _pid(season_block.get("year"))
    season_type = _pid(season_block.get("type"))
    if season_type is None:
        season_type = 2
        report["no_season_type"] += 1

    home_c, away_c = _competitors(summary)
    if home_c is None or away_c is None:
        report["no_competitors"] += 1
        return None

    def team_field(c, key):
        return (c.get("team") or {}).get(key)

    home_id = _pid(team_field(home_c, "id") or c_id(home_c))
    away_id = _pid(team_field(away_c, "id") or c_id(away_c))
    home_abbr = team_field(home_c, "abbreviation") or ""
    away_abbr = team_field(away_c, "abbreviation") or ""

    home_final = _pid(home_c.get("score"))
    away_final = _pid(away_c.get("score"))
    if home_final is None or away_final is None:
        report["no_final_score"] += 1
        return None

    if home_c.get("winner") is not None:
        home_won = int(bool(home_c.get("winner")))
    else:
        home_won = int(home_final > away_final)

    comps = header.get("competitions") or [{}]
    game_dt = comps[0].get("date") or ""
    game_date = game_dt[:10]

    rows = []
    for n, p in enumerate(plays, 1):
        period_block = p.get("period") or {}
        period = _pid(period_block.get("number"))
        if period is None:
            report["no_period"] += 1
            continue
        period_display = period_block.get("displayValue") or ""

        type_block = p.get("type") or {}
        type_text = type_block.get("text") or ""
        if type_audit is not None and type_text:
            type_audit.add(type_text)

        so = is_shootout(period, season_type, period_display)
        clock_raw = (p.get("clock") or {}).get("displayValue")
        # ESPN's NHL feed reports clock.displayValue as time ELAPSED in the
        # period, not time remaining. v1-v5 read it as remaining and inverted
        # every clock column in the output. Measured on the full 2025-26
        # season: correlation between seconds_elapsed and wallclock was 0.830
        # under the old reading and 0.994 under this one, and Period Start of
        # the first period came out at 1200 seconds instead of 0.
        pc_elapsed = None if so else parse_clock(clock_raw)

        start = period_start_elapsed(period, season_type)
        plen = period_length(period, season_type)
        if pc_elapsed is None or start is None or plen is None:
            elapsed = None
            pc_left = None
            left_reg = None
            if not so:
                report["no_clock"] += 1
        else:
            elapsed = start + pc_elapsed
            pc_left = plen - pc_elapsed
            left_reg = REGULATION_SECONDS - elapsed

        wc = p.get("wallclock")
        wc_unix = parse_wallclock(wc)
        if wc_unix is None:
            report["no_wallclock"] += 1

        parts = p.get("participants") or []
        ath = []
        for q in parts[:3]:
            a = (q.get("athlete") or {}).get("id")
            ath.append(_pid(a))
        while len(ath) < 3:
            ath.append(None)

        coord = p.get("coordinate") or {}
        strength = p.get("strength") or {}
        shot = p.get("shotInfo") or {}

        rows.append({
            "game_id": str(game_id),
            "season": season,
            "season_type": season_type,
            "game_date": game_date,
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_final": home_final,
            "away_final": away_final,
            "home_won": home_won,
            "play_number": _pid(p.get("sequenceNumber")) or n,
            "play_id": str(p.get("id") or ""),
            "period": period,
            "period_display": period_display,
            "is_shootout": so,
            "type_text": type_text,
            "text": p.get("text") or "",
            "wallclock": wc or "",
            "wallclock_unix": wc_unix,
            "pc_seconds_left": pc_left,
            "seconds_elapsed": elapsed,
            "seconds_left_reg": left_reg,
            "home_score": _pid(p.get("homeScore")),
            "away_score": _pid(p.get("awayScore")),
            "actor_team_id": _pid((p.get("team") or {}).get("id")),
            "athlete_id_1": ath[0],
            "athlete_id_2": ath[1],
            "athlete_id_3": ath[2],
            "scoring_play": bool(p.get("scoringPlay", False)),
            "score_value": _pid(p.get("scoreValue")),
            "shooting_play": bool(p.get("shootingPlay", False)),
            "strength_text": strength.get("text") or "",
            "shot_type_text": shot.get("text") or "",
            "coord_x": coord.get("x"),
            "coord_y": coord.get("y"),
        })

    if not rows:
        report["no_plays"] += 1
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values(["period", "play_number"]).reset_index(drop=True)

    # Scores forced non-decreasing. Same failure mode as the NBA feed: the
    # points are right, adjacent rows are occasionally swapped, and a running
    # score that moves backwards breaks anything sampled on a time grid.
    hs = pd.to_numeric(df["home_score"], errors="coerce").ffill().fillna(0)
    as_ = pd.to_numeric(df["away_score"], errors="coerce").ffill().fillna(0)
    hs_fixed, as_fixed = hs.cummax(), as_.cummax()
    report["score_corrections"] += int((hs_fixed != hs).sum() + (as_fixed != as_).sum())
    df["home_score"] = hs_fixed.astype(int)
    df["away_score"] = as_fixed.astype(int)
    df["score_margin"] = df["home_score"] - df["away_score"]

    # The final-score check lives in score_report(), which runs over the
    # assembled season instead of here. Two reasons: it has to understand
    # shootouts (see score_report), and counting it here makes the number wrong
    # on any resumed run, because cached games never pass through this
    # function.
    return df


def score_report(df: pd.DataFrame) -> dict:
    """
    Score integrity, computed from the assembled season table.

    THE SHOOTOUT RULE. A shootout game ends the run of play TIED. The NHL then
    credits the winner one extra goal in the official line, so the header reads
    X+1 to X while the last play-by-play row reads X to X. v2 compared the two
    unconditionally and reported 111 disagreements and 53 "winner would have
    flipped" on 2025-26 - which was the check being wrong, not the data.

    A shootout game is accepted if its last play-by-play row is EITHER the
    header line (ESPN sometimes does fold the shootout goal into the running
    score - 10 of 119 games in 2025-26) OR that line with the winner's extra
    goal removed, i.e. tied at the loser's total. Anything else is a real
    disagreement and is reported as `shootout_score_unexplained`, so a genuinely
    broken shootout game is not laundered by the rule.

    Non-shootout games are compared straight, and only they can trip the
    winner-flip test: in a shootout game the play-by-play winner is tied by
    rule, and `home_won` comes from the header regardless.
    """
    out = {k: 0 for k in SCORE_KEYS}
    if df.empty:
        return out

    last = (df.sort_values(["game_id", "period", "play_number"])
              .groupby("game_id", sort=False).tail(1))
    so_games = set(df.loc[df["is_shootout"], "game_id"].unique())

    for r in last.itertuples():
        h, a = int(r.home_score), int(r.away_score)
        hf, af = int(r.home_final), int(r.away_final)
        if r.game_id in so_games:
            out["shootout_games"] += 1
            tied = min(hf, af)
            if (h, a) != (hf, af) and (h, a) != (tied, tied):
                out["shootout_score_unexplained"] += 1
            continue
        if (h, a) != (hf, af):
            out["final_score_disagreement"] += 1
            if (hf > af) != (h > a):
                out["winner_would_have_flipped"] += 1
    return out


def c_id(c: dict):
    """Some ESPN payloads put the team id on the competitor, not on .team."""
    return c.get("id")


# -- per-game fetch -----------------------------------------------------------

def fetch_game(game: dict, cache_dir: Path, audit: bool,
               sleep: float) -> tuple[Optional[str], dict, set]:
    """
    Fetch and shard one game.

    Returns (failed_game_id or None, local counter dict, local type set).
    Counters are LOCAL and merged by the caller: `d[k] += 1` is not atomic, so
    incrementing one shared dict from several worker threads silently loses
    counts, and these counters are the only evidence of data-quality problems.
    """
    local = {k: 0 for k in COUNTER_KEYS}
    types: set = set() if audit else set()
    gid = game["game_id"]
    shard = cache_dir / f"{gid}.parquet"
    if shard.exists():
        return None, local, types

    summary = get_json(SUMMARY_URL, {"event": gid})
    time.sleep(sleep)
    if not summary:
        return gid, local, types

    df = parse_summary(gid, summary, local, types if audit else None)
    if df is None or df.empty:
        # An empty shard is still written so a game ESPN genuinely has no
        # play-by-play for is not re-fetched on every resume.
        pd.DataFrame().to_parquet(shard, index=False)
        return None, local, types
    df.to_parquet(shard, index=False)
    return None, local, types


# -- per-season build ---------------------------------------------------------

def build_season(season: int, raw_dir: Path, out_dir: Path, report: dict,
                 type_audit: Optional[set], workers: int,
                 sleep: float) -> Optional[pd.DataFrame]:
    dates = season_dates(season, raw_dir)
    if not dates:
        print(f"  no schedule for {season}")
        return None
    print(f"  {len(dates):,} game dates {dates[0]} to {dates[-1]}")

    cache_dir = raw_dir / f"espn_games_{season}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. game ids, one scoreboard call per date
    id_cache = raw_dir / f"espn_gameids_{season}.json"
    if id_cache.exists():
        games = json.loads(id_cache.read_text())
        print(f"  {len(games):,} game ids (cached)")
    else:
        games = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(espn_games_on, d): d for d in dates}
            done = 0
            for f in as_completed(futs):
                games.extend(f.result())
                done += 1
                if done % 50 == 0:
                    print(f"    scoreboard {done}/{len(dates)} dates, "
                          f"{len(games):,} games")
        seen, uniq = set(), []
        for g in games:
            if g["game_id"] not in seen:
                seen.add(g["game_id"])
                uniq.append(g)
        games = uniq
        if not games:
            # Do NOT cache an empty result. v1 would have written an empty
            # id file and every later run would have "resumed" into nothing.
            print("  0 game ids - ESPN returned no events for any date. "
                  "Nothing cached; fix the fetch and re-run.")
            return None
        id_cache.write_text(json.dumps(games))
        print(f"  {len(games):,} game ids")

    if not games:
        return None

    # 2. one summary call per game
    todo = [g for g in games if not (cache_dir / f"{g['game_id']}.parquet").exists()]
    print(f"  {len(todo):,} games to fetch ({len(games) - len(todo):,} cached)")
    failed = []
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(fetch_game, g, cache_dir, type_audit is not None,
                              sleep) for g in todo]
            done = 0
            for f in as_completed(futs):
                bad, local, types = f.result()
                for k, v in local.items():
                    report[k] += v
                if type_audit is not None:
                    type_audit |= types
                if bad:
                    failed.append(bad)
                done += 1
                if done % 100 == 0 or done == len(todo):
                    print(f"    {done}/{len(todo)} games, {len(failed)} failed")
    report["failed_games"] += len(failed)
    if failed:
        report["failed_ids"].extend(failed[:50])

    # 3. assemble
    frames = []
    for g in games:
        p = cache_dir / f"{g['game_id']}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        if len(d):
            frames.append(d)
        else:
            report["games_without_pbp"] += 1
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    for k, v in score_report(df).items():
        report[k] += v
    return df


# -- main ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    default=[2021, 2026],
                    help="ESPN season numbers (ENDING year), inclusive. "
                         "Default 2021 2026 = the 2020-21 through 2025-26 seasons.")
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent ESPN requests. Raise it and ESPN is more "
                         "likely to throttle; the retry/backoff handles that but "
                         "the run gets slower, not faster.")
    ap.add_argument("--sleep", type=float, default=0.10,
                    help="Pause after each game request, per worker.")
    ap.add_argument("--rebuild", action="store_true",
                    help="Rewrite season parquets that already exist. Rebuilds "
                         "from the cached per-game shards - no games are "
                         "re-fetched from ESPN.")
    ap.add_argument("--audit-types", action="store_true",
                    help="Print every distinct play type_text seen.")
    args = ap.parse_args()

    first, last = args.seasons
    raw_dir = args.outdir / "espn_nhl_raw"
    out_dir = args.outdir / "pbp_espn_nhl"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {k: 0 for k in COUNTER_KEYS}
    report.update({k: 0 for k in SCORE_KEYS})
    report.update({"failed_games": 0, "games_without_pbp": 0, "failed_ids": []})
    type_audit: Optional[set] = set() if args.audit_types else None

    totals = []
    for season in range(first, last + 1):
        print(f"\n=== ESPN season {season} ({season - 1}-{str(season)[-2:]}) ===")
        out_path = out_dir / f"pbp_{season}.parquet"
        if out_path.exists() and not args.rebuild:
            existing = pd.read_parquet(out_path, columns=["game_id"])
            print(f"  exists: {len(existing):,} rows, "
                  f"{existing['game_id'].nunique():,} games - skipping")
            continue

        df = build_season(season, raw_dir, out_dir, report, type_audit,
                          args.workers, args.sleep)
        if df is None or df.empty:
            print(f"  no data for {season}")
            continue

        df.to_parquet(out_path, index=False)
        n_games = df["game_id"].nunique()
        no_wc = int(df["wallclock_unix"].isna().sum())
        totals.append((season, len(df), n_games, no_wc))
        print(f"  wrote {out_path.name}: {len(df):,} rows, {n_games:,} games, "
              f"{no_wc:,} rows with no wallclock ({no_wc / len(df):.2%})")

    if totals:
        print("\nSeason totals:")
        print(f"  {'season':>7} {'rows':>11} {'games':>7} {'no wallclock':>13}")
        for s, r, gm, w in totals:
            print(f"  {s:>7} {r:>11,} {gm:>7,} {w:>13,}")
        print(f"  {'TOTAL':>7} {sum(t[1] for t in totals):>11,} "
              f"{sum(t[2] for t in totals):>7,} {sum(t[3] for t in totals):>13,}")

    print("\nData quality:")
    print(f"  games ESPN returned no play-by-play for            : {report['games_without_pbp']:,}")
    print(f"  games whose summary request failed outright        : {report['failed_games']:,}")
    print(f"  games with no home/away competitor block           : {report['no_competitors']:,}")
    print(f"  games with no final score in the header            : {report['no_final_score']:,}")
    print(f"  games with no season type (assumed regular season) : {report['no_season_type']:,}")
    print(f"  plays with no period number (dropped)              : {report['no_period']:,}")
    print(f"  non-shootout plays with an unreadable game clock   : {report['no_clock']:,}")
    print(f"  plays with no wallclock                            : {report['no_wallclock']:,}")
    print(f"  score cells corrected for going backwards (new only): {report['score_corrections']:,}")
    print(f"  shootout games (pbp ends tied by rule)             : {report['shootout_games']:,}")
    print(f"    of those, a shootout line the rule cannot explain: {report['shootout_score_unexplained']:,}")
    print(f"  non-shootout games whose pbp final != the header   : {report['final_score_disagreement']:,}")
    print(f"    of those, games where the WINNER would be wrong  : {report['winner_would_have_flipped']:,}")
    if report["failed_ids"]:
        print(f"\n  failed game ids (first {len(report['failed_ids'])}): "
              f"{report['failed_ids']}")
        print("  Re-run the same command; cached games are skipped and only "
              "these are retried.")

    if type_audit is not None:
        print(f"\nDistinct play type_text seen ({len(type_audit)}):")
        for t in sorted(type_audit):
            print(f"  {t}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
