#!/usr/bin/env python
"""
Build season-to-date (YTD) team, skater and goalie stats as they stood BEFORE
each game, seasons 2011-2026.

SOURCE - AND WHY IT IS NOT YOUR ESPN PLAY-BY-PLAY
------------------------------------------------
You sent pbp_2026.parquet, the ESPN file nhl_espn_pbp_v5.py built. That file
exists to carry `wallclock`, which is what joins play-by-play to the Kalshi
trade tape. It cannot carry this job. ESPN's feed has no on-ice skaters, no
line changes and no time-on-ice, and its shot type sits in free text rather
than a field. Without on-ice data there is no TOI, no per-60 rate, no on-ice
Corsi or xGF%, no zone starts and no reliable goalie attribution on goals -
which is most of what "advanced" and "sabermetric" mean in hockey.

This script therefore reads the sportsdataverse `nhl_pbp_full` release, the
NHL-API/hockeyR feed that goes back to 2011 - the one you already have back to
2011. Verified on the 2024 file: 1,106,678 rows, 94 columns, including
home_on_1..7_id / away_on_1..7_id, home_goalie_id / away_goalie_id,
strength_state, home_skaters / away_skaters, shot_distance, shot_angle, a
precomputed `xg`, and 661,226 CHANGE events. Seasons 2011 through 2026 are all
present. Files are cached under <outdir>/nhl_pbp_raw, so a re-run is free.

ONE JOIN KEY WARNING. This feed is keyed by NHL game_id (e.g. 2023020001),
NOT by espn_game_id. Your odds and Kalshi tables are keyed by espn_game_id.
Joining the two needs a crosswalk on (game_date, home_team, away_team) that
this script does not build - it emits game_date and team codes on every row so
you can build one.

THINGS I CHECKED IN THE DATA RATHER THAN ASSUMED
------------------------------------------------
* BLOCKED_SHOT has `event_team_abbr` set to the BLOCKING team, not the
  shooting team. event_player_1 is the shooter and matched the event team on
  0.0% of sampled rows; event_player_2 is the blocker and matched on 92.7%.
  Attributing a blocked shot to `event_team_abbr` would put roughly 48,000
  Corsi events a season on the wrong side. This script flips it.
* `strength_state` is written from the EVENT team's perspective, so it is
  useless for a player whose team did not generate the event. `home_skaters`
  and `away_skaters` are 100% populated and goalie-exclusive, so all strength
  logic is derived from those instead.
* `event_goalie_id` is populated on SHOT (100%), MISSED_SHOT (99.1%) and GOAL
  (94.5%). The opposing on-ice goalie fills the rest.
* TOI from consecutive-event gaps was validated on a full game: intervals sum
  to exactly 3600s, the top skater came to 26.9 minutes, and the two goalies
  to 60.0 and 57.5 - the 57.5 being a pulled goalie, which is correct.

NO LEAKAGE
----------
Every YTD column is a cumulative sum over that entity's PRIOR games in the
same season, shifted by one game. The row for a game contains nothing from
that game. Rate columns (percentages, per-60s) are computed FROM the shifted
totals, never shifted after the fact - dividing two independently shifted
series is where this normally goes wrong. `..._gp_prior` on every table tells
you the sample size behind the row, so opening-night rows (all zero, gp_prior
= 0) can be dropped or flagged rather than silently treated as real.

Playoffs are a separate accumulation from the regular season: --season-types
controls which are built, and the cumulative reset is on (entity, season,
season_type) so playoff YTD does not inherit an 82-game regular-season total.

WHAT COMES OUT
--------------
Six files under <outdir>/stats. The `_box` files are per-game values for that
game - the audit trail. The `_ytd` files are what you model on.

  team_box.parquet     team_ytd.parquet
  skater_box.parquet   skater_ytd.parquet
  goalie_box.parquet   goalie_ytd.parquet

TEAM     basic       gf ga sf sa hits blocks giveaways takeaways fow fol pim
                     pp_opp pp_goals pk_opp pk_ga w l otl
         advanced    cf ca ff fa xgf xga, all-situations and 5v5, plus toi_5v5
         sabermetric pdo_5v5, sh%/sv% splits, hdcf/hdca (xg>=0.10),
                     scf/sca (xg>=0.05), CF% split by score state
SKATER   basic       toi all/5v5/pp/sh, g a1 a2 p shots pim +/- hits
                     hits_taken blocks giveaways takeaways fow fol
         advanced    icf iff ixg, on-ice cf ca ff fa xgf xga gf ga at 5v5
         sabermetric CF%/xGF% and their RELATIVE versions (on-ice minus the
                     team's rate without the player), IPP, finishing (g-ixg),
                     penalties drawn/taken per 60, OZ/DZ/NZ start share
GOALIE   basic       toi sa sv ga sv% gaa starts wins shutouts
         advanced    sa/ga/sv% split ev/pp/sh, sa per 60, xga faced
         sabermetric GSAx (xga-ga), GSAx/60, high-danger sv% (xg>=0.10)

Setup
-----
  pip install pandas pyarrow requests numpy

Usage
-----
  python nhl_ytd_stats_v6.py --seasons 2011 2026 --outdir data
  python nhl_ytd_stats_v6.py --seasons 2011 2026 --outdir data --season-types 2 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

PBP_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases"
           "/download/nhl_pbp_full/play_by_play_{season}.parquet")

ON_HOME = [f"home_on_{i}_id" for i in range(1, 8)]
ON_AWAY = [f"away_on_{i}_id" for i in range(1, 8)]

SHOT_TYPES = ("SHOT", "GOAL", "MISSED_SHOT", "BLOCKED_SHOT")
FENWICK_TYPES = ("SHOT", "GOAL", "MISSED_SHOT")

# Danger thresholds on the feed's own xg. Named constants, not magic numbers -
# change them here and every derived column follows.
HD_XG = 0.10
SC_XG = 0.05

NEEDED = [
    "game_id", "season", "season_type", "game_date", "home_abbr", "away_abbr",
    "event_idx", "event_type", "secondary_type", "event_team_abbr",
    "event_team_type", "period", "period_type", "game_seconds",
    "home_score", "away_score",
    "event_player_1_id", "event_player_2_id", "event_player_3_id",
    "event_goalie_id", "penalty_minutes", "penalty_severity",
    "home_skaters", "away_skaters", "x_fixed", "xg",
    "home_goalie_id", "away_goalie_id",
] + ON_HOME + ON_AWAY


# -- io -----------------------------------------------------------------------

def download(season: int, raw_dir: Path) -> Optional[Path]:
    path = raw_dir / f"play_by_play_{season}.parquet"
    if path.exists() and path.stat().st_size > 0:
        return path
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = PBP_URL.format(season=season)
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            if r.status_code != 200:
                print(f"  HTTP {r.status_code} for season {season}")
                return None
            tmp = path.with_suffix(".part")
            n = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    n += len(chunk)
            tmp.replace(path)
        print(f"  fetched play_by_play_{season}.parquet ({n / 1e6:.0f} MB)")
        return path
    except requests.RequestException as e:
        print(f"  download failed for {season}: {e}")
        return None


def load_season(path: Path, season: int,
                season_types: set[int]) -> pd.DataFrame:
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [c for c in NEEDED if c in have]
    missing = [c for c in NEEDED if c not in have]
    df = pd.read_parquet(path, columns=cols)
    for c in missing:
        df[c] = np.nan
    if missing:
        print(f"    columns absent in this season, filled null: {missing}")

    # season_type arrives as 'R'/'P' in this feed; normalise to ESPN's ints so
    # one flag works across every script in the project.
    # This feed writes season as 20232024. Every other table in the project
    # keys on the ESPN ending year, so normalise here rather than leaving two
    # conventions to collide at join time.
    df["season"] = season

    # game_date is missing from some seasons of this feed (2023 has 93 columns
    # and no date at all; 2024 has 94 and does). Ordering the YTD accumulation
    # is the one thing that CANNOT be approximated, so fall back to the NHL
    # game_id, which encodes season, type and a sequence number that runs in
    # scheduled order within each. It sorts identically to the date and is
    # present in every season.
    if "game_date" not in df.columns or df["game_date"].isna().all():
        df["game_date"] = pd.NA
        df["has_game_date"] = False
    else:
        df["has_game_date"] = True
    df["game_date"] = df["game_date"].astype("string")

    st = df["season_type"].astype(str).str.upper().str[0]
    df["season_type"] = st.map({"P": 3, "R": 2, "1": 1, "2": 2, "3": 3})
    df = df[df["season_type"].isin(season_types)]

    # Shootout attempts are not hockey for accounting purposes - they are not
    # shots on goal, not xG, and not goalie work in any rate stat.
    pt = df["period_type"].astype(str).str.upper()
    df = df[pt != "SHOOTOUT"]

    df = df.sort_values(["game_id", "period", "game_seconds", "event_idx"])
    return df.reset_index(drop=True)


# -- shared derivations -------------------------------------------------------

def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Per-event fields every downstream aggregation needs, all vectorised."""
    g = df["game_id"]
    nxt = df["game_seconds"].shift(-1)
    same = g.eq(g.shift(-1))
    df["dur"] = np.where(same, nxt - df["game_seconds"], 0.0)
    df["dur"] = df["dur"].astype("float32").clip(lower=0.0)

    df["home_skaters"] = pd.to_numeric(df["home_skaters"],
                                       errors="coerce").fillna(0).astype("int8")
    df["away_skaters"] = pd.to_numeric(df["away_skaters"],
                                       errors="coerce").fillna(0).astype("int8")

    # Attacking side of a shot attempt, as 1=home / 0=away / -1=none.
    # BLOCKED_SHOT is the exception: this feed sets event_team to the BLOCKING
    # team, so it has to be flipped or every blocked attempt lands on the wrong
    # team's Corsi.
    ett = df["event_team_type"].astype(str)
    side = np.where(ett == "home", 1, np.where(ett == "away", 0, -1)).astype("int8")
    df["event_side"] = side
    blocked = df["event_type"].eq("BLOCKED_SHOT").to_numpy()
    att = np.where(blocked & (side >= 0), 1 - side, side).astype("int8")
    df["attack_side"] = att

    df["xg"] = pd.to_numeric(df["xg"], errors="coerce").fillna(0.0).astype("float32")
    et = df["event_type"]
    df["is_corsi"] = et.isin(SHOT_TYPES).to_numpy()
    df["is_fenwick"] = et.isin(FENWICK_TYPES).to_numpy()
    df["is_goal"] = et.eq("GOAL").to_numpy()
    df["is_sog"] = et.isin(("SHOT", "GOAL")).to_numpy()
    df["is_5v5"] = ((df["home_skaters"] == 5) & (df["away_skaters"] == 5)).to_numpy()

    df["fen_xg"] = np.where(df["is_fenwick"], df["xg"], 0.0).astype("float32")
    df["hd"] = (df["is_fenwick"] & (df["xg"] >= HD_XG)).astype("int16")
    df["sc"] = (df["is_fenwick"] & (df["xg"] >= SC_XG)).astype("int16")
    for c in ("is_corsi", "is_fenwick", "is_goal", "is_sog"):
        df[c] = df[c].astype("int16")
    return df


def strength_code(own: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """0 = 5v5, 1 = PP, 2 = SH, 3 = other even."""
    return np.where((own == 5) & (opp == 5), 0,
                    np.where(own > opp, 1,
                             np.where(own < opp, 2, 3))).astype("int8")


STRENGTH_NAME = {0: "5v5", 1: "pp", 2: "sh", 3: "ev"}


def onice_long(df: pd.DataFrame, extra: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    One row per (event, on-ice skater), both sides, goalies removed.

    Built with numpy repeat rather than DataFrame.melt: a full season is 1.1M
    events and 14 on-ice slots, and melting the frame with its text columns
    attached was enough to exhaust memory outright.

    Goalies sit in the same on-ice slots as skaters, so they are dropped by
    comparing against that side's goalie id on the SAME event. Dropping "the
    last slot" or "whoever is a goalie by position" would both be wrong: slot
    order is not fixed, and a skater can dress as an emergency goalie.
    """
    frames = []
    for side_code, (cols, gcol) in enumerate(
            ((ON_AWAY, "away_goalie_id"), (ON_HOME, "home_goalie_id"))):
        vals = df[cols].to_numpy(dtype="float64")
        n, k = vals.shape
        pid = vals.ravel()
        gid_rep = np.repeat(df[gcol].to_numpy(dtype="float64"), k)
        keep = ~np.isnan(pid) & (pid != gid_rep)

        out = {"player_id": pid[keep].astype("int64"),
               "side": np.full(keep.sum(), side_code, dtype="int8")}
        own = np.repeat((df["home_skaters"] if side_code else df["away_skaters"])
                        .to_numpy(), k)[keep]
        opp = np.repeat((df["away_skaters"] if side_code else df["home_skaters"])
                        .to_numpy(), k)[keep]
        out["strength"] = strength_code(own, opp)
        for name, arr in extra.items():
            out[name] = np.repeat(arr, k)[keep]
        frames.append(pd.DataFrame(out))
    return pd.concat(frames, ignore_index=True)


def game_meta(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("game_id", as_index=False).agg(
        season=("season", "first"), season_type=("season_type", "first"),
        game_date=("game_date", "first"),
        home_team=("home_abbr", "first"), away_team=("away_abbr", "first"),
        home_final=("home_score", "max"), away_final=("away_score", "max"))


def attach_teams(box: pd.DataFrame, meta: pd.DataFrame,
                 side_col: str) -> pd.DataFrame:
    box = box.merge(meta, on="game_id", how="left")
    home = box[side_col] == 1
    box["team"] = np.where(home, box["home_team"], box["away_team"])
    box["opponent"] = np.where(home, box["away_team"], box["home_team"])
    box["is_home"] = home.astype("int8")
    return box


# -- team box -----------------------------------------------------------------

FOR_AGG = {"cf": "is_corsi", "ff": "is_fenwick", "sf": "is_sog",
           "gf": "is_goal", "xgf": "fen_xg", "hdcf": "hd", "scf": "sc"}
AGAINST_NAME = {"cf": "ca", "ff": "fa", "sf": "sa", "gf": "ga", "xgf": "xga",
                "hdcf": "hdca", "scf": "sca"}


def team_box(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-game team box.

    Every "against" number is the mirror of the opponent's "for" number on the
    same game, so it is produced by flipping the side rather than aggregated a
    second time. One aggregation cannot disagree with itself; two can, and a
    CF/CA pair that does not balance is the classic silent Corsi bug.
    """
    meta = game_meta(df)
    ev = df.loc[df["is_corsi"].astype(bool) & (df["attack_side"] >= 0)].copy()
    own = np.where(ev["attack_side"] == 1, ev["home_skaters"], ev["away_skaters"])
    opp = np.where(ev["attack_side"] == 1, ev["away_skaters"], ev["home_skaters"])
    ev["str_code"] = strength_code(own, opp)

    spec = {k: (v, "sum") for k, v in FOR_AGG.items()}
    allsit = ev.groupby(["game_id", "attack_side"]).agg(**spec).reset_index()
    bystr = (ev.groupby(["game_id", "attack_side", "str_code"])
               .agg(**spec).reset_index())

    fors = [allsit]
    for code, name in STRENGTH_NAME.items():
        sub = bystr[bystr["str_code"] == code].drop(columns=["str_code"])
        sub = sub.rename(columns={k: f"{k}_{name}" for k in FOR_AGG})
        fors.append(sub)
    fr = fors[0]
    for extra in fors[1:]:
        fr = fr.merge(extra, on=["game_id", "attack_side"], how="outer")
    fr = fr.rename(columns={"attack_side": "team_side"})

    # mirror: against(this side) == for(other side), with the strength label
    # flipped too, since their power play is our penalty kill
    ag = fr.copy()
    ag["team_side"] = 1 - ag["team_side"]
    ren = {}
    for k, a in AGAINST_NAME.items():
        ren[k] = a
        for code, name in STRENGTH_NAME.items():
            mirror = {"pp": "sh", "sh": "pp"}.get(name, name)
            ren[f"{k}_{name}"] = f"{a}_{mirror}"
    ag = ag.rename(columns=ren)
    box = fr.merge(ag, on=["game_id", "team_side"], how="outer")

    # counting events credited straight to the acting team
    misc = df[df["event_side"] >= 0]
    m = misc.groupby(["game_id", "event_side"]).agg(
        hits=("event_type", lambda s: int((s == "HIT").sum())),
        giveaways=("event_type", lambda s: int((s == "GIVEAWAY").sum())),
        takeaways=("event_type", lambda s: int((s == "TAKEAWAY").sum())),
        fow=("event_type", lambda s: int((s == "FACEOFF").sum())),
        pen_taken=("event_type", lambda s: int((s == "PENALTY").sum())),
        pim=("penalty_minutes", "sum"),
    ).reset_index().rename(columns={"event_side": "team_side"})
    # a block is credited to the blocking team, which IS event_team here
    blk = (df[df["event_type"].eq("BLOCKED_SHOT") & (df["event_side"] >= 0)]
           .groupby(["game_id", "event_side"]).size().rename("blocks")
           .reset_index().rename(columns={"event_side": "team_side"}))
    box = box.merge(m, on=["game_id", "team_side"], how="left")
    box = box.merge(blk, on=["game_id", "team_side"], how="left")

    toi = df.groupby("game_id")["dur"].sum().rename("toi_all")
    t5 = df[df["is_5v5"]].groupby("game_id")["dur"].sum().rename("toi_5v5")
    box = box.merge(toi, on="game_id", how="left").merge(t5, on="game_id",
                                                         how="left")
    box = attach_teams(box, meta, "team_side")

    mirror_cols = {"fow": "fol", "pen_taken": "pp_opp"}
    mir = box[["game_id", "team_side"] + list(mirror_cols)].copy()
    mir["team_side"] = 1 - mir["team_side"]
    box = box.merge(mir.rename(columns=mirror_cols),
                    on=["game_id", "team_side"], how="left")
    box["pk_opp"] = box["pen_taken"]

    num = box.select_dtypes(include=[np.number]).columns
    box[num] = box[num].fillna(0)
    box["pp_goals"] = box.get("gf_pp", 0)
    box["pk_ga"] = box.get("ga_sh", 0)

    gf = np.where(box["is_home"] == 1, box["home_final"], box["away_final"])
    ga = np.where(box["is_home"] == 1, box["away_final"], box["home_final"])
    box["goals_for_final"] = gf
    box["goals_against_final"] = ga
    box["w"] = (gf > ga).astype("int8")
    box["l"] = (gf < ga).astype("int8")
    box["gp"] = 1
    return box


# -- skater box ---------------------------------------------------------------

def skater_box(df: pd.DataFrame) -> pd.DataFrame:
    meta = game_meta(df)
    gid = df["game_id"].to_numpy()

    # --- time on ice: every event contributes its interval to whoever was out
    toi_long = onice_long(df, {"game_id": gid, "dur": df["dur"].to_numpy()})
    toi = (toi_long.groupby(["game_id", "player_id", "side"])["dur"].sum()
           .rename("toi_all").reset_index())
    for code, name in STRENGTH_NAME.items():
        t = (toi_long[toi_long["strength"] == code]
             .groupby(["game_id", "player_id", "side"])["dur"].sum()
             .rename(f"toi_{name}"))
        toi = toi.merge(t, on=["game_id", "player_id", "side"], how="left")
    del toi_long

    # --- on-ice shot events, 5v5 only. Filtered BEFORE the explosion, so this
    # is ~90k events rather than the full 1.1M.
    m5 = df["is_corsi"].astype(bool) & (df["attack_side"] >= 0) & df["is_5v5"]
    ev = df.loc[m5]
    on_long = onice_long(ev, {
        "game_id": ev["game_id"].to_numpy(),
        "att": ev["attack_side"].to_numpy(),
        "corsi": ev["is_corsi"].to_numpy(),
        "fen": ev["is_fenwick"].to_numpy(),
        "goal": ev["is_goal"].to_numpy(),
        "fxg": ev["fen_xg"].to_numpy(),
    })
    forus = (on_long["att"] == on_long["side"]).to_numpy()
    on_long["on_cf"] = np.where(forus, on_long["corsi"], 0)
    on_long["on_ca"] = np.where(~forus, on_long["corsi"], 0)
    on_long["on_ff"] = np.where(forus, on_long["fen"], 0)
    on_long["on_fa"] = np.where(~forus, on_long["fen"], 0)
    on_long["on_gf"] = np.where(forus, on_long["goal"], 0)
    on_long["on_ga"] = np.where(~forus, on_long["goal"], 0)
    on_long["on_xgf"] = np.where(forus, on_long["fxg"], 0.0)
    on_long["on_xga"] = np.where(~forus, on_long["fxg"], 0.0)
    oncols = ["on_cf", "on_ca", "on_ff", "on_fa", "on_gf", "on_ga",
              "on_xgf", "on_xga"]
    on = (on_long.groupby(["game_id", "player_id", "side"])[oncols].sum()
          .reset_index())
    del on_long

    # --- zone starts, 5v5. x_fixed points toward the EVENT team's attacking
    # end, so the sign has to be read relative to whoever won the draw.
    fo = df.loc[df["event_type"].eq("FACEOFF") & df["is_5v5"]
                & (df["event_side"] >= 0)]
    z = onice_long(fo, {"game_id": fo["game_id"].to_numpy(),
                        "evside": fo["event_side"].to_numpy(),
                        "x": pd.to_numeric(fo["x_fixed"],
                                           errors="coerce").to_numpy()})
    sign = np.where(z["side"] == z["evside"], 1.0, -1.0)
    xz = z["x"].to_numpy() * sign
    z["oz_starts"] = (xz > 25).astype("int16")
    z["dz_starts"] = (xz < -25).astype("int16")
    z["nz_starts"] = ((xz >= -25) & (xz <= 25)).astype("int16")
    zs = (z.groupby(["game_id", "player_id", "side"])
          [["oz_starts", "dz_starts", "nz_starts"]].sum().reset_index())
    del z

    # --- individual events straight off the event rows
    def ind(mask, col, who="event_player_1_id"):
        s = df.loc[mask, ["game_id", who]].dropna(subset=[who])
        s = s.rename(columns={who: "player_id"})
        s["player_id"] = s["player_id"].astype("int64")
        return s.groupby(["game_id", "player_id"]).size().rename(col).reset_index()

    isgoal = df["event_type"].eq("GOAL")
    parts = [
        ind(isgoal, "g"),
        ind(isgoal, "a1", "event_player_2_id"),
        ind(isgoal, "a2", "event_player_3_id"),
        ind(df["is_sog"].astype(bool), "shots"),
        ind(df["is_corsi"].astype(bool), "icf"),
        ind(df["is_fenwick"].astype(bool), "iff"),
        ind(df["event_type"].eq("HIT"), "hits"),
        ind(df["event_type"].eq("HIT"), "hits_taken", "event_player_2_id"),
        ind(df["event_type"].eq("BLOCKED_SHOT"), "blocks", "event_player_2_id"),
        ind(df["event_type"].eq("GIVEAWAY"), "giveaways"),
        ind(df["event_type"].eq("TAKEAWAY"), "takeaways"),
        ind(df["event_type"].eq("FACEOFF"), "fow"),
        ind(df["event_type"].eq("FACEOFF"), "fol", "event_player_2_id"),
        ind(df["event_type"].eq("PENALTY"), "pen_taken"),
        ind(df["event_type"].eq("PENALTY"), "pen_drawn", "event_player_2_id"),
    ]

    def ind_sum(mask, col, valcol):
        s = df.loc[mask, ["game_id", "event_player_1_id", valcol]].dropna(
            subset=["event_player_1_id"])
        s = s.rename(columns={"event_player_1_id": "player_id"})
        s["player_id"] = s["player_id"].astype("int64")
        return (s.groupby(["game_id", "player_id"])[valcol].sum()
                .rename(col).reset_index())

    parts.append(ind_sum(df["is_fenwick"].astype(bool), "ixg", "xg"))
    parts.append(ind_sum(df["event_type"].eq("PENALTY"), "pim",
                         "penalty_minutes"))

    box = toi.merge(on, on=["game_id", "player_id", "side"], how="left")
    box = box.merge(zs, on=["game_id", "player_id", "side"], how="left")
    for p in parts:
        box = box.merge(p, on=["game_id", "player_id"], how="left")

    box = attach_teams(box, meta, "side")
    num = box.select_dtypes(include=[np.number]).columns
    box[num] = box[num].fillna(0)
    box["p"] = box["g"] + box["a1"] + box["a2"]
    box["plus_minus"] = box["on_gf"] - box["on_ga"]
    box["gp"] = 1
    return box


# -- goalie box ---------------------------------------------------------------

def goalie_box(df: pd.DataFrame) -> pd.DataFrame:
    """
    Goalie workload taken from the shots themselves.

    Shots against, goals against and xGA all come off the same rows, so they
    cannot drift apart. Blocked attempts are excluded - they never reached the
    goalie and are not his work.
    """
    meta = game_meta(df)
    ev = df.loc[df["is_fenwick"].astype(bool) & (df["attack_side"] >= 0)].copy()
    ev["def_side"] = (1 - ev["attack_side"]).astype("int8")
    opp_goalie = np.where(ev["def_side"] == 1, ev["home_goalie_id"],
                          ev["away_goalie_id"])
    gid = pd.to_numeric(ev["event_goalie_id"], errors="coerce").to_numpy()
    gid = np.where(np.isnan(gid), opp_goalie, gid)
    ev["goalie_id"] = gid
    ev = ev[~ev["goalie_id"].isna()]
    ev["goalie_id"] = ev["goalie_id"].astype("int64")

    own = np.where(ev["def_side"] == 1, ev["home_skaters"], ev["away_skaters"])
    opp = np.where(ev["def_side"] == 1, ev["away_skaters"], ev["home_skaters"])
    ev["str_code"] = strength_code(own, opp)
    ev["hd_sa"] = ((ev["xg"] >= HD_XG) & ev["is_sog"].astype(bool)).astype("int16")
    ev["hd_ga"] = ((ev["xg"] >= HD_XG) & ev["is_goal"].astype(bool)).astype("int16")

    keys = ["game_id", "goalie_id", "def_side"]
    box = ev.groupby(keys).agg(sa=("is_sog", "sum"), ga=("is_goal", "sum"),
                               xga=("xg", "sum"), hd_sa=("hd_sa", "sum"),
                               hd_ga=("hd_ga", "sum")).reset_index()
    for code, name in STRENGTH_NAME.items():
        sub = ev[ev["str_code"] == code].groupby(keys).agg(
            **{f"sa_{name}": ("is_sog", "sum"), f"ga_{name}": ("is_goal", "sum"),
               f"xga_{name}": ("xg", "sum")}).reset_index()
        box = box.merge(sub, on=keys, how="left")

    frames = []
    for side_code, gcol in ((1, "home_goalie_id"), (0, "away_goalie_id")):
        t = df[["game_id", gcol, "dur"]].dropna(subset=[gcol]).copy()
        t = t.rename(columns={gcol: "goalie_id"})
        t["goalie_id"] = t["goalie_id"].astype("int64")
        t["def_side"] = np.int8(side_code)
        frames.append(t)
    toi = (pd.concat(frames, ignore_index=True).groupby(keys)["dur"].sum()
           .rename("toi").reset_index())
    box = box.merge(toi, on=keys, how="outer")

    box = attach_teams(box, meta, "def_side")
    num = box.select_dtypes(include=[np.number]).columns
    box[num] = box[num].fillna(0)
    box["sv"] = box["sa"] - box["ga"]
    box["gp"] = 1
    # Most ice time in the game is the starter. Ties do not occur in practice;
    # if one did, the first row wins deterministically rather than both.
    box["gs"] = (box.groupby(["game_id", "def_side"])["toi"]
                 .rank(method="first", ascending=False).eq(1).astype("int8"))
    box["shutout"] = ((box["ga"] == 0) & (box["toi"] >= 3000)).astype("int8")
    return box


# -- YTD ----------------------------------------------------------------------

def to_ytd(box: pd.DataFrame, keys: list[str], sum_cols: list[str],
           id_cols: list[str]) -> pd.DataFrame:
    """
    Cumulative totals over PRIOR games only.

    The shift happens on the cumulative sums, before any division. Shifting a
    ratio, or dividing two separately shifted series, is the classic way this
    leaks or goes subtly wrong.
    """
    b = box.sort_values(keys + ["game_id"]).copy()
    grp = b.groupby(keys, sort=False)
    cum = grp[sum_cols].cumsum() - b[sum_cols]
    out = pd.concat([b[id_cols + keys + ["game_id", "game_date"]], cum], axis=1)
    out["gp_prior"] = grp.cumcount()
    return out.reset_index(drop=True)


def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where(b > 0, a / b, np.nan)


def team_rates(y: pd.DataFrame) -> pd.DataFrame:
    y["cf_pct"] = safe_div(y["cf"], y["cf"] + y["ca"])
    y["ff_pct"] = safe_div(y["ff"], y["ff"] + y["fa"])
    y["sf_pct"] = safe_div(y["sf"], y["sf"] + y["sa"])
    y["xgf_pct"] = safe_div(y["xgf"], y["xgf"] + y["xga"])
    y["cf_pct_5v5"] = safe_div(y["cf_5v5"], y["cf_5v5"] + y["ca_5v5"])
    y["xgf_pct_5v5"] = safe_div(y["xgf_5v5"], y["xgf_5v5"] + y["xga_5v5"])
    y["hdcf_pct"] = safe_div(y["hdcf"], y["hdcf"] + y["hdca"])
    y["sh_pct"] = safe_div(y["gf"], y["sf"])
    y["sv_pct"] = safe_div(y["sa"] - y["ga"], y["sa"])
    y["sh_pct_5v5"] = safe_div(y["gf_5v5"], y["sf_5v5"])
    y["sv_pct_5v5"] = safe_div(y["sa_5v5"] - y["ga_5v5"], y["sa_5v5"])
    y["pdo_5v5"] = y["sh_pct_5v5"] + y["sv_pct_5v5"]
    y["pp_pct"] = safe_div(y["pp_goals"], y["pp_opp"])
    y["pk_pct"] = 1.0 - safe_div(y["pk_ga"], y["pk_opp"])
    y["gf_minus_xgf"] = y["gf"] - y["xgf"]
    y["ga_minus_xga"] = y["ga"] - y["xga"]
    for c in ("cf", "ca", "xgf", "xga", "sf", "sa", "gf", "ga", "hdcf", "hdca"):
        y[f"{c}_per60"] = safe_div(y[c] * 3600.0, y["toi_all"])
    for c in ("cf_5v5", "ca_5v5", "xgf_5v5", "xga_5v5"):
        y[f"{c}_per60"] = safe_div(y[c] * 3600.0, y["toi_5v5"])
    return y


def skater_rates(y: pd.DataFrame) -> pd.DataFrame:
    y["p"] = y["g"] + y["a1"] + y["a2"]
    y["cf_pct_5v5"] = safe_div(y["on_cf"], y["on_cf"] + y["on_ca"])
    y["xgf_pct_5v5"] = safe_div(y["on_xgf"], y["on_xgf"] + y["on_xga"])
    y["gf_pct_5v5"] = safe_div(y["on_gf"], y["on_gf"] + y["on_ga"])
    y["finishing"] = y["g"] - y["ixg"]
    y["sh_pct"] = safe_div(y["g"], y["shots"])
    y["oz_start_pct"] = safe_div(y["oz_starts"],
                                 y["oz_starts"] + y["dz_starts"])
    y["toi_per_gp"] = safe_div(y["toi_all"], y["gp_prior"])
    for c in ("g", "a1", "a2", "p", "shots", "icf", "iff", "ixg", "hits",
              "blocks", "giveaways", "takeaways", "pen_taken", "pen_drawn"):
        y[f"{c}_per60"] = safe_div(y[c] * 3600.0, y["toi_all"])
    for c in ("on_cf", "on_ca", "on_xgf", "on_xga", "on_gf", "on_ga"):
        y[f"{c}_per60"] = safe_div(y[c] * 3600.0, y["toi_5v5"])
    y["fow_pct"] = safe_div(y["fow"], y["fow"] + y["fol"])
    y["ipp"] = safe_div(y["p"], y["on_gf"])
    return y


def add_relative(y: pd.DataFrame) -> pd.DataFrame:
    """
    CF%/xGF% relative to the player's own team over the same prior games.

    Rel is what separates a driver from a passenger on a good team. The team
    baseline here is the team's own YTD 5v5 totals, so a player is compared
    against the team he plays for, not the league.
    """
    tm = (y.groupby(["season", "season_type", "team", "game_id"])
            [["on_cf", "on_ca", "on_xgf", "on_xga"]].sum().reset_index())
    tm = tm.rename(columns={"on_cf": "t_cf", "on_ca": "t_ca",
                            "on_xgf": "t_xgf", "on_xga": "t_xga"})
    y = y.merge(tm, on=["season", "season_type", "team", "game_id"], how="left")
    # Every 5v5 event is credited to five skaters a side, so the team totals
    # summed across players are 5x the team's own. Dividing out keeps the
    # baseline on the same scale as the player's rate.
    off_cf = y["t_cf"] - y["on_cf"] * 1.0
    off_ca = y["t_ca"] - y["on_ca"] * 1.0
    off_xgf = y["t_xgf"] - y["on_xgf"]
    off_xga = y["t_xga"] - y["on_xga"]
    y["cf_pct_rel"] = y["cf_pct_5v5"] - safe_div(off_cf, off_cf + off_ca)
    y["xgf_pct_rel"] = y["xgf_pct_5v5"] - safe_div(off_xgf, off_xgf + off_xga)
    return y.drop(columns=["t_cf", "t_ca", "t_xgf", "t_xga"])


def goalie_rates(y: pd.DataFrame) -> pd.DataFrame:
    y["sv"] = y["sa"] - y["ga"]
    y["sv_pct"] = safe_div(y["sv"], y["sa"])
    y["gaa"] = safe_div(y["ga"] * 3600.0, y["toi"])
    y["gsax"] = y["xga"] - y["ga"]
    y["gsax_per60"] = safe_div(y["gsax"] * 3600.0, y["toi"])
    y["sa_per60"] = safe_div(y["sa"] * 3600.0, y["toi"])
    y["xga_per60"] = safe_div(y["xga"] * 3600.0, y["toi"])
    y["hd_sv_pct"] = safe_div(y["hd_sa"] - y["hd_ga"], y["hd_sa"])
    for s in ("5v5", "pp", "sh", "ev"):
        if f"sa_{s}" in y.columns:
            y[f"sv_pct_{s}"] = safe_div(y[f"sa_{s}"] - y[f"ga_{s}"], y[f"sa_{s}"])
    y["toi_per_gp"] = safe_div(y["toi"], y["gp_prior"])
    return y


# -- main ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    default=[2011, 2026],
                    help="Season ENDING year, inclusive. Default 2011 2026.")
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    ap.add_argument("--xg-cal-seasons", type=int, default=1,
                    help="How many PRIOR seasons feed the xG calibration "
                         "factor. 1 tracks the source's model changes most "
                         "closely; 2-3 is steadier but lags a regime shift. "
                         "Never includes the season being built.")
    ap.add_argument("--season-types", type=int, nargs="+", default=[2, 3],
                    help="2 regular, 3 postseason. Default 2 3. Regular and "
                         "postseason accumulate separately.")
    args = ap.parse_args()

    # The crosswalk written by nhl_pbp_merge_v1.py. Without it every output
    # here is keyed by NHL game_id and will not join to the odds or Kalshi
    # tables, which key on espn_game_id. It is optional only so this script
    # still runs before the merge has been done.
    xw_path = args.outdir / "pbp_merged" / "game_crosswalk.parquet"
    xwalk = None
    if xw_path.exists():
        xwalk = pd.read_parquet(xw_path)[["game_id", "espn_game_id"]]
        xwalk["game_id"] = pd.to_numeric(xwalk["game_id"], errors="coerce")
        xwalk = xwalk.dropna().drop_duplicates("game_id")
        print(f"crosswalk: {len(xwalk):,} games carry an espn_game_id")
    else:
        print(f"NOTE: {xw_path} not found - outputs will be keyed by NHL "
              f"game_id only. Run nhl_pbp_merge_v1.py to add espn_game_id.")

    raw_dir = args.outdir / "nhl_pbp_raw"
    out_dir = args.outdir / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    first, last = args.seasons
    st = set(args.season_types)

    teams, skaters, goalies = [], [], []
    hist: list[tuple[float, float]] = []
    factors = []
    for season in range(first, last + 1):
        print(f"\n=== season {season} ===")
        path = download(season, raw_dir)
        if path is None:
            continue
        df = load_season(path, season, st)
        if df.empty:
            print("  no events after filtering")
            continue
        df = annotate(df)

        # --- own xG, if nhl_xg_model_v1.py has been run.
        # Preferred over the vendor column: the vendor changed its model at
        # the 2025 season, and after per-season scaling the same kinds of shot
        # still moved -29% (tip-ins) to +23% (pokes), which is a reweighting no
        # multiplier undoes. The own model is fit on features recorded
        # identically back to 2011 and is calibrated by construction, so no
        # scaling factor is applied to it at all.
        own_path = args.outdir / "xg_own" / f"xg_{season}.parquet"
        use_own = own_path.exists()
        if use_own:
            own = pd.read_parquet(own_path)
            own["game_id"] = pd.to_numeric(own["game_id"], errors="coerce")
            df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
            before = len(df)
            df = df.merge(own, on=["game_id", "event_idx"], how="left")
            assert len(df) == before, "xg_own join changed the row count"
            hit = df["xg_own"].notna() & df["is_fenwick"].astype(bool)
            fen = int(df["is_fenwick"].sum())
            df["xg"] = df["xg_own"].fillna(0.0).astype("float32")
            df["fen_xg"] = np.where(df["is_fenwick"].astype(bool),
                                    df["xg"], 0.0).astype("float32")
            df["hd"] = (df["is_fenwick"].astype(bool)
                        & (df["xg"] >= HD_XG)).astype("int16")
            df["sc"] = (df["is_fenwick"].astype(bool)
                        & (df["xg"] >= SC_XG)).astype("int16")
            insample = bool(df["xg_in_sample"].fillna(False).any())
            print(f"  xg_own: {int(hit.sum()):,} of {fen:,} unblocked attempts "
                  f"matched" + ("  [IN-SAMPLE SEASON]" if insample else ""))

        # --- xG recalibration.
        # The feed's own xg is not calibrated to goals: on 2023-24 it summed to
        # 3.96 per team per game against 3.07 actual. Left alone that inflates
        # every xGA and turns GSAx into nonsense - the 2023-24 leader came out
        # at +70 goals saved over 59 games, roughly triple any real figure.
        # A single scaling factor per season fixes the level without touching
        # the shot-to-shot ranking, which is the part worth having.
        # The factor uses ONLY seasons already processed, so nothing from this
        # season informs its own numbers. The first season in the range has no
        # prior and is left unscaled - its factor prints as 1.000 so you can
        # see which season that is.
        if use_own:
            # Calibrated already; scaling it would undo that.
            factors.append((season, 1.0, float(df["is_goal"].sum())
                            / max(float(df["fen_xg"].sum()), 1e-9)))
            hist.append((float(df["is_goal"].sum()), float(df["fen_xg"].sum())))
            print(f"  {len(df):,} events, {df['game_id'].nunique():,} games, "
                  f"own xG (no scaling applied)")
            t = team_box(df); teams.append(t)
            sk = skater_box(df); skaters.append(sk)
            g = goalie_box(df); goalies.append(g)
            print(f"  box rows: team {len(t):,}  skater {len(sk):,}  "
                  f"goalie {len(g):,}")
            del df
            continue

        raw_goals = float(df["is_goal"].sum())
        raw_xg = float(df["fen_xg"].sum())
        # TRAILING window, not all history. v4 pooled every prior season and
        # got 0.7393 for 2026 - but the source changed its xg model: the
        # season's own goals/xg ratio sat near 0.71 from 2011 to 2021, moved to
        # 0.77 by 2024, then jumped to 0.9476 in 2025 and 0.9094 in 2026. A
        # pooled factor anchored on the old regime under-scaled recent xg
        # badly, which pushed xGA below GA and left every 2026 goalie with a
        # negative GSAx. A short trailing window tracks the regime instead of
        # averaging across it.
        hist.append((raw_goals, raw_xg))
        window = hist[-(args.xg_cal_seasons + 1):-1]
        if window:
            factor = (sum(g for g, _ in window) /
                      max(sum(x for _, x in window), 1e-9))
        else:
            factor = 1.0
        if factor != 1.0:
            df["xg"] = (df["xg"] * factor).astype("float32")
            df["fen_xg"] = (df["fen_xg"] * factor).astype("float32")
            df["hd"] = (df["is_fenwick"].astype(bool)
                        & (df["xg"] >= HD_XG)).astype("int16")
            df["sc"] = (df["is_fenwick"].astype(bool)
                        & (df["xg"] >= SC_XG)).astype("int16")
        factors.append((season, factor, raw_goals / max(raw_xg, 1e-9)))
        print(f"  {len(df):,} events, {df['game_id'].nunique():,} games, "
              f"xg factor {factor:.4f} (this season's own ratio "
              f"{raw_goals / max(raw_xg, 1e-9):.4f})")

        t = team_box(df); teams.append(t)
        s = skater_box(df); skaters.append(s)
        g = goalie_box(df); goalies.append(g)
        print(f"  box rows: team {len(t):,}  skater {len(s):,}  goalie {len(g):,}")
        del df

    if not teams:
        print("Nothing built.", file=sys.stderr)
        return 1

    tb = pd.concat(teams, ignore_index=True)
    sb = pd.concat(skaters, ignore_index=True)
    gb = pd.concat(goalies, ignore_index=True)
    def add_espn(d: pd.DataFrame) -> pd.DataFrame:
        if xwalk is None:
            return d
        d = d.copy()
        d["game_id"] = pd.to_numeric(d["game_id"], errors="coerce")
        return d.merge(xwalk, on="game_id", how="left")

    tb, sb, gb = add_espn(tb), add_espn(sb), add_espn(gb)
    for name, d in (("team_box", tb), ("skater_box", sb), ("goalie_box", gb)):
        d.to_parquet(out_dir / f"{name}.parquet", index=False)

    skip = {"game_id", "season", "season_type", "game_date", "team", "opponent",
            "is_home", "team_side", "side", "def_side", "player_id",
            "goalie_id", "home_team", "away_team", "home_abbr", "away_abbr",
            "espn_game_id",
            "home_final", "away_final", "gp_prior"}

    tsum = [c for c in tb.select_dtypes(include=[np.number]).columns if c not in skip]
    ty = to_ytd(tb, ["season", "season_type", "team"], tsum,
                ["is_home", "opponent"] + (["espn_game_id"] if xwalk is not None
                                           else []))
    ty = team_rates(ty)
    ty.to_parquet(out_dir / "team_ytd.parquet", index=False)

    ssum = [c for c in sb.select_dtypes(include=[np.number]).columns if c not in skip]
    sy = to_ytd(sb, ["season", "season_type", "player_id"], ssum,
                ["team", "opponent", "is_home"] + (["espn_game_id"]
                                                   if xwalk is not None else []))
    sy = skater_rates(sy)
    sy = add_relative(sy)
    sy.to_parquet(out_dir / "skater_ytd.parquet", index=False)

    gsum = [c for c in gb.select_dtypes(include=[np.number]).columns if c not in skip]
    gy = to_ytd(gb, ["season", "season_type", "goalie_id"], gsum,
                ["team", "opponent", "is_home"] + (["espn_game_id"]
                                                   if xwalk is not None else []))
    gy = goalie_rates(gy)
    gy.to_parquet(out_dir / "goalie_ytd.parquet", index=False)

    # -- report --------------------------------------------------------------
    print(f"\nWrote {out_dir}")
    print(f"  team_box    {len(tb):>9,} rows   team_ytd    {len(ty):>9,} rows "
          f"({len(ty.columns)} cols)")
    print(f"  skater_box  {len(sb):>9,} rows   skater_ytd  {len(sy):>9,} rows "
          f"({len(sy.columns)} cols)")
    print(f"  goalie_box  {len(gb):>9,} rows   goalie_ytd  {len(gy):>9,} rows "
          f"({len(gy.columns)} cols)")

    print(f"\nxG calibration - factor 1.0000 means the own model from "
          f"nhl_xg_model_v1.py was used and needs no scaling; anything else "
          f"is the vendor column scaled by the trailing "
          f"{args.xg_cal_seasons} prior season(s):")
    print(f"  {'season':>7} {'factor':>8} {'own ratio':>10}")
    for sn, f, own in factors:
        drift = abs(f - own) / max(own, 1e-9)
        flag = "  <- source xg model shifted" if drift > 0.10 else ""
        print(f"  {sn:>7} {f:>8.4f} {own:>10.4f}{flag}")
    print("  A flagged row means the factor and the season's own ratio "
          "disagree by over 10%: the scaling is a season behind a change in "
          "the source's xg model, so xGA and GSAx on that season carry a "
          "known level bias.")

    nodate = tb.loc[tb["game_date"].isna(), "season"].unique()
    if len(nodate):
        print(f"\nSeasons with no game_date in the source feed: "
              f"{sorted(int(x) for x in nodate)}")
        print("  YTD ordering for these used the NHL game_id sequence, which "
              "runs in scheduled order. Join on game_id, not on date, for "
              "these seasons.")

    if xwalk is not None:
        miss = int(tb["espn_game_id"].isna().sum())
        print(f"\nespn_game_id attached to {len(tb) - miss:,} of {len(tb):,} "
              f"team rows ({miss:,} unmatched)")

    print("\nGames per season:")
    print(tb.groupby(["season", "season_type"])["game_id"].nunique().to_string())

    print("\nSanity - per-game team box (should look like an NHL box score):")
    print(tb[["gf", "ga", "sf", "sa", "cf", "ca", "xgf", "hits", "blocks",
              "fow"]].describe(percentiles=[0.5]).round(2).to_string())

    print("\nSanity - team totals must balance across the two sides:")
    chk = tb.groupby("game_id")[["cf", "ca", "sf", "sa", "gf", "ga"]].sum()
    print(f"  games where sum(cf) != sum(ca): "
          f"{int((chk['cf'] != chk['ca']).sum()):,} of {len(chk):,}")
    print(f"  games where sum(gf) != sum(ga): "
          f"{int((chk['gf'] != chk['ga']).sum()):,} of {len(chk):,}")

    print("\nSanity - ice time (seconds per game):")
    print(f"  team toi_all  median {tb['toi_all'].median():,.0f}")
    print(f"  skater toi_all median {sb['toi_all'].median():,.0f}  "
          f"p95 {sb['toi_all'].quantile(0.95):,.0f}")
    print(f"  goalie toi    median {gb['toi'].median():,.0f}")

    print("\nSanity - no leakage: opening-night rows must be all zero.")
    op = ty[ty["gp_prior"] == 0]
    print(f"  team opening rows {len(op):,}, max cf on them "
          f"{op['cf'].max() if len(op) else 0:.0f} (must be 0)")

    print("\nYTD leaders, most recent season, min 20 prior games:")
    last_season = int(ty["season"].max())
    lead = ty[(ty["season"] == last_season) & (ty["gp_prior"] >= 20)]
    if len(lead):
        tail = lead.sort_values("game_id").groupby("team").tail(1)
        print(tail.nlargest(5, "xgf_pct_5v5")[
            ["team", "gp_prior", "cf_pct_5v5", "xgf_pct_5v5", "pdo_5v5"]]
            .round(3).to_string(index=False))
    gl = gy[(gy["season"] == last_season) & (gy["gp_prior"] >= 20)]
    if len(gl):
        tail = gl.sort_values("game_id").groupby("goalie_id").tail(1)
        print("\n  goalie GSAx leaders:")
        print(tail.nlargest(5, "gsax")[
            ["goalie_id", "team", "gp_prior", "sv_pct", "gsax", "gsax_per60"]]
            .round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
