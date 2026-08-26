#!/usr/bin/env python
"""
nhl_wp_features_v10.py
=====================
Build per-event live win-probability features for NHL games.

One output parquet per season:  <root>\features\nhl_wp_features_<season>.parquet

Row = one play-by-play event (regular season only, shootout period dropped).
Label = home_won (1/0), resolved through OT and shootout.

Includes an analytic Skellam baseline win probability calibrated to the
Pinnacle closing moneyline + total at t=0, which decays into pure game state
as remaining time shrinks.  The trainer uses its logit as base_margin so the
trees only learn the residual.

Usage
-----
python nhl_wp_features_v1.py --seasons 2021 2022 2023 2024 2025 2026
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.special import gammaln

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

DEFAULT_ROOT = os.environ.get(
    "NHL_ROOT", r"C:\Users\saint\OneDrive\Documents\NHL_AUG_2026\data"
)

REG_SECONDS = 3600
OT_SECONDS = 300
POIS_K = 24

# ----------------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------------


def find_one(root, *patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(root, pat)))
        if hits:
            return hits[0]
    return None


def resolve_paths(root):
    p = {}
    p["pbp_dir"] = find_dir(root, "pbp_merged", "pbp")
    p["odds_dir"] = find_dir(root, "odds")
    p["stats_dir"] = find_dir(root, "stats")
    p["xg_dir"] = find_dir(root, "xg_own", "xg")
    p["out_dir"] = os.path.join(root, "features")
    os.makedirs(p["out_dir"], exist_ok=True)
    missing = [k for k, v in p.items() if v is None]
    if missing:
        sys.exit(f"ERROR: could not locate directories {missing} under {root}")
    return p


def find_dir(root, *names):
    for n in names:
        d = os.path.join(root, n)
        if os.path.isdir(d):
            return d
    return None


def season_pbp_path(pbp_dir, season):
    return find_one(
        pbp_dir,
        f"pbp_{season}.parquet",
        f"*{season}*.parquet",
    )


def season_xg_path(xg_dir, season):
    return find_one(xg_dir, f"xg_{season}.parquet", f"*{season}*.parquet")


# ----------------------------------------------------------------------------
# season-type normalisation  (pbp uses R/P/PR, box/ytd tables use 1/2/3/4)
# ----------------------------------------------------------------------------

REGULAR_TOKENS = {"R", "REG", "REGULAR", "2", "02"}


def is_regular(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64") == 2
    s = series.astype(str).str.strip().str.upper()
    return s.isin(REGULAR_TOKENS)


# ----------------------------------------------------------------------------
# Skellam machinery
# ----------------------------------------------------------------------------


def pois_grid(mu, K=POIS_K):
    """(n,K+1) Poisson pmf."""
    mu = np.asarray(mu, dtype=np.float64)
    k = np.arange(K + 1)
    safe = np.maximum(mu, 1e-12)
    logp = -mu[:, None] + k[None, :] * np.log(safe)[:, None] - gammaln(k + 1.0)[None, :]
    return np.exp(logp)


def skellam_win(mu_h, mu_a, diff, p_tie_home, K=POIS_K):
    """P(home ahead at end of the window) + P(level) * p_tie_home."""
    ph = pois_grid(mu_h, K)
    pa = pois_grid(mu_a, K)
    cdf = np.cumsum(ph, axis=1)
    a = np.arange(K + 1)[None, :]
    d = np.asarray(diff, dtype=np.int64)[:, None]

    thresh = a + 1 - d  # need N_h >= thresh
    sf = 1.0 - np.take_along_axis(cdf, np.clip(thresh - 1, 0, K), axis=1)
    sf = np.where(thresh <= 0, 1.0, sf)
    sf = np.where(thresh > K, 0.0, sf)
    p_win = (pa * sf).sum(axis=1)

    j = a - d
    ph_at = np.take_along_axis(ph, np.clip(j, 0, K), axis=1)
    ph_at = np.where((j < 0) | (j > K), 0.0, ph_at)
    p_lvl = (pa * ph_at).sum(axis=1)

    return p_win + p_lvl * np.asarray(p_tie_home, dtype=np.float64)


def ot_win_prob(lam_h, lam_a, secs, ot_mult, p_so_home, m_h=1.0, m_a=1.0, w=0.0):
    """3-on-3 sudden death for `secs`, then shootout.

    An overtime power play is a two-segment race: the penalty runs for `w`
    seconds at the adjusted rates, then both teams revert to even 3-on-3.
    With w = 0 this is the plain single-segment calculation.
    """
    lam_h = np.asarray(lam_h, dtype=np.float64) * ot_mult
    lam_a = np.asarray(lam_a, dtype=np.float64) * ot_mult
    secs = np.asarray(secs, dtype=np.float64)
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    w = np.minimum(w, secs)

    h1 = lam_h * np.asarray(m_h, dtype=np.float64)
    a1 = lam_a * np.asarray(m_a, dtype=np.float64)
    t1 = h1 + a1
    s1 = np.where(t1 > 0, h1 / np.maximum(t1, 1e-12), 0.5)
    surv1 = np.exp(-t1 * w)

    t2 = lam_h + lam_a
    s2 = np.where(t2 > 0, lam_h / np.maximum(t2, 1e-12), 0.5)
    surv2 = np.exp(-t2 * (secs - w))

    p_home_first = s1 * (1.0 - surv1) + surv1 * s2 * (1.0 - surv2)
    p_none = surv1 * surv2
    return p_home_first + p_none * p_so_home


def solve_rate_split(p_home_target, mu_total, ot_mult, p_so_home, iters=60):
    """Find log(lam_h/lam_a) so that pregame WP == the vig-free closing price."""
    n = len(p_home_target)
    lo = np.full(n, -3.0)
    hi = np.full(n, 3.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        w = np.exp(mid) / (1.0 + np.exp(mid))
        lh = mu_total * w
        la = mu_total * (1.0 - w)
        p_ot = ot_win_prob(lh / REG_SECONDS, la / REG_SECONDS, OT_SECONDS, ot_mult, p_so_home)
        p = skellam_win(lh, la, np.zeros(n, dtype=np.int64), p_ot)
        hi = np.where(p > p_home_target, mid, hi)
        lo = np.where(p > p_home_target, lo, mid)
    mid = 0.5 * (lo + hi)
    w = np.exp(mid) / (1.0 + np.exp(mid))
    return mu_total * w, mu_total * (1.0 - w)


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


# ----------------------------------------------------------------------------
# league priors (OT scoring multiplier, shootout home rate, league total)
# ----------------------------------------------------------------------------


def pull_flags(hs, aw, home_score, away_score, secs_left_reg, max_secs):
    """A pulled goalie is a trailing team with 6+ skaters late in regulation.
    Six skaters with time on the clock and a tied or leading team is a delayed
    penalty extra attacker, which lasts seconds, not the rest of the game."""
    late = secs_left_reg <= max_secs
    he = (hs >= 6) & (home_score < away_score) & late
    ae = (aw >= 6) & (away_score < home_score) & late
    return he, ae


def state_key(n_for, n_against, own_empty, opp_empty):
    return f"{int(n_for)}v{int(n_against)}|{int(bool(own_empty))}{int(bool(opp_empty))}"


BASE_STATE = state_key(5, 5, False, False)


def accumulate_strength(d, st_time, st_gf, max_secs):
    """Time on ice and goals-for by (skaters_for, skaters_against, nets), pooled
    over both teams so mirrored states share a sample."""
    d = d[d["period"] <= 4].copy()
    if d.empty:
        return
    per = d["period"].to_numpy("int64")
    ps = d["period_seconds"].to_numpy("float64")
    d["_t"] = np.where(per <= 3, (per - 1) * 1200.0 + ps, REG_SECONDS + ps)
    d = d.sort_values(["game_id", "_t"], kind="stable")
    nxt = d.groupby("game_id")["_t"].shift(-1)
    dur = (nxt - d["_t"]).fillna(0.0).clip(0, 300).to_numpy("float64")

    hs = pd.to_numeric(d["home_skaters"], errors="coerce").to_numpy("float64")
    aw = pd.to_numeric(d["away_skaters"], errors="coerce").to_numpy("float64")
    is_ot = d["period"].to_numpy("int64") == 4
    slr = np.where(~is_ot, np.maximum(REG_SECONDS - d["_t"].to_numpy("float64"), 0.0), 0.0)
    he, ae = pull_flags(hs, aw,
                        d["home_score"].to_numpy("float64"),
                        d["away_score"].to_numpy("float64"), slr, max_secs)
    goal = (d["event_type"].to_numpy(object) == "GOAL")
    home_ev = (d["event_team_abbr"].to_numpy(object) == d["home_abbr"].to_numpy(object))

    ok = np.isfinite(hs) & np.isfinite(aw) & ~is_ot
    for i in np.flatnonzero(ok):
        kh = state_key(hs[i], aw[i], he[i], ae[i])
        ka = state_key(aw[i], hs[i], ae[i], he[i])
        st_time[kh] = st_time.get(kh, 0.0) + dur[i]
        st_time[ka] = st_time.get(ka, 0.0) + dur[i]
        if goal[i]:
            k = kh if home_ev[i] else ka
            st_gf[k] = st_gf.get(k, 0.0) + 1.0


OT_BASE_STATE = "3v3"


def ot_state_key(n_for, n_against):
    return f"{int(n_for)}v{int(n_against)}"


def accumulate_ot_strength(d, ot_time, ot_gf):
    """Overtime is 3-on-3, so a 4-on-3 must be measured against 3-on-3 rather
    than against the 5-on-5 table (which already has the OT rate multiplier
    applied on top of it)."""
    d = d[d["period"] == 4].copy()
    if d.empty:
        return
    d["_t"] = d["period_seconds"].astype("float64")
    d = d.sort_values(["game_id", "_t"], kind="stable")
    nxt = d.groupby("game_id")["_t"].shift(-1)
    dur = (nxt - d["_t"]).fillna(0.0).clip(0, 300).to_numpy("float64")

    hs = pd.to_numeric(d["home_skaters"], errors="coerce").to_numpy("float64")
    aw = pd.to_numeric(d["away_skaters"], errors="coerce").to_numpy("float64")
    goal = d["event_type"].to_numpy(object) == "GOAL"
    home_ev = d["event_team_abbr"].to_numpy(object) == d["home_abbr"].to_numpy(object)
    ok = np.isfinite(hs) & np.isfinite(aw) & (hs >= 3) & (aw >= 3)

    for i in np.flatnonzero(ok):
        kh = ot_state_key(hs[i], aw[i])
        ka = ot_state_key(aw[i], hs[i])
        ot_time[kh] = ot_time.get(kh, 0.0) + dur[i]
        ot_time[ka] = ot_time.get(ka, 0.0) + dur[i]
        if goal[i]:
            k = kh if home_ev[i] else ka
            ot_gf[k] = ot_gf.get(k, 0.0) + 1.0


def ot_multipliers(ot_time, ot_gf, min_minutes, prior_goals):
    base_t = ot_time.get(OT_BASE_STATE, 0.0)
    base_g = ot_gf.get(OT_BASE_STATE, 0.0)
    if base_t <= 0 or base_g <= 0:
        return {}, {}
    base_rate = base_g / base_t
    mult, diag = {}, {}
    for k, t in ot_time.items():
        if t < min_minutes * 60.0:
            continue
        # regular-season overtime is 3-on-3; anything above 4 a side is scrape
        # noise, not a real state
        n_for, n_against = (int(x) for x in k.split("v"))
        if max(n_for, n_against) > 4:
            continue
        expected = t * base_rate
        mult[k] = float((ot_gf.get(k, 0.0) + prior_goals) / (expected + prior_goals))
        diag[k] = round(t / 60.0, 1)
    mult[OT_BASE_STATE] = 1.0
    return mult, diag


def accumulate_uneven_duration(d, dur_sum, dur_n):
    """How long an uneven-strength state actually lasts, measured by looking
    forward to when the skater counts equalise. Used only as a league constant,
    for rows where the penalty tracker and the skater counts disagree."""
    d = d[d["period"] <= 3].copy()
    if d.empty:
        return
    d["_t"] = (d["period"].to_numpy("int64") - 1) * 1200.0 + d["period_seconds"].to_numpy("float64")
    d = d.sort_values(["game_id", "_t"], kind="stable")
    hs = pd.to_numeric(d["home_skaters"], errors="coerce").to_numpy("float64")
    aw = pd.to_numeric(d["away_skaters"], errors="coerce").to_numpy("float64")
    diff = hs - aw
    t = d["_t"].to_numpy("float64")
    gid = d["game_id"].to_numpy("int64")

    n = len(t)
    rem = np.zeros(n)
    nxt = t[-1] if n else 0.0
    for i in range(n - 1, -1, -1):
        if i == n - 1 or gid[i] != gid[i + 1]:
            nxt = t[i]
        elif diff[i] != diff[i + 1]:
            nxt = t[i + 1]
        rem[i] = nxt - t[i]

    ok = (diff != 0) & (hs < 6) & (aw < 6) & np.isfinite(diff)
    for i in np.flatnonzero(ok):
        k = str(int(abs(diff[i])))
        dur_sum[k] = dur_sum.get(k, 0.0) + rem[i]
        dur_n[k] = dur_n.get(k, 0) + 1


EN_TIME_BUCKET = 60.0
EN_MAX_SECS_LEFT = 900.0


def en_key(score_diff, secs_left):
    sd = int(np.clip(score_diff, -3, 3))
    b = int(min(secs_left, EN_MAX_SECS_LEFT) // EN_TIME_BUCKET)
    return f"{sd}@{b}"


def accumulate_expected_en(d, en_sum, en_n, pull_max_secs):
    """Expected REMAINING empty-net seconds given (score differential, time
    left). The model already reprices once a goalie is actually pulled; this is
    what it needs in order to see the pull coming.

    Measured by summing forward the time actually spent at 6 skaters, which is
    shorter than the time from the pull to the buzzer because the state ends as
    soon as anybody scores.
    """
    d = d[d["period"] <= 3].copy()
    if d.empty:
        return
    per = d["period"].to_numpy("int64")
    ps = d["period_seconds"].to_numpy("float64")
    d["_t"] = (per - 1) * 1200.0 + ps
    d = d.sort_values(["game_id", "_t"], kind="stable")

    gid = d["game_id"].to_numpy("int64")
    t = d["_t"].to_numpy("float64")
    slr = np.clip(REG_SECONDS - t, 0, REG_SECONDS)
    nxt = d.groupby("game_id")["_t"].shift(-1)
    dur = (nxt - d["_t"]).fillna(0.0).clip(0, 300).to_numpy("float64")

    hs = pd.to_numeric(d["home_skaters"], errors="coerce").to_numpy("float64")
    aw = pd.to_numeric(d["away_skaters"], errors="coerce").to_numpy("float64")
    hsc = d["home_score"].to_numpy("float64")
    asc = d["away_score"].to_numpy("float64")
    he, ae = pull_flags(hs, aw, hsc, asc, slr, pull_max_secs)
    en_dur = np.where(he | ae, dur, 0.0)

    # remaining empty-net seconds, summed forward within each game
    n = len(t)
    rem = np.zeros(n)
    run = 0.0
    for i in range(n - 1, -1, -1):
        if i == n - 1 or gid[i] != gid[i + 1]:
            run = 0.0
        run += en_dur[i]
        rem[i] = run

    sd = hsc - asc
    ok = ~(he | ae) & np.isfinite(sd)
    for i in np.flatnonzero(ok):
        k = en_key(sd[i], slr[i])
        en_sum[k] = en_sum.get(k, 0.0) + rem[i]
        en_n[k] = en_n.get(k, 0) + 1


def strength_multipliers(st_time, st_gf, min_minutes, prior_goals):
    """Rate relative to 5v5, shrunk toward 1.0 by `prior_goals` pseudo-goals so
    a thin zero-goal state cannot produce a multiplier of 0."""
    base_t = st_time.get(BASE_STATE, 0.0)
    base_g = st_gf.get(BASE_STATE, 0.0)
    if base_t <= 0 or base_g <= 0:
        return {}, {}
    base_rate = base_g / base_t
    mult, diag = {}, {}
    for k, t in st_time.items():
        if t < min_minutes * 60.0:
            continue
        # 6 skaters with neither net flagged empty is a delayed penalty; the
        # model never prices those, so keep them out of the table
        skaters, nets = k.split("|")
        n_for, n_against = (int(x) for x in skaters.split("v"))
        if max(n_for, n_against) >= 6 and nets == "00":
            continue
        expected = t * base_rate
        mult[k] = float((st_gf.get(k, 0.0) + prior_goals) / (expected + prior_goals))
        diag[k] = round(t / 60.0, 1)
    mult[BASE_STATE] = 1.0
    return mult, diag


def league_goalie_rates(paths, prior_seasons):
    """League goals-saved-above-expected per shot is NOT zero -- the xG model
    drifts season to season. Shrinking a goalie toward 0 instead of toward this
    biases low-shot goalies and turns the stat into a proxy for shots faced."""
    fp = find_one(paths["stats_dir"], "goalie_box.parquet", "*goalie_box*.parquet")
    if fp is None:
        return 0.0, 0.095
    b = pd.read_parquet(fp, columns=["season", "season_type", "sa", "ga", "xga"])
    b = b[is_regular(b["season_type"]) & b["season"].isin(prior_seasons)]
    sa = float(b["sa"].sum())
    if sa <= 0:
        return 0.0, 0.095
    return float((b["xga"].sum() - b["ga"].sum()) / sa), float(b["xga"].sum() / sa)


def build_league_priors(paths, prior_seasons, out_path, rebuild=False,
                        min_minutes=30.0, prior_goals=5.0, pull_max_secs=240.0,
                        ot_min_minutes=20.0):
    if os.path.exists(out_path) and not rebuild:
        with open(out_path) as fh:
            return json.load(fh)

    ot_games = 0
    ot_decided = 0
    st_time = {}
    st_gf = {}
    ot_time = {}
    ot_gf = {}
    dur_sum = {}
    dur_n = {}
    en_sum = {}
    en_n = {}
    so_games = 0
    so_home = 0
    goals = 0
    games = 0

    for season in prior_seasons:
        fp = season_pbp_path(paths["pbp_dir"], season)
        if fp is None:
            continue
        cols = [
            "game_id",
            "period",
            "period_seconds",
            "event_type",
            "event_team_abbr",
            "home_abbr",
            "away_abbr",
            "home_score",
            "away_score",
            "season_type",
            "home_skaters",
            "away_skaters",
            "home_goalie_id",
            "away_goalie_id",
        ]
        d = pd.read_parquet(fp, columns=cols)
        d = d[is_regular(d["season_type"])]
        if d.empty:
            continue
        d["game_id"] = d["game_id"].astype("int64")

        last = d.groupby("game_id").tail(1)
        goals += int((last["home_score"] + last["away_score"]).sum())
        games += last["game_id"].nunique()

        accumulate_strength(d, st_time, st_gf, pull_max_secs)
        accumulate_ot_strength(d, ot_time, ot_gf)
        accumulate_uneven_duration(d, dur_sum, dur_n)
        accumulate_expected_en(d, en_sum, en_n, pull_max_secs)

        reached_ot = d.loc[d["period"] == 4, "game_id"].unique()
        reached_so = d.loc[d["period"] == 5, "game_id"].unique()
        ot_games += len(reached_ot)
        ot_decided += len(reached_ot) - len(reached_so)

        so = d[(d["period"] == 5) & (d["event_type"] == "GOAL")]
        if len(so):
            meta = d.drop_duplicates("game_id").set_index("game_id")[["home_abbr", "away_abbr"]]
            cnt = so.groupby(["game_id", "event_team_abbr"]).size().unstack(fill_value=0)
            for gid in cnt.index:
                ha = meta.at[gid, "home_abbr"]
                aa = meta.at[gid, "away_abbr"]
                h = cnt.at[gid, ha] if ha in cnt.columns else 0
                a = cnt.at[gid, aa] if aa in cnt.columns else 0
                so_games += 1
                so_home += int(h > a)

    # P(OT ends in a goal) = 1 - exp(-rate * 300) at combined league rate
    p_decided = ot_decided / max(ot_games, 1)
    league_goals_per_game = goals / max(games, 1)
    base_rate = league_goals_per_game / REG_SECONDS  # goals per second, both teams
    ot_rate = -np.log(max(1e-6, 1.0 - p_decided)) / OT_SECONDS
    ot_mult = float(ot_rate / max(base_rate, 1e-9))

    mult, st_diag = strength_multipliers(st_time, st_gf, min_minutes, prior_goals)
    so_lg, so_h, so_a = shootout_league_rates(paths, prior_seasons)
    otm, ot_diag = ot_multipliers(ot_time, ot_gf, ot_min_minutes, prior_goals)

    gsp, xsp = league_goalie_rates(paths, prior_seasons)

    pri = {
        "prior_seasons": list(prior_seasons),
        "league_gsax_per_shot": gsp,
        "league_xga_per_shot": xsp,
        "strength_multipliers": mult,
        "expected_en_secs": {k: round(en_sum[k] / max(en_n[k], 1), 2)
                            for k in sorted(en_sum) if en_n[k] >= 200},
        "uneven_mean_remaining_secs": {k: round(dur_sum[k] / max(dur_n[k], 1), 1)
                                      for k in sorted(dur_sum)},
        "ot_strength_multipliers": otm,
        "ot_strength_sample_minutes": ot_diag,
        "strength_sample_minutes": st_diag,
        "league_goals_per_game": float(league_goals_per_game),
        "ot_reached": int(ot_games),
        "ot_decided_in_ot": int(ot_decided),
        "p_ot_decided": float(p_decided),
        "ot_rate_multiplier": ot_mult,
        "so_games": int(so_games),
        "p_shootout_home": float(so_home / max(so_games, 1)),
        "so_league_rate": so_lg,
        "so_home_shooter_rate": so_h,
        "so_away_shooter_rate": so_a,
        "pull_max_secs": float(pull_max_secs),
    }
    with open(out_path, "w") as fh:
        json.dump(pri, fh, indent=2)
    print(f"[priors] {json.dumps(pri, indent=2)}")
    return pri


# ----------------------------------------------------------------------------
# odds
# ----------------------------------------------------------------------------


def load_odds(paths):
    ml_fp = find_one(paths["odds_dir"], "closing_lines_pinnacle.parquet", "*closing_lines*.parquet")
    tot_fp = find_one(paths["odds_dir"], "closing_totals_pinnacle.parquet", "*closing_totals*.parquet")
    if ml_fp is None:
        sys.exit(f"ERROR: no closing_lines parquet in {paths['odds_dir']}")
    ml = pd.read_parquet(ml_fp)[["espn_game_id", "p_home_close", "home_close_ml", "away_close_ml", "overround_close"]]
    ml["espn_game_id"] = ml["espn_game_id"].astype(str)
    ml = ml.drop_duplicates("espn_game_id")
    if tot_fp is not None:
        tot = pd.read_parquet(tot_fp)[["espn_game_id", "total_close", "p_over_close"]]
        tot["espn_game_id"] = tot["espn_game_id"].astype(str)
        tot = tot.drop_duplicates("espn_game_id")
        ml = ml.merge(tot, on="espn_game_id", how="left")
    else:
        ml["total_close"] = np.nan
        ml["p_over_close"] = np.nan
    print(f"[odds] {len(ml):,} games with closing lines")
    return ml


# ----------------------------------------------------------------------------
# YTD stats
# ----------------------------------------------------------------------------

TEAM_YTD_COLS = [
    "game_id", "team", "gp_prior",
    "cf_pct", "ff_pct", "sf_pct", "xgf_pct", "cf_pct_5v5", "xgf_pct_5v5",
    "hdcf_pct", "sh_pct", "sv_pct", "sh_pct_5v5", "sv_pct_5v5", "pdo_5v5",
    "pp_pct", "pk_pct", "gf_minus_xgf", "ga_minus_xga",
    "cf_per60", "ca_per60", "xgf_per60", "xga_per60", "sf_per60", "sa_per60",
    "gf_per60", "ga_per60", "hdcf_per60", "hdca_per60",
    "xgf_5v5_per60", "xga_5v5_per60",
]

TEAM_DIFF_COLS = [c for c in TEAM_YTD_COLS if c not in ("game_id", "team", "gp_prior")]

SKATER_YTD_COLS = [
    "game_id", "player_id", "toi_per_gp", "gp_prior",
    "on_xgf_per60", "on_xga_per60", "on_gf_per60", "on_ga_per60",
    "ixg_per60", "p_per60", "shots_per60", "icf_per60",
    "xgf_pct_5v5", "cf_pct_rel", "xgf_pct_rel", "finishing",
    "pen_taken_per60", "pen_drawn_per60",
]

SKATER_AGG_COLS = [
    "on_xgf_per60", "on_xga_per60", "on_gf_per60", "on_ga_per60",
    "ixg_per60", "p_per60", "shots_per60", "xgf_pct_rel", "cf_pct_rel",
    "toi_per_gp",
]

GOALIE_YTD_COLS = [
    "game_id", "goalie_id", "team", "gp_prior", "sa", "gsax", "toi",
    "sv_pct", "hd_sv_pct", "sv_pct_5v5", "gsax_per60", "sa_per60", "toi_per_gp",
]


def load_stats(paths, seasons):
    sd = paths["stats_dir"]
    out = {}

    fp = find_one(sd, "team_ytd.parquet", "*team_ytd*.parquet")
    t = pd.read_parquet(fp)
    t = t[is_regular(t["season_type"]) & t["season"].isin(seasons)]
    t["game_id"] = t["game_id"].astype("int64")
    out["team"] = t[[c for c in TEAM_YTD_COLS if c in t.columns]].copy()

    fp = find_one(sd, "goalie_ytd.parquet", "*goalie_ytd*.parquet")
    g = pd.read_parquet(fp)
    g = g[is_regular(g["season_type"]) & g["season"].isin(seasons)]
    g["game_id"] = g["game_id"].astype("int64")
    g["goalie_id"] = g["goalie_id"].astype("int64")
    g["season"] = g["season"].astype("int64")
    out["goalie"] = g[[c for c in GOALIE_YTD_COLS if c in g.columns] + ["season"]].copy()

    fp = find_one(sd, "skater_ytd.parquet", "*skater_ytd*.parquet")
    s = pd.read_parquet(fp, columns=SKATER_YTD_COLS + ["season", "season_type"])
    s = s[is_regular(s["season_type"]) & s["season"].isin(seasons)]
    s["game_id"] = s["game_id"].astype("int64")
    s["player_id"] = s["player_id"].astype("int64")
    out["skater"] = s.drop(columns=["season_type"])

    fp = find_one(sd, "team_box.parquet", "*team_box*.parquet")
    b = pd.read_parquet(fp, columns=["game_id", "espn_game_id", "season", "season_type",
                                     "home_final", "away_final"])
    b = b[is_regular(b["season_type"])]
    b["game_id"] = b["game_id"].astype("int64")
    out["finals"] = b.drop_duplicates("game_id")[["game_id", "home_final", "away_final"]]

    return out


# ----------------------------------------------------------------------------
# goalie quality:  regressed goals-saved-above-expected per shot
# ----------------------------------------------------------------------------


def goalie_quality(gy, shot_prior_k, league_gsax_per_shot=0.0):
    g = gy.copy()
    g["sa"] = g["sa"].fillna(0.0)
    g["gsax"] = g["gsax"].fillna(0.0)
    # shrink toward the LEAGUE mean, not toward zero
    g["gsax_per_shot"] = ((g["gsax"] + shot_prior_k * league_gsax_per_shot)
                          / (g["sa"] + shot_prior_k))
    g["gsax_dev"] = g["gsax_per_shot"] - league_gsax_per_shot
    g["g_toi"] = g["toi"].fillna(0.0)

    # team replacement level: pooled prior gsax of every OTHER goalie the team
    # has used this season, as of this game
    team_tot = g.groupby(["season", "team", "game_id"], as_index=False).agg(
        team_gsax=("gsax", "sum"), team_sa=("sa", "sum")
    )
    g = g.merge(team_tot, on=["season", "team", "game_id"], how="left")
    other_gsax = g["team_gsax"] - g["gsax"]
    other_sa = g["team_sa"] - g["sa"]
    g["team_repl_gsax_per_shot"] = ((other_gsax + shot_prior_k * league_gsax_per_shot)
                                    / (other_sa + shot_prior_k))
    g["gsax_vs_repl"] = g["gsax_per_shot"] - g["team_repl_gsax_per_shot"]

    keep = [
        "game_id", "goalie_id", "gsax_per_shot", "gsax_dev", "gsax_vs_repl",
        "team_repl_gsax_per_shot", "sv_pct", "hd_sv_pct", "sv_pct_5v5",
        "gp_prior", "toi_per_gp", "sa",
    ]
    g = g[keep].rename(
        columns={
            "gp_prior": "g_gp_prior",
            "toi_per_gp": "g_toi_per_gp",
            "sa": "g_sa_prior",
            "sv_pct": "g_sv_pct",
            "hd_sv_pct": "g_hd_sv_pct",
            "sv_pct_5v5": "g_sv_pct_5v5",
        }
    )
    return g


# ----------------------------------------------------------------------------
# fast (game_id, player_id) lookup
# ----------------------------------------------------------------------------

PID_BASE = 10_000_000


class PlayerLookup:
    def __init__(self, df, id_col, value_cols):
        key = df["game_id"].to_numpy("int64") * PID_BASE + df[id_col].to_numpy("int64")
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        self.vals = df[value_cols].to_numpy("float64")[order]
        self.cols = list(value_cols)

    def get(self, game_id, player_id):
        """player_id may be float with NaN. Returns (n, len(cols)) with NaN misses."""
        pid = np.asarray(player_id, dtype="float64")
        ok = np.isfinite(pid)
        q = np.where(ok, game_id.astype("int64") * PID_BASE + np.nan_to_num(pid).astype("int64"), -1)
        pos = np.searchsorted(self.key, q)
        pos_c = np.clip(pos, 0, len(self.key) - 1)
        hit = ok & (self.key[pos_c] == q)
        out = np.full((len(q), len(self.cols)), np.nan)
        out[hit] = self.vals[pos_c[hit]]
        return out, hit


# ----------------------------------------------------------------------------
# penalty stack -> power-play seconds remaining
# ----------------------------------------------------------------------------


def penalty_clock(df):
    """
    Track live minor/major penalties to derive PP seconds remaining per side.
    Returns (home_pp_secs_left, away_pp_secs_left) where home_pp_* is time the
    HOME team is on the power play (i.e. an AWAY penalty is running).
    Approximate: minors end on a PP goal, majors do not, misconducts ignored.
    """
    n = len(df)
    home_left = np.zeros(n)
    away_left = np.zeros(n)

    gid = df["game_id"].to_numpy("int64")
    t = df["abs_seconds"].to_numpy("float64")
    etype = df["event_type"].to_numpy(object)
    eteam = df["event_team_abbr"].to_numpy(object)
    home_abbr = df["home_abbr"].to_numpy(object)
    sev = df["penalty_severity"].to_numpy(object)
    mins = df["penalty_minutes"].to_numpy("float64")

    stack_home = []  # penalties ON the home team -> away power play
    stack_away = []
    cur = None

    for i in range(n):
        if gid[i] != cur:
            cur = gid[i]
            stack_home = []
            stack_away = []

        now = t[i]
        stack_home = [p for p in stack_home if p[0] > now]
        stack_away = [p for p in stack_away if p[0] > now]

        et = etype[i]
        if et == "PENALTY":
            m = mins[i]
            s = str(sev[i]).upper() if sev[i] is not None else ""
            if np.isfinite(m) and m > 0 and "MISCONDUCT" not in s:
                on_home = eteam[i] == home_abbr[i]
                major = ("MAJOR" in s) or ("MATCH" in s) or m >= 5
                (stack_home if on_home else stack_away).append((now + m * 60.0, major))
        elif et == "GOAL":
            scorer_home = eteam[i] == home_abbr[i]
            # a minor against the team that was scored on ends
            victim_stack = stack_home if not scorer_home else stack_away
            minors = [k for k, p in enumerate(victim_stack) if not p[1]]
            if minors:
                k = min(minors, key=lambda j: victim_stack[j][0])
                exp, _ = victim_stack[k]
                if exp - now > 120.0:  # double minor: drop the first half only
                    victim_stack[k] = (exp - 120.0, False)
                else:
                    victim_stack.pop(k)

        home_left[i] = max([p[0] for p in stack_away], default=now) - now
        away_left[i] = max([p[0] for p in stack_home], default=now) - now

    return np.clip(home_left, 0, 600), np.clip(away_left, 0, 600)


# ----------------------------------------------------------------------------
# per-season build
# ----------------------------------------------------------------------------

PBP_COLS = [
    "game_id", "season", "season_type", "espn_game_id", "game_date",
    "home_abbr", "away_abbr", "event_idx", "event_type", "event_team_abbr",
    "period", "period_type", "period_seconds", "home_score", "away_score",
    "strength_state", "home_skaters", "away_skaters",
    "empty_net", "extra_attacker",
    "home_goalie_id", "away_goalie_id",
    "penalty_severity", "penalty_minutes",
    "shot_distance", "shot_angle", "xg",
    "wallclock_unix", "wallclock_extrapolated",
] + [f"{s}_on_{i}_id" for s in ("home", "away") for i in range(1, 8)]


def build_season(season, paths, odds, stats, priors, so_prior, args):
    fp = season_pbp_path(paths["pbp_dir"], season)
    if fp is None:
        print(f"[{season}] no pbp file, skipped")
        return None

    import pyarrow.parquet as pqf

    have = set(pqf.ParquetFile(fp).schema_arrow.names)
    cols = [c for c in PBP_COLS if c in have]
    d = pd.read_parquet(fp, columns=cols)
    for c in PBP_COLS:
        if c not in d.columns:
            d[c] = np.nan

    d = d[is_regular(d["season_type"])].copy()
    if d.empty:
        print(f"[{season}] no regular-season rows, skipped")
        return None

    d["game_id"] = d["game_id"].astype("int64")
    d["espn_game_id"] = d["espn_game_id"].astype(str)
    d = d.sort_values(["game_id", "event_idx"], kind="stable").reset_index(drop=True)

    # ---- drop shootout, drop administrative rows -----------------------------
    d = d[d["period"] <= 4].copy()
    d = d[~d["event_type"].isin(["PERIOD_END", "GAME_END", "SHOOTOUT_COMPLETE"])].copy()
    d = d.reset_index(drop=True)

    # ---- clock ---------------------------------------------------------------
    per = d["period"].to_numpy("int64")
    psec = d["period_seconds"].to_numpy("float64")
    d["abs_seconds"] = np.where(per <= 3, (per - 1) * 1200.0 + psec, REG_SECONDS + psec)
    secs_left_reg = np.where(per <= 3, (3 - per) * 1200.0 + (1200.0 - psec), 0.0)
    d["secs_left_reg"] = np.clip(secs_left_reg, 0, REG_SECONDS)
    d["secs_left_ot"] = np.where(per == 4, np.clip(OT_SECONDS - psec, 0, OT_SECONDS), 0.0)
    d["is_ot"] = (per == 4).astype(np.int8)
    d["frac_left"] = d["secs_left_reg"] / REG_SECONDS

    # ---- labels --------------------------------------------------------------
    d = attach_labels(d, stats["finals"], fp)
    if d is None:
        return None

    # ---- odds ----------------------------------------------------------------
    d = d.merge(odds, on="espn_game_id", how="left")
    miss = d["p_home_close"].isna().groupby(d["game_id"]).first().mean()
    print(f"[{season}] games missing closing line: {miss:.3%}")

    league_total = priors["league_goals_per_game"]
    d["total_close_f"] = d["total_close"].fillna(league_total)
    d["p_home_close_f"] = d["p_home_close"].fillna(args.default_home_p)
    d["has_odds"] = d["p_home_close"].notna().astype(np.int8)

    # ---- game state ----------------------------------------------------------
    d["score_diff"] = d["home_score"].astype("int64") - d["away_score"].astype("int64")
    d["total_goals"] = d["home_score"].astype("int64") + d["away_score"].astype("int64")
    d["abs_score_diff"] = d["score_diff"].abs()
    d["is_tied"] = (d["score_diff"] == 0).astype(np.int8)

    hs = d["home_skaters"].astype("float64")
    aws = d["away_skaters"].astype("float64")
    d["skater_diff"] = hs - aws
    d["home_skaters_n"] = hs
    d["away_skaters_n"] = aws
    d["is_even"] = (hs == aws).astype(np.int8)
    d["home_pp"] = (hs > aws).astype(np.int8)
    d["away_pp"] = (aws > hs).astype(np.int8)
    d["is_3v3"] = ((hs == 3) & (aws == 3)).astype(np.int8)

    # a pulled goalie means 6 skaters; a null goalie id is a scrape gap, not a pull
    _he, _ae = pull_flags(hs.to_numpy("float64"), aws.to_numpy("float64"),
                          d["home_score"].to_numpy("float64"),
                          d["away_score"].to_numpy("float64"),
                          d["secs_left_reg"].to_numpy("float64"), args.pull_max_secs)
    d["home_goalie_pulled"] = _he.astype(np.int8)
    d["away_goalie_pulled"] = _ae.astype(np.int8)
    for c in ("home_goalie_id", "away_goalie_id"):
        g = d.groupby("game_id")[c]
        d[c] = g.ffill()
        d[c] = d.groupby("game_id")[c].bfill()
    d["net_empty_diff"] = d["away_goalie_pulled"].astype(np.int8) - d["home_goalie_pulled"].astype(np.int8)

    hpp, app = penalty_clock(d)
    d["home_pp_secs_left"] = hpp
    d["away_pp_secs_left"] = app
    d["pp_secs_diff"] = hpp - app
    agree = float(np.mean((hpp > 0).astype(int) == d["home_pp"].to_numpy()))
    print(f"[{season}] penalty-clock vs on-ice skater count agreement: {agree:.3f}")


    # ---- running in-game tallies --------------------------------------------
    d = attach_running(d)

    # ---- team YTD ------------------------------------------------------------
    d = attach_team_ytd(d, stats["team"])

    # ---- goalies -------------------------------------------------------------
    d = attach_goalies(d, stats["goalie"], priors, args)

    # ---- on-ice skaters ------------------------------------------------------
    d = attach_skaters(d, stats["skater"], args)

    # ---- shootout skill ------------------------------------------------------
    d = attach_shootout(d, so_prior, priors, args)

    # ---- baseline (needs strength state, penalty clock and goalies) ----------
    d = attach_baseline(d, priors, args)

    # ---- interactions --------------------------------------------------------
    d["logit_close_x_fracleft"] = d["logit_close"] * d["frac_left"]
    d["logit_close_x_sqrtleft"] = d["logit_close"] * np.sqrt(d["frac_left"])
    d["score_diff_x_fracleft"] = d["score_diff"] * d["frac_left"]
    d["score_diff_per_sqrt_left"] = d["score_diff"] / np.sqrt(np.maximum(d["secs_left_reg"], 60.0))
    d["xg_diff_minus_score_diff"] = d["xg_diff_run"] - d["score_diff"]

    keep = list(dict.fromkeys(META_COLS + FEATURES + ["home_won"]))
    keep = [c for c in keep if c in d.columns]
    out = d[keep].copy()
    for c in dict.fromkeys(FEATURES):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")

    dest = os.path.join(paths["out_dir"], f"nhl_wp_features_{season}.parquet")
    out.to_parquet(dest, index=False)
    print(f"[{season}] wrote {len(out):,} rows / {out['game_id'].nunique():,} games -> {dest}")
    return dest


def attach_labels(d, finals, pbp_fp):
    fin = finals.copy()
    d = d.merge(fin, on="game_id", how="left")

    # shootout winner: more successful attempts in period 5
    so = pd.read_parquet(
        pbp_fp,
        columns=["game_id", "period", "event_type", "event_team_abbr", "home_abbr", "season_type"],
    )
    so = so[is_regular(so["season_type"]) & (so["period"] == 5) & (so["event_type"] == "GOAL")]
    if len(so):
        so["game_id"] = so["game_id"].astype("int64")
        so["is_home"] = (so["event_team_abbr"] == so["home_abbr"]).astype(int)
        agg = so.groupby("game_id")["is_home"].agg(["sum", "count"])
        agg["so_home_win"] = (agg["sum"] * 2 > agg["count"]).astype(int)
        d = d.merge(agg[["so_home_win"]], on="game_id", how="left")
    else:
        d["so_home_win"] = np.nan

    hf = d["home_final"].to_numpy("float64")
    af = d["away_final"].to_numpy("float64")
    sow = d["so_home_win"].to_numpy("float64")
    won = np.where(hf > af, 1.0, np.where(hf < af, 0.0, sow))
    d["home_won"] = won

    bad = ~np.isfinite(won)
    if bad.any():
        drop_games = d.loc[bad, "game_id"].nunique()
        print(f"[labels] dropping {drop_games} games with unresolved result")
        d = d[~bad].copy()
    if d.empty:
        return None
    d["home_won"] = d["home_won"].astype(np.int8)
    return d.drop(columns=["home_final", "away_final", "so_home_win"], errors="ignore")


def attach_baseline(d, priors, args):
    ot_mult = priors["ot_rate_multiplier"]
    p_so_row = d["p_shootout_home"].to_numpy("float64")

    g = d.drop_duplicates("game_id")[
        ["game_id", "p_home_close_f", "total_close_f", "p_shootout_home"]].copy()
    lam_h_g, lam_a_g = solve_rate_split(
        g["p_home_close_f"].to_numpy("float64"),
        g["total_close_f"].to_numpy("float64"),
        ot_mult,
        g["p_shootout_home"].to_numpy("float64"),
    )
    g["lam_h_game"] = lam_h_g
    g["lam_a_game"] = lam_a_g
    d = d.merge(g[["game_id", "lam_h_game", "lam_a_game"]], on="game_id", how="left")

    lam_h = d["lam_h_game"].to_numpy("float64") / REG_SECONDS
    lam_a = d["lam_a_game"].to_numpy("float64") / REG_SECONDS
    diff = (d["home_score"].astype("int64") - d["away_score"].astype("int64")).to_numpy("int64")
    slr = d["secs_left_reg"].to_numpy("float64")
    slo = d["secs_left_ot"].to_numpy("float64")
    is_ot = d["is_ot"].to_numpy("int8") == 1

    gm_h, gm_a = goalie_window(d, priors)
    d["goalie_mult_home"] = gm_h
    d["goalie_mult_away"] = gm_a
    lam_h = lam_h * gm_h
    lam_a = lam_a * gm_a

    om_h, om_a, ow = ot_strength_window(d, priors)
    d["ot_mult_home"] = om_h
    d["ot_mult_away"] = om_a
    d["ot_window_secs"] = ow

    m_h, m_a, w_state = strength_window(d, priors)
    d["state_mult_home"] = m_h
    d["state_mult_away"] = m_a
    d["state_window_secs"] = w_state
    w = np.minimum(w_state, slr)
    # independent Poissons add, so a piecewise rate is still one Skellam
    eff_h = lam_h * (m_h * w + (slr - w))
    eff_a = lam_a * (m_a * w + (slr - w))

    n = len(d)
    wp = np.empty(n)
    chunk = args.chunk_rows
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        sl = slice(a, b)
        p_ot_full = ot_win_prob(lam_h[sl], lam_a[sl], OT_SECONDS, ot_mult, p_so_row[sl])
        reg = skellam_win(eff_h[sl], eff_a[sl], diff[sl], p_ot_full)
        p_ot_now = ot_win_prob(lam_h[sl], lam_a[sl], slo[sl], ot_mult, p_so_row[sl],
                               om_h[sl], om_a[sl], ow[sl])
        ot_here = is_ot[sl]
        wp[sl] = np.where(
            ot_here,
            np.where(diff[sl] > 0, 1.0, np.where(diff[sl] < 0, 0.0, p_ot_now)),
            reg,
        )

    d["baseline_wp"] = np.clip(wp, 1e-5, 1 - 1e-5)
    d["baseline_logit"] = logit(d["baseline_wp"])
    d["logit_close"] = logit(d["p_home_close_f"])
    d["lam_total"] = d["lam_h_game"] + d["lam_a_game"]
    d["lam_ratio"] = np.log(d["lam_h_game"] / np.maximum(d["lam_a_game"], 1e-9))
    return d


def goalie_window(d, priors):
    """Scale each team's scoring rate by how the opposing goalie now in net
    differs from the starter the market priced.

    Goals allowed per shot is (expected per shot - saved above expected per
    shot), so the ratio of those two quantities is the rate multiplier. With
    the starter in net the deviation cancels and the multiplier is exactly 1,
    which leaves the t=0 match to the closing line intact.
    """
    xgps = priors.get("league_xga_per_shot", 0.095)
    n = len(d)
    gm_h = np.ones(n)
    gm_a = np.ones(n)

    for att, dfn in (("h", "a"), ("a", "h")):
        cur = d[f"{dfn}_gsax_dev"].to_numpy("float64")
        base = d[f"{dfn}_starter_gsax_dev"].to_numpy("float64")
        pulled = d[f"{'away' if dfn == 'a' else 'home'}_goalie_pulled"].to_numpy("int8") == 1
        num = xgps - cur
        den = xgps - base
        m = np.where(np.isfinite(num) & np.isfinite(den) & (den > 1e-6), num / den, 1.0)
        m = np.where(pulled, 1.0, m)   # empty net is handled by the strength layer
        m = np.clip(m, 0.5, 2.0)
        if att == "h":
            gm_h = m
        else:
            gm_a = m

    ch = int((np.abs(gm_h - 1.0) > 1e-9).sum()) + int((np.abs(gm_a - 1.0) > 1e-9).sum())
    print(f"[goalie] {ch:,} of {2 * n:,} team-rows adjusted for a non-starter in net")
    return gm_h, gm_a


def ot_strength_window(d, priors):
    """Rate multipliers and penalty-clock window for rows in overtime."""
    mult = priors.get("ot_strength_multipliers", {})
    n = len(d)
    m_h = np.ones(n)
    m_a = np.ones(n)
    w = np.zeros(n)
    if not mult:
        return m_h, m_a, w

    is_ot = d["is_ot"].to_numpy("int8") == 1
    hs = d["home_skaters_n"].to_numpy("float64")
    aw = d["away_skaters_n"].to_numpy("float64")
    pp = np.maximum(d["home_pp_secs_left"].to_numpy("float64"),
                    d["away_pp_secs_left"].to_numpy("float64"))
    slo = d["secs_left_ot"].to_numpy("float64")
    fallback_tab = priors.get("uneven_mean_remaining_secs", {})
    gap = np.abs(hs - aw)
    fb = np.zeros(n)
    for k, v in fallback_tab.items():
        fb[gap == float(k)] = v
    pp = np.where((pp <= 0) & (fb > 0), fb, pp)
    cand = is_ot & (hs != aw) & (pp > 0) & np.isfinite(hs) & np.isfinite(aw)

    hit = 0
    for i in np.flatnonzero(cand):
        kh = mult.get(ot_state_key(hs[i], aw[i]))
        ka = mult.get(ot_state_key(aw[i], hs[i]))
        if kh is not None and ka is not None:
            m_h[i] = kh
            m_a[i] = ka
            w[i] = min(pp[i], slo[i])
            hit += 1
    print(f"[ot] {hit:,} of {int((is_ot & (hs != aw)).sum()):,} uneven overtime rows adjusted")
    return m_h, m_a, w


def anticipated_en_window(d, priors):
    """Rate multipliers and window length for a goalie pull the model can see
    coming but that has not happened yet. The trailing team's net goes empty,
    so it gains a little and concedes a lot."""
    mult = priors.get("strength_multipliers", {})
    tab = priors.get("expected_en_secs", {})
    n = len(d)
    m_h = np.ones(n)
    m_a = np.ones(n)
    w = np.zeros(n)
    m_own = mult.get("6v5|10")
    m_opp = mult.get("5v6|01")
    if not tab or m_own is None or m_opp is None:
        return m_h, m_a, w

    sd = (d["home_score"].to_numpy("float64") - d["away_score"].to_numpy("float64"))
    slr = d["secs_left_reg"].to_numpy("float64")
    is_ot = d["is_ot"].to_numpy("int8") == 1
    already = ((d["home_goalie_pulled"].to_numpy("int8") == 1)
               | (d["away_goalie_pulled"].to_numpy("int8") == 1))
    # only when the pull decision is actually imminent -- the same window the
    # pull detector uses. Further out, the score changes before anyone pulls.
    horizon = float(priors.get("pull_max_secs", 240.0))
    cand = ~is_ot & ~already & (np.abs(sd) >= 1) & (slr > 0) & (slr <= horizon)

    hit = 0
    for i in np.flatnonzero(cand):
        e = tab.get(en_key(sd[i], slr[i]))
        if not e:
            continue
        w[i] = min(e, slr[i])
        if sd[i] < 0:          # home trailing, home pulls
            m_h[i] = m_own
            m_a[i] = m_opp
        else:                   # home leading, away pulls
            m_h[i] = m_opp
            m_a[i] = m_own
        hit += 1
    print(f"[en] {hit:,} of {int(cand.sum()):,} rows given an anticipated pull "
          f"(mean window {w[w > 0].mean() if (w > 0).any() else 0:.1f}s)")
    return m_h, m_a, w


SO_ATTEMPT_TYPES = ["SHOT", "GOAL", "MISSED_SHOT", "FAILED_SHOT_ATTEMPT"]


def shootout_win_prob(ch, ca, rounds=3):
    """P(home wins the shootout): `rounds` alternating attempts, then
    alternating sudden death. ch = rate home shooters convert, ca = away."""
    ch = np.clip(np.asarray(ch, dtype="float64"), 0.02, 0.9)
    ca = np.clip(np.asarray(ca, dtype="float64"), 0.02, 0.9)
    from math import comb
    ph = np.stack([comb(rounds, k) * ch ** k * (1 - ch) ** (rounds - k)
                   for k in range(rounds + 1)])
    pa = np.stack([comb(rounds, k) * ca ** k * (1 - ca) ** (rounds - k)
                   for k in range(rounds + 1)])
    win = np.zeros_like(ch)
    tie = np.zeros_like(ch)
    for i in range(rounds + 1):
        for j in range(rounds + 1):
            if i > j:
                win += ph[i] * pa[j]
            elif i == j:
                tie += ph[i] * pa[j]
    hw = ch * (1 - ca)
    aw = (1 - ch) * ca
    dec = hw + aw
    return win + tie * np.where(dec > 0, hw / np.maximum(dec, 1e-12), 0.5)


def shootout_league_rates(paths, seasons):
    """League conversion overall, and split by whether the shooter is the home
    team -- that split is what reproduces the observed home shootout win rate
    without bolting on a separate offset."""
    tot = h_n = h_g = a_n = a_g = g_tot = 0.0
    cols = ["period", "event_type", "event_team_abbr", "home_abbr", "season_type"]
    for s in seasons:
        fp = season_pbp_path(paths["pbp_dir"], s)
        if fp is None:
            continue
        d = pd.read_parquet(fp, columns=cols)
        d = d[is_regular(d["season_type"]) & (d["period"] == 5)
              & d["event_type"].isin(SO_ATTEMPT_TYPES)]
        if d.empty:
            continue
        sc = (d["event_type"] == "GOAL").to_numpy()
        hm = (d["event_team_abbr"] == d["home_abbr"]).to_numpy()
        tot += len(d)
        g_tot += sc.sum()
        h_n += hm.sum()
        h_g += sc[hm].sum()
        a_n += (~hm).sum()
        a_g += sc[~hm].sum()
    if tot == 0:
        return 0.318, 0.318, 0.318
    return (float(g_tot / tot), float(h_g / max(h_n, 1)), float(a_g / max(a_n, 1)))


def build_shootout_history(paths, seasons):
    """Per goalie, per shootout: attempts faced and goals allowed, with the
    game date, so a strictly-prior career record can be summed for any game."""
    rows = []
    # NHL game_id is chronological within and across seasons, so it orders a
    # goalie's appearances without needing a date column (which some pbp
    # exports do not carry).
    cols = ["game_id", "period", "event_type", "event_goalie_id", "season_type"]
    for s in seasons:
        fp = season_pbp_path(paths["pbp_dir"], s)
        if fp is None:
            continue
        d = pd.read_parquet(fp, columns=cols)
        d = d[is_regular(d["season_type"]) & (d["period"] == 5)
              & d["event_type"].isin(SO_ATTEMPT_TYPES) & d["event_goalie_id"].notna()]
        if d.empty:
            continue
        d = d.copy()
        d["scored"] = (d["event_type"] == "GOAL").astype(float)
        g = d.groupby(["event_goalie_id", "game_id"], as_index=False).agg(
            att=("scored", "size"), ga=("scored", "sum"))
        rows.append(g)
    if not rows:
        return pd.DataFrame(columns=["goalie_id", "game_id", "att", "ga"])
    h = pd.concat(rows, ignore_index=True)
    h = h.rename(columns={"event_goalie_id": "goalie_id"})
    h["goalie_id"] = h["goalie_id"].astype("int64")
    h["game_id"] = h["game_id"].astype("int64")
    return h.sort_values(["goalie_id", "game_id"]).reset_index(drop=True)


class ShootoutPrior:
    """Career shootout record strictly before a given date."""

    def __init__(self, hist):
        self.by_goalie = {}
        for gid, grp in hist.groupby("goalie_id"):
            keys = grp["game_id"].to_numpy("int64")
            att = np.cumsum(grp["att"].to_numpy("float64"))
            ga = np.cumsum(grp["ga"].to_numpy("float64"))
            self.by_goalie[int(gid)] = (keys, att, ga)

    def prior(self, goalie_ids, game_ids):
        gi = np.asarray(goalie_ids, dtype="float64")
        dt = np.asarray(game_ids, dtype="int64")
        att = np.zeros(len(gi))
        ga = np.zeros(len(gi))
        for gid in np.unique(gi[np.isfinite(gi)]):
            rec = self.by_goalie.get(int(gid))
            if rec is None:
                continue
            m = gi == gid
            d_, a_, g_ = rec
            pos = np.searchsorted(d_, dt[m], side="left")   # strictly before
            att[m] = np.where(pos > 0, a_[np.clip(pos - 1, 0, len(a_) - 1)], 0.0)
            ga[m] = np.where(pos > 0, g_[np.clip(pos - 1, 0, len(g_) - 1)], 0.0)
        return att, ga


def attach_shootout(d, so_prior, priors, args):
    """Per-row P(home wins a shootout) from the two goalies currently in net."""
    lg = priors.get("so_league_rate", 0.318)
    lg_h = priors.get("so_home_shooter_rate", lg)
    lg_a = priors.get("so_away_shooter_rate", lg)
    k = args.shootout_prior_attempts

    keys = d["game_id"].to_numpy("int64")
    out = {}
    for side, col in (("h", "home_goalie_id"), ("a", "away_goalie_id")):
        att, ga = so_prior.prior(d[col].to_numpy("float64"), keys)
        allowed = (ga + k * lg) / (att + k)      # shrunk toward league
        out[side] = (allowed, att)
        d[f"{side}_so_allowed"] = allowed
        d[f"{side}_so_att"] = att

    # home shooters face the AWAY goalie, and vice versa
    ch = np.clip(lg_h + (out["a"][0] - lg), 0.05, 0.75)
    ca = np.clip(lg_a + (out["h"][0] - lg), 0.05, 0.75)
    d["p_shootout_home"] = shootout_win_prob(ch, ca)
    print(f"[so] p_shootout_home mean {d['p_shootout_home'].mean():.4f} "
          f"sd {d['p_shootout_home'].std():.4f} "
          f"range {d['p_shootout_home'].min():.3f}-{d['p_shootout_home'].max():.3f}")
    return d


def strength_window(d, priors):
    """Current-state rate multipliers and how long the state is known to last.

    Empty net: the pull is a late-game decision, so the state is assumed to run
    to the end of regulation. Uneven strength: the tracked penalty clock.
    Even strength: no adjustment (window 0), which leaves the pregame
    calibration at t=0 untouched.
    """
    mult = priors.get("strength_multipliers", {})
    n = len(d)
    m_h = np.ones(n)
    m_a = np.ones(n)

    hs = d["home_skaters_n"].to_numpy("float64")
    aw = d["away_skaters_n"].to_numpy("float64")
    he = d["home_goalie_pulled"].to_numpy("int8") == 1
    ae = d["away_goalie_pulled"].to_numpy("int8") == 1
    is_ot = d["is_ot"].to_numpy("int8") == 1

    empty = he | ae
    uneven = (hs != aw) | empty
    pp_w = np.maximum(d["home_pp_secs_left"].to_numpy("float64"),
                      d["away_pp_secs_left"].to_numpy("float64"))

    # the skater counts are ground truth for WHETHER a power play is on; the
    # penalty tracker only estimates how long is left. Where they disagree,
    # use the measured mean duration for that skater differential.
    fallback_tab = priors.get("uneven_mean_remaining_secs", {})
    sk_gap = np.abs(hs - aw)
    fb = np.zeros(len(d))
    for k, v in fallback_tab.items():
        fb[sk_gap == float(k)] = v
    # A team with 6 skaters and no pull flag is a delayed penalty, which lasts
    # seconds, not a penalty-clock state. Applying the measured penalty
    # duration there would hold a large multiplier for over a minute.
    penalty_state = (np.maximum(hs, aw) <= 5)
    gap_no_clock = (hs != aw) & ~empty & penalty_state & (pp_w <= 0) & (fb > 0)
    pp_w = np.where(gap_no_clock, fb, pp_w)
    n_fb = int(gap_no_clock.sum())

    w = np.where(empty, d["secs_left_reg"].to_numpy("float64"), np.where(hs != aw, pp_w, 0.0))
    w = np.where(is_ot, 0.0, w)

    hit = 0
    for i in np.flatnonzero(uneven & ~is_ot & (w > 0) & np.isfinite(hs) & np.isfinite(aw)):
        kh = state_key(hs[i], aw[i], he[i], ae[i])
        ka = state_key(aw[i], hs[i], ae[i], he[i])
        if kh in mult and ka in mult:
            m_h[i] = mult[kh]
            m_a[i] = mult[ka]
            hit += 1
    tot = int((uneven & ~is_ot).sum())
    print(f"[state] {hit:,} of {tot:,} uneven-strength rows matched a rate multiplier "
          f"({n_fb:,} used the measured-duration fallback)")
    return m_h, m_a, w


def attach_running(d):
    gid = d["game_id"].to_numpy("int64")
    et = d["event_type"].to_numpy(object)
    is_home_ev = (d["event_team_abbr"].to_numpy(object) == d["home_abbr"].to_numpy(object))

    xg = pd.to_numeric(d["xg"], errors="coerce").fillna(0.0).to_numpy("float64")
    fen = np.isin(et, ["SHOT", "MISSED_SHOT", "GOAL"])
    sog = np.isin(et, ["SHOT", "GOAL"])
    blk = et == "BLOCKED_SHOT"
    hit = et == "HIT"
    gv = et == "GIVEAWAY"
    tk = et == "TAKEAWAY"
    fo = et == "FACEOFF"
    pen = et == "PENALTY"
    is_shot_ev = np.isin(et, ["SHOT", "MISSED_SHOT", "GOAL"])

    def cs(mask_home):
        return pd.Series(mask_home.astype("float64")).groupby(gid).cumsum().to_numpy()

    d["sog_h"] = cs(sog & is_home_ev)
    d["sog_a"] = cs(sog & ~is_home_ev)
    d["fen_h"] = cs(fen & is_home_ev)
    d["fen_a"] = cs(fen & ~is_home_ev)
    d["blk_h"] = cs(blk & is_home_ev)
    d["blk_a"] = cs(blk & ~is_home_ev)
    d["hit_h"] = cs(hit & is_home_ev)
    d["hit_a"] = cs(hit & ~is_home_ev)
    d["gv_h"] = cs(gv & is_home_ev)
    d["gv_a"] = cs(gv & ~is_home_ev)
    d["tk_h"] = cs(tk & is_home_ev)
    d["tk_a"] = cs(tk & ~is_home_ev)
    d["fow_h"] = cs(fo & is_home_ev)
    d["fow_a"] = cs(fo & ~is_home_ev)
    d["pen_h"] = cs(pen & is_home_ev)
    d["pen_a"] = cs(pen & ~is_home_ev)

    d["xgf_h"] = pd.Series(np.where(is_shot_ev & is_home_ev, xg, 0.0)).groupby(gid).cumsum().to_numpy()
    d["xgf_a"] = pd.Series(np.where(is_shot_ev & ~is_home_ev, xg, 0.0)).groupby(gid).cumsum().to_numpy()

    elapsed = np.maximum(d["abs_seconds"].to_numpy("float64"), 60.0)
    per60 = 3600.0 / elapsed

    d["sog_diff_run"] = d["sog_h"] - d["sog_a"]
    d["fen_diff_run"] = d["fen_h"] - d["fen_a"]
    d["hit_diff_run"] = d["hit_h"] - d["hit_a"]
    d["gv_diff_run"] = d["gv_h"] - d["gv_a"]
    d["tk_diff_run"] = d["tk_h"] - d["tk_a"]
    d["pen_diff_run"] = d["pen_h"] - d["pen_a"]
    d["xg_diff_run"] = d["xgf_h"] - d["xgf_a"]
    d["xg_total_run"] = d["xgf_h"] + d["xgf_a"]
    d["sog_h_per60"] = d["sog_h"] * per60
    d["sog_a_per60"] = d["sog_a"] * per60
    d["xgf_h_per60"] = d["xgf_h"] * per60
    d["xgf_a_per60"] = d["xgf_a"] * per60
    tot_fen = d["fen_h"] + d["fen_a"]
    d["fen_share_h"] = np.where(tot_fen > 0, d["fen_h"] / np.maximum(tot_fen, 1), 0.5)
    tot_xg = d["xgf_h"] + d["xgf_a"]
    d["xg_share_h"] = np.where(tot_xg > 0, d["xgf_h"] / np.maximum(tot_xg, 1e-6), 0.5)
    tot_fo = d["fow_h"] + d["fow_a"]
    d["fo_share_h"] = np.where(tot_fo > 0, d["fow_h"] / np.maximum(tot_fo, 1), 0.5)
    d["h_shoot_luck"] = d["home_score"] - d["xgf_h"]
    d["a_shoot_luck"] = d["away_score"] - d["xgf_a"]
    d["luck_diff"] = d["h_shoot_luck"] - d["a_shoot_luck"]
    # power-play opportunities so far (a penalty on the away team = home PP)
    d["pp_opp_h_run"] = d["pen_a"]
    d["pp_opp_a_run"] = d["pen_h"]
    return d


def attach_team_ytd(d, ty):
    ty = ty.copy()
    home = ty.rename(columns={c: f"h_{c}" for c in TEAM_DIFF_COLS + ["gp_prior"]})
    away = ty.rename(columns={c: f"a_{c}" for c in TEAM_DIFF_COLS + ["gp_prior"]})
    d = d.merge(
        home.drop(columns=["team"]).assign(team=ty["team"]),
        left_on=["game_id", "home_abbr"], right_on=["game_id", "team"], how="left",
    ).drop(columns=["team"], errors="ignore")
    d = d.merge(
        away.drop(columns=["team"]).assign(team=ty["team"]),
        left_on=["game_id", "away_abbr"], right_on=["game_id", "team"], how="left",
    ).drop(columns=["team"], errors="ignore")
    for c in TEAM_DIFF_COLS:
        d[f"tdiff_{c}"] = d[f"h_{c}"] - d[f"a_{c}"]
    d["team_gp_prior"] = d[["h_gp_prior", "a_gp_prior"]].min(axis=1)
    return d


def attach_goalies(d, gy, priors, args):
    gq = goalie_quality(gy, args.goalie_shot_prior,
                        priors.get("league_gsax_per_shot", 0.0))
    cols = [c for c in gq.columns if c not in ("game_id", "goalie_id")]
    lk = PlayerLookup(gq, "goalie_id", cols)

    gid = d["game_id"].to_numpy("int64")
    for side, idcol in (("h", "home_goalie_id"), ("a", "away_goalie_id")):
        vals, hit = lk.get(gid, d[idcol].to_numpy("float64"))
        for j, c in enumerate(cols):
            d[f"{side}_{c}"] = vals[:, j]
        d[f"{side}_goalie_known"] = hit.astype(np.int8)

    # starter = goalie on the ice at the opening faceoff
    first = d.groupby("game_id").head(1)[["game_id", "home_goalie_id", "away_goalie_id"]]
    first = first.rename(columns={"home_goalie_id": "h_starter_id", "away_goalie_id": "a_starter_id"})
    d = d.merge(first, on="game_id", how="left")
    d["h_is_starter"] = (d["home_goalie_id"] == d["h_starter_id"]).astype(np.int8)
    d["a_is_starter"] = (d["away_goalie_id"] == d["a_starter_id"]).astype(np.int8)

    # the closing line already prices the STARTER, so only the deviation of
    # whoever is actually in net belongs in the rates
    gid = d["game_id"].to_numpy("int64")
    for side, col in (("h", "h_starter_id"), ("a", "a_starter_id")):
        vals, hit = lk.get(gid, d[col].to_numpy("float64"))
        d[f"{side}_starter_gsax_dev"] = np.where(hit, vals[:, cols.index("gsax_dev")], np.nan)

    d["g_gsax_diff"] = d["h_gsax_per_shot"].fillna(0.0) - d["a_gsax_per_shot"].fillna(0.0)
    d["g_vs_repl_diff"] = d["h_gsax_vs_repl"].fillna(0.0) - d["a_gsax_vs_repl"].fillna(0.0)
    d["g_svpct_diff"] = d["h_g_sv_pct"] - d["a_g_sv_pct"]
    d["g_hd_svpct_diff"] = d["h_g_hd_sv_pct"] - d["a_g_hd_sv_pct"]
    return d.drop(columns=["h_starter_id", "a_starter_id"], errors="ignore")


def attach_skaters(d, sy, args):
    lk = PlayerLookup(sy, "player_id", SKATER_AGG_COLS)
    gid = d["game_id"].to_numpy("int64")
    n = len(d)

    for side in ("home", "away"):
        acc = np.zeros((n, len(SKATER_AGG_COLS)))
        cnt = np.zeros(n)
        toi_w = np.zeros((n, len(SKATER_AGG_COLS)))
        wsum = np.zeros(n)
        for slot in range(1, 8):
            col = f"{side}_on_{slot}_id"
            vals, hit = lk.get(gid, d[col].to_numpy("float64"))
            good = hit & np.isfinite(vals).all(axis=1)
            v = np.where(good[:, None], vals, 0.0)
            acc += v
            cnt += good
            w = np.where(good, np.nan_to_num(vals[:, SKATER_AGG_COLS.index("toi_per_gp")]), 0.0)
            toi_w += v * w[:, None]
            wsum += w
        pre = "h" if side == "home" else "a"
        cnt_s = np.maximum(cnt, 1)
        for j, c in enumerate(SKATER_AGG_COLS):
            d[f"{pre}_ice_{c}"] = np.where(cnt > 0, acc[:, j] / cnt_s, np.nan)
            d[f"{pre}_icew_{c}"] = np.where(wsum > 0, toi_w[:, j] / np.maximum(wsum, 1e-9), np.nan)
        d[f"{pre}_ice_n"] = cnt
        d[f"{pre}_ice_sum_ixg60"] = np.where(cnt > 0, acc[:, SKATER_AGG_COLS.index("ixg_per60")], np.nan)

    for c in SKATER_AGG_COLS:
        d[f"ice_diff_{c}"] = d[f"h_ice_{c}"] - d[f"a_ice_{c}"]
        d[f"icew_diff_{c}"] = d[f"h_icew_{c}"] - d[f"a_icew_{c}"]

    # roster strength: every skater who dressed in this game, toi-weighted
    sy2 = sy.copy()
    sy2["w"] = sy2["toi_per_gp"].fillna(0.0)
    for c in ("on_xgf_per60", "on_xga_per60", "xgf_pct_rel", "p_per60"):
        sy2[f"w_{c}"] = sy2[c] * sy2["w"]
    grp = sy2.groupby("game_id", as_index=False).agg(
        {**{f"w_{c}": "sum" for c in ("on_xgf_per60", "on_xga_per60", "xgf_pct_rel", "p_per60")},
         "w": "sum"}
    )
    for c in ("on_xgf_per60", "on_xga_per60", "xgf_pct_rel", "p_per60"):
        grp[f"game_roster_{c}"] = grp[f"w_{c}"] / np.maximum(grp["w"], 1e-9)
    d = d.merge(
        grp[["game_id"] + [f"game_roster_{c}" for c in
                           ("on_xgf_per60", "on_xga_per60", "xgf_pct_rel", "p_per60")]],
        on="game_id", how="left",
    )
    return d


# ----------------------------------------------------------------------------
# column manifests
# ----------------------------------------------------------------------------

META_COLS = [
    "game_id", "espn_game_id", "season", "game_date", "home_abbr", "away_abbr",
    "event_idx", "event_type", "period", "period_seconds", "abs_seconds",
    "home_score", "away_score", "wallclock_unix", "wallclock_extrapolated",
    "p_home_close", "total_close", "has_odds",
]

FEATURES = (
    [
        # clock
        "secs_left_reg", "secs_left_ot", "frac_left", "is_ot", "period",
        "abs_seconds",
        # score
        "score_diff", "abs_score_diff", "is_tied", "total_goals",
        "score_diff_x_fracleft", "score_diff_per_sqrt_left",
        # strength / nets
        "skater_diff", "home_skaters_n", "away_skaters_n", "is_even",
        "home_pp", "away_pp", "is_3v3",
        "home_goalie_pulled", "away_goalie_pulled", "net_empty_diff",
        "home_pp_secs_left", "away_pp_secs_left", "pp_secs_diff",
        "state_mult_home", "state_mult_away", "state_window_secs",
        "goalie_mult_home", "goalie_mult_away",
        "p_shootout_home", "h_so_allowed", "a_so_allowed", "h_so_att", "a_so_att",
        "ot_mult_home", "ot_mult_away", "ot_window_secs",
        "h_gsax_dev", "a_gsax_dev", "h_starter_gsax_dev", "a_starter_gsax_dev",
        # market / baseline
        "baseline_wp", "baseline_logit", "logit_close", "total_close_f",
        "p_over_close", "lam_total", "lam_ratio", "has_odds",
        "logit_close_x_fracleft", "logit_close_x_sqrtleft",
        # in-game flow
        "sog_diff_run", "fen_diff_run", "hit_diff_run", "gv_diff_run",
        "tk_diff_run", "pen_diff_run", "xg_diff_run", "xg_total_run",
        "sog_h_per60", "sog_a_per60", "xgf_h_per60", "xgf_a_per60",
        "fen_share_h", "xg_share_h", "fo_share_h",
        "h_shoot_luck", "a_shoot_luck", "luck_diff",
        "pp_opp_h_run", "pp_opp_a_run", "xg_diff_minus_score_diff",
        # goalies
        "h_gsax_per_shot", "a_gsax_per_shot", "g_gsax_diff",
        "h_gsax_vs_repl", "a_gsax_vs_repl", "g_vs_repl_diff",
        "h_g_sv_pct", "a_g_sv_pct", "g_svpct_diff",
        "h_g_hd_sv_pct", "a_g_hd_sv_pct", "g_hd_svpct_diff",
        "h_g_sv_pct_5v5", "a_g_sv_pct_5v5",
        "h_g_gp_prior", "a_g_gp_prior", "h_g_sa_prior", "a_g_sa_prior",
        "h_is_starter", "a_is_starter", "h_goalie_known", "a_goalie_known",
        "h_team_repl_gsax_per_shot", "a_team_repl_gsax_per_shot",
        # team ytd
        "team_gp_prior",
    ]
    + [f"tdiff_{c}" for c in TEAM_DIFF_COLS]
    + [f"h_{c}" for c in ("xgf_pct_5v5", "cf_pct_5v5", "pdo_5v5", "pp_pct", "pk_pct",
                          "gf_minus_xgf", "ga_minus_xga", "xgf_per60", "xga_per60")]
    + [f"a_{c}" for c in ("xgf_pct_5v5", "cf_pct_5v5", "pdo_5v5", "pp_pct", "pk_pct",
                          "gf_minus_xgf", "ga_minus_xga", "xgf_per60", "xga_per60")]
    # on-ice skaters
    + [f"h_ice_{c}" for c in SKATER_AGG_COLS]
    + [f"a_ice_{c}" for c in SKATER_AGG_COLS]
    + [f"ice_diff_{c}" for c in SKATER_AGG_COLS]
    + [f"icew_diff_{c}" for c in SKATER_AGG_COLS]
    + ["h_ice_n", "a_ice_n", "h_ice_sum_ixg60", "a_ice_sum_ixg60"]
    + [f"game_roster_{c}" for c in ("on_xgf_per60", "on_xga_per60", "xgf_pct_rel", "p_per60")]
)

FEATURES = list(dict.fromkeys(FEATURES))


# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--seasons", nargs="+", type=int,
                    default=[2021, 2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--prior-seasons", nargs="+", type=int,
                    default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--rebuild-priors", action="store_true")
    ap.add_argument("--goalie-shot-prior", type=float, default=300.0)
    ap.add_argument("--default-home-p", type=float, default=0.54)
    ap.add_argument("--chunk-rows", type=int, default=400_000)
    ap.add_argument("--strength-min-minutes", type=float, default=30.0)
    ap.add_argument("--strength-prior-goals", type=float, default=5.0)
    ap.add_argument("--pull-max-secs", type=float, default=240.0)
    ap.add_argument("--ot-strength-min-minutes", type=float, default=20.0)
    ap.add_argument("--shootout-prior-attempts", type=float, default=66.0)
    args = ap.parse_args()

    paths = resolve_paths(args.root)
    print(f"[paths] {json.dumps(paths, indent=2)}")

    priors_fp = os.path.join(paths["out_dir"], "nhl_league_priors_v10.json")
    priors = build_league_priors(paths, args.prior_seasons, priors_fp,
                                 args.rebuild_priors, args.strength_min_minutes,
                                 args.strength_prior_goals, args.pull_max_secs,
                                 args.ot_strength_min_minutes)
    print(f"[priors] ot_mult={priors['ot_rate_multiplier']:.3f} "
          f"p_so_home={priors['p_shootout_home']:.4f} "
          f"league_gpg={priors['league_goals_per_game']:.3f}")

    odds = load_odds(paths)
    stats = load_stats(paths, args.seasons)

    hist = build_shootout_history(paths, sorted(set(args.seasons) | set(args.prior_seasons)))
    so_prior = ShootoutPrior(hist)
    print(f"[so] shootout history: {len(hist):,} goalie-games, "
          f"{hist['att'].sum():,.0f} attempts, "
          f"{hist['goalie_id'].nunique():,} goalies")

    for s in args.seasons:
        build_season(s, paths, odds, stats, priors, so_prior, args)

    print(f"\n[done] {len(FEATURES)} feature columns")


if __name__ == "__main__":
    main()
