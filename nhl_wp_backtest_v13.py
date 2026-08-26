#!/usr/bin/env python
"""
nhl_wp_backtest_v13.py
======================
Backtest the NHL live WP model against the Kalshi trade tape.

Mechanics
---------
  * every play-by-play event with a wallclock timestamp is a decision point
  * anchor = wallclock + --lag (default 10s); fill = the FIRST printed trade in
    (anchor, anchor + --window] (default 5s)
  * both sides are priced off that single print: home costs
    home_yes_price_cents, away costs 101 - home_yes_price_cents
  * size = edge-scaled. The edge of every qualifying bet is converted to a
    z-score across the bet book, multiplier = 1 + --size-slope * z clipped to
    [--size-min, --size-max], then the book is rescaled so the AVERAGE bet is
    exactly --contracts (default 100). Bigger edge -> bigger bet, same average
    size as flat sizing. --size-slope 0 restores flat sizing. Sizes are NOT
    capped at the printed size: the book is deep and quotes a cent wide, so a
    small print reflects what that taker wanted, not the depth available.
    --cap-to-print restores the old behaviour.
  * Kalshi taker fee = ceil(0.07 * C * P * (1-P)) rounded up to the cent
  * bet is taken when net edge after fees >= --min-edge (default 0.02)

Usage
-----
python nhl_wp_backtest_v14.py --seasons 2026 --tag v1
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_ROOT = os.environ.get(
    "NHL_ROOT", r"C:\Users\saint\OneDrive\Documents\NHL_AUG_2026\data"
)

FEE_RATE = 0.07

# Kalshi / pbp abbreviation variants -> canonical pbp code
TEAM_ALIAS = {
    "TB": "TBL", "TAM": "TBL", "TBA": "TBL",
    "SJ": "SJS", "SAN": "SJS",
    "LA": "LAK", "LOS": "LAK",
    "NJ": "NJD", "NJY": "NJD", "NEW": "NJD",
    "WAS": "WSH",
    "CLS": "CBJ", "CBS": "CBJ",
    "MON": "MTL", "MON.": "MTL",
    "VGS": "VGK", "LV": "VGK", "VEG": "VGK",
    "WPG": "WPG", "WIN": "WPG",
    "CGY": "CGY", "CAL": "CGY",
    "ANA": "ANA", "ANH": "ANA",
    "PHX": "ARI", "ARZ": "ARI",
    "UTA": "UTA", "UTAH": "UTA",
    "SEA": "SEA",
    "FLA": "FLA", "FLO": "FLA",
    "NAS": "NSH",
    "COL": "COL", "COLO": "COL",
    "EDM": "EDM", "EDO": "EDM",
}


def canon(x):
    s = pd.Series(x, dtype="object").astype(str).str.strip().str.upper()
    return s.map(lambda v: TEAM_ALIAS.get(v, v))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def apply_calibrator(cal, p, z):
    k = cal["kind"]
    if k == "raw":
        return p
    if k == "platt":
        return sigmoid(cal["a"] * z + cal["b"])
    if k == "isotonic":
        return np.interp(p, np.array(cal["x"]), np.array(cal["y"]))
    raise ValueError(k)


def fee_dollars(contracts, price):
    """Kalshi taker fee, rounded up to the cent."""
    raw = FEE_RATE * contracts * price * (1.0 - price)
    return np.ceil(raw * 100.0 - 1e-9) / 100.0


SLOPE_BOUNDS = (0.2, 3.0)


def edge_scaled_size(edge, target_avg, slope, lo, hi):
    """
    Contracts per bet, scaled by how large the edge is relative to the rest of
    the bet book, with the mean pinned to target_avg.

    z          = (edge - mean(edge)) / sd(edge)   across all qualifying bets
    multiplier = clip(1 + slope * z, lo, hi)
    contracts  = round(target_avg * multiplier / mean(multiplier)), floor of 1

    Rounding and the floor of 1 both push the realised mean off target, so the
    scale factor is solved for in a short loop instead of applied once.
    Returns (contracts, z).
    """
    e = np.asarray(edge, dtype="float64")
    sd = float(np.std(e))
    if slope == 0.0 or not np.isfinite(sd) or sd <= 0.0:
        z = np.zeros_like(e)
    else:
        z = (e - float(np.mean(e))) / sd

    mult = np.clip(1.0 + slope * z, lo, hi)
    m = float(np.mean(mult))
    if not np.isfinite(m) or m <= 0.0:
        mult = np.ones_like(e)
        m = 1.0
    mult = mult / m

    scale = float(target_avg)
    contracts = np.maximum(1.0, np.rint(scale * mult))
    for _ in range(40):
        realised = float(np.mean(contracts))
        if realised <= 0 or abs(realised - target_avg) < 1e-9:
            break
        scale *= target_avg / realised
        new = np.maximum(1.0, np.rint(scale * mult))
        if np.array_equal(new, contracts):
            break
        contracts = new
    return contracts, z


# ----------------------------------------------------------------------------


def _irls(X, y, ridge=1.0, target=None):
    """Ridge-penalised logistic regression, Newton steps, any design matrix."""
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")
    k = X.shape[1]
    beta = np.zeros(k) if target is None else np.array(target, dtype="float64")
    tgt = np.zeros(k) if target is None else np.array(target, dtype="float64")
    converged = False
    for _ in range(100):
        lin = np.clip(X @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-lin))
        w = np.maximum(p * (1 - p), 1e-9)
        g = X.T @ (y - p) - ridge * (beta - tgt)
        H = (X * w[:, None]).T @ X + ridge * np.eye(k)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -1.0, 1.0)
        beta += step
        if not np.all(np.isfinite(beta)):
            break
        if np.max(np.abs(step)) < 1e-10:
            converged = True
            break
    return beta, bool(converged and np.all(np.isfinite(beta)))


def tail_design(z):
    """z, z*|z|, 1 -- so the effective slope is a + c*|z| and can differ at the
    tails from the middle without introducing bin edges."""
    z = np.asarray(z, dtype="float64")
    return np.column_stack([z, z * np.abs(z), np.ones_like(z)])


def fit_tail_shrink(z, y, a_global, b_global):
    beta, ok = _irls(tail_design(z), y, target=[1.0, 0.0, 0.0])
    if not ok:
        print("[tails] fit did not converge, using the global shrink")
        return None
    a, c, b = beta
    zmax = float(np.percentile(np.abs(z), 99.9))
    # the mapping must stay monotone in z over the range we actually see
    if a + 2 * c * zmax <= 0 or not (SLOPE_BOUNDS[0] <= a <= SLOPE_BOUNDS[1]):
        print(f"[tails] rejected: slope {a:.3f}, curvature {c:+.4f} would be "
              f"non-monotone by |z|={zmax:.1f}; using the global shrink")
        return None
    print(f"[tails] slope {a:.4f}  curvature {c:+.5f}  intercept {b:+.4f}")
    print("[tails] effective slope by distance from even money:")
    for zz in (0.5, 1.0, 2.0, 3.0, 4.0):
        pp = 1.0 / (1.0 + np.exp(-zz))
        print(f"          |logit|={zz:.1f} (p={pp:.3f} / {1 - pp:.3f}): "
              f"{a + c * zz:.4f}   [global {a_global:.4f}]")
    return float(a), float(c), float(b)


def _newton_platt(z, y, ridge=1.0):
    """Ridge-penalised logistic fit of y on the baseline logit.

    Without the penalty this diverges wherever the two classes are perfectly
    separated -- e.g. a three-goal lead with a minute left, where the sign of
    the logit predicts the winner every time and the likelihood is maximised by
    sending the slope to infinity. Returns converged=False if it still runs
    away, so the caller can fall back.
    """
    z = np.asarray(z, dtype="float64")
    y = np.asarray(y, dtype="float64")
    a, b = 1.0, 0.0
    converged = False
    for _ in range(100):
        lin = np.clip(a * z + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-lin))
        w = np.maximum(p * (1 - p), 1e-9)
        r = y - p
        # penalty pulls the slope toward 1 and the intercept toward 0
        g = np.array([np.sum(r * z) - ridge * (a - 1.0), np.sum(r) - ridge * b])
        H = np.array([[np.sum(w * z * z) + ridge, np.sum(w * z)],
                      [np.sum(w * z), np.sum(w) + ridge]])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -1.0, 1.0)
        a += step[0]
        b += step[1]
        if not (np.isfinite(a) and np.isfinite(b)):
            break
        if np.max(np.abs(step)) < 1e-10:
            converged = True
            break
    ok = (converged and np.isfinite(a) and np.isfinite(b)
          and SLOPE_BOUNDS[0] <= a <= SLOPE_BOUNDS[1] and abs(b) <= 2.0)
    return float(a), float(b), bool(ok)


def _ll(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


TIME_EDGES = [0.0, 120.0, 300.0, 600.0, 1200.0, 2400.0, 3600.1]
TIME_LABELS = ["0-2m", "2-5m", "5-10m", "10-20m", "20-40m", "40-60m"]


def state_cell(score_diff, secs_left_reg, period):
    """(|score differential|, time left in regulation), with overtime separate."""
    sd = np.minimum(np.abs(np.asarray(score_diff, dtype="float64")), 3).astype(int)
    slr = np.asarray(secs_left_reg, dtype="float64")
    per = np.asarray(period, dtype="float64")
    tb = np.digitize(slr, TIME_EDGES[1:-1], right=False)
    lab = np.array(TIME_LABELS, dtype=object)[np.clip(tb, 0, len(TIME_LABELS) - 1)]
    out = np.array([f"sd{a}|{b}" for a, b in zip(sd, lab)], dtype=object)
    return np.where(per >= 4, "OT", out)


def period_bucket(period):
    p = np.asarray(period, dtype="float64")
    return np.where(p >= 4, 4, p).astype("int64")


def fit_base_shrink(root, seasons, min_rows=5000, prior_rows=50000.0,
                    want_tails=False):
    """Single-parameter (slope + intercept) recalibration of the analytic
    baseline, fitted on seasons the backtest never sees. Slope < 1 means the
    baseline is overconfident and its logits need pulling toward the middle."""
    fdir = os.path.join(root, "features")
    zs, ys, ps, cs = [], [], [], []
    for s in seasons:
        fp = os.path.join(fdir, f"nhl_wp_features_{s}.parquet")
        if not os.path.exists(fp):
            print(f"[warn] missing {fp}")
            continue
        d = pd.read_parquet(fp, columns=["baseline_logit", "home_won", "period",
                                         "score_diff", "secs_left_reg"])
        zs.append(d["baseline_logit"].to_numpy("float64"))
        ys.append(d["home_won"].to_numpy("float64"))
        ps.append(period_bucket(d["period"].to_numpy("float64")))
        cs.append(state_cell(d["score_diff"], d["secs_left_reg"], d["period"]))
    if not zs:
        sys.exit("ERROR: no feature files for --base-calib-seasons")
    z = np.concatenate(zs)
    y = np.concatenate(ys)
    per = np.concatenate(ps)
    cell = np.concatenate(cs)
    ok = np.isfinite(z)
    z, y, per, cell = z[ok], y[ok], per[ok], cell[ok]

    a, b, _ = _newton_platt(z, y)
    raw = 1.0 / (1.0 + np.exp(-z))
    cal = 1.0 / (1.0 + np.exp(-(a * z + b)))
    print(f"[shrink] fitted on {seasons} n={len(z):,}  slope={a:.4f} intercept={b:+.4f}")
    print(f"[shrink] in-sample logloss raw {_ll(y, raw):.5f} -> shrunk {_ll(y, cal):.5f}")

    by_period = {}
    for pb in sorted(np.unique(per)):
        m = per == pb
        if m.sum() < 5000:
            print(f"[shrink] period {pb}: only {m.sum():,} rows, using the global fit")
            by_period[int(pb)] = (a, b)
            continue
        pa, pbi, pok = _newton_platt(z[m], y[m])
        if not pok:
            pa, pbi = a, b
        cp = 1.0 / (1.0 + np.exp(-(pa * z[m] + pbi)))
        lbl = "OT" if pb == 4 else str(pb)
        print(f"[shrink] period {lbl}: n={m.sum():,} slope={pa:.4f} intercept={pbi:+.4f} "
              f"logloss {_ll(y[m], 1.0 / (1.0 + np.exp(-z[m]))):.5f} -> {_ll(y[m], cp):.5f}")
        by_period[int(pb)] = (float(pa), float(pbi))

    tails = None
    if want_tails:
        tails = fit_tail_shrink(z, y, a, b)
        if tails is not None:
            ta, tc, tb = tails
            zt = np.clip(ta * z + tc * z * np.abs(z) + tb, -30, 30)
            print(f"[tails] in-sample logloss global {_ll(y, 1 / (1 + np.exp(-(a * z + b)))):.5f}"
                  f" -> tail-aware {_ll(y, 1 / (1 + np.exp(-zt))):.5f}")

    by_cell = {}
    rows = []
    for c in sorted(set(cell.tolist())):
        m = cell == c
        n = int(m.sum())
        if n < min_rows:
            by_cell[c] = (a, b)
            rows.append((c, n, a, b, True))
            continue
        ca, cb, cok = _newton_platt(z[m], y[m])
        if not cok:
            by_cell[c] = (a, b)
            rows.append((c, n, a, b, True))
            continue
        # pull thin cells toward the global fit
        wgt = n / (n + prior_rows)
        fa = wgt * ca + (1 - wgt) * a
        fb = wgt * cb + (1 - wgt) * b
        by_cell[c] = (float(fa), float(fb))
        rows.append((c, n, fa, fb, False))
    tab = pd.DataFrame(rows, columns=["cell", "n", "slope", "intercept", "global_fallback"])
    print("\n[shrink] per-state fits (blended toward global):")
    print(tab.to_string(index=False, formatters={
        "n": "{:,.0f}".format, "slope": "{:.4f}".format, "intercept": "{:+.4f}".format}))
    return float(a), float(b), by_period, by_cell, tails


def load_model(root, tag):
    mdir = os.path.join(root, "models")
    meta_fp = os.path.join(mdir, f"nhl_wp_{tag}_meta.json")
    if not os.path.exists(meta_fp):
        sys.exit(f"ERROR: no model meta at {meta_fp}")
    with open(meta_fp) as fh:
        meta = json.load(fh)
    models = []
    for i, fn in enumerate(meta["members"]):
        b = xgb.Booster()
        b.load_model(os.path.join(mdir, fn))
        models.append((b, meta["best_iterations"][i]))
    return meta, models


def predict(df, meta, models):
    X = df[meta["features"]]
    dm = xgb.DMatrix(X, feature_names=meta["features"])
    if meta["base_margin"] == "baseline":
        dm.set_base_margin(df["baseline_logit"].to_numpy("float64"))
    zs = []
    for b, it in models:
        zs.append(logit(b.predict(dm, iteration_range=(0, it + 1))))
    z = np.mean(zs, axis=0)
    return apply_calibrator(meta["calibrator"], sigmoid(z), z)


def load_kalshi(root):
    kdir = os.path.join(root, "kalshi")
    cands = sorted(glob.glob(os.path.join(kdir, "*.parquet")))
    if cands:
        fp = cands[0]
        t = pd.read_parquet(fp)
    else:
        cands = sorted(glob.glob(os.path.join(kdir, "*.csv")))
        if not cands:
            sys.exit(f"ERROR: no Kalshi trade file in {kdir}")
        fp = cands[0]
        t = pd.read_csv(fp)
    print(f"[kalshi] {os.path.basename(fp)}: {len(t):,} trades")
    need = {"game_date", "home_team", "away_team", "timestamp_unix",
            "home_yes_price_cents", "count_fp"}
    missing = need - set(t.columns)
    if missing:
        sys.exit(f"ERROR: Kalshi file missing columns {sorted(missing)}")
    t["timestamp_unix"] = pd.to_numeric(t["timestamp_unix"], errors="coerce")
    t["home_yes_price_cents"] = pd.to_numeric(t["home_yes_price_cents"], errors="coerce")
    t["count_fp"] = pd.to_numeric(t["count_fp"], errors="coerce").fillna(0.0)
    t = t[t["timestamp_unix"].notna() & t["home_yes_price_cents"].notna()]
    t["k_date"] = pd.to_datetime(t["game_date"], errors="coerce").dt.normalize()
    t["k_home"] = canon(t["home_team"])
    t["k_away"] = canon(t["away_team"])
    return t


def map_kalshi_to_games(t, games):
    """games: DataFrame game_id, game_date, home_abbr, away_abbr."""
    g = games.copy()
    g["g_date"] = pd.to_datetime(g["game_date"], errors="coerce").dt.normalize()
    g["g_home"] = canon(g["home_abbr"])
    g["g_away"] = canon(g["away_abbr"])

    keyed = {}
    for gid, dt, h, a in zip(g["game_id"], g["g_date"], g["g_home"], g["g_away"]):
        keyed[(dt, h, a)] = gid

    out = np.full(len(t), -1, dtype="int64")
    kd = t["k_date"].to_numpy()
    kh = t["k_home"].to_numpy()
    ka = t["k_away"].to_numpy()
    one = np.timedelta64(1, "D")
    for shift in (0, 1, -1):
        todo = out < 0
        if not todo.any():
            break
        dates = kd + shift * one
        for i in np.flatnonzero(todo):
            gid = keyed.get((pd.Timestamp(dates[i]), kh[i], ka[i]))
            if gid is not None:
                out[i] = gid
    t = t.assign(game_id=out)
    matched = (out >= 0)
    print(f"[kalshi] matched {matched.mean():.3%} of trades to {pd.unique(out[matched]).size:,} games")
    if (~matched).any():
        bad = (
            t.loc[~matched, ["k_date", "k_away", "k_home"]]
            .drop_duplicates()
            .head(20)
        )
        print("[kalshi] unmatched (first 20 date/away/home):")
        print(bad.to_string(index=False))
    return t[matched].copy()


# ----------------------------------------------------------------------------


def finish_types(feat):
    """How each game finished. MUST be called on the full feature table: in
    breaks mode the only surviving period-4 row is the one that exists when the
    game is going to a shootout, so classifying after the filter would label
    every overtime game 'regulation'."""
    reached_ot = feat.groupby("game_id")["period"].max() >= 4
    ot_rows = feat[feat["period"] >= 4]
    last_ot = (ot_rows.sort_values(["game_id", "event_idx"], kind="stable")
               .groupby("game_id").tail(1).set_index("game_id"))
    out = {}
    for gid, was_ot in reached_ot.items():
        if not was_ot:
            out[gid] = "regulation"
        elif gid in last_ot.index and int(last_ot.at[gid, "score_diff"]) == 0:
            out[gid] = "shootout"
        else:
            out[gid] = "overtime"
    return out


def post_bet_events(feat_all, b, window):
    """What happened in the `window` seconds after each fill.

    Scores only ever increase within a game, so the score at the last row in
    the window is the max over the window -- no per-bet slice scan needed. The
    power-play flags use a running count instead, since they toggle.
    """
    cols = ["game_id", "wallclock_unix", "home_score", "away_score",
            "home_pp", "away_pp"]
    f = feat_all[cols].dropna(subset=["wallclock_unix"])
    f = f.sort_values(["game_id", "wallclock_unix"], kind="stable")

    gid = f["game_id"].to_numpy("int64")
    w = f["wallclock_unix"].to_numpy("float64")
    hs = f["home_score"].to_numpy("float64")
    asc = f["away_score"].to_numpy("float64")
    hpp = f["home_pp"].to_numpy("float64")
    app = f["away_pp"].to_numpy("float64")

    uniq, first = np.unique(gid, return_index=True)
    starts = dict(zip(uniq, first))
    ends = dict(zip(uniq, list(first[1:]) + [len(gid)]))

    n = len(b)
    g_home = np.zeros(n, dtype=bool)
    g_away = np.zeros(n, dtype=bool)
    p_home = np.zeros(n, dtype=bool)
    p_away = np.zeros(n, dtype=bool)

    bg = b["game_id"].to_numpy("int64")
    bt = b["fill_ts"].to_numpy("float64")
    bhs = b["home_score"].to_numpy("float64")
    bas = b["away_score"].to_numpy("float64")
    bhp = b["home_pp"].to_numpy("float64") if "home_pp" in b.columns else np.zeros(n)
    bap = b["away_pp"].to_numpy("float64") if "away_pp" in b.columns else np.zeros(n)

    for g in np.unique(bg):
        if g not in starts:
            continue
        a0, a1 = starts[g], ends[g]
        ww = w[a0:a1]
        chp = np.concatenate([[0.0], np.cumsum(hpp[a0:a1])])
        cap = np.concatenate([[0.0], np.cumsum(app[a0:a1])])
        m = np.flatnonzero(bg == g)
        lo = np.searchsorted(ww, bt[m], side="right")
        hi = np.searchsorted(ww, bt[m] + window, side="right")
        has = hi > lo
        idx = np.clip(hi - 1, 0, len(ww) - 1)
        g_home[m] = has & (hs[a0:a1][idx] > bhs[m])
        g_away[m] = has & (asc[a0:a1][idx] > bas[m])
        p_home[m] = has & ((chp[hi] - chp[lo]) > 0) & (bhp[m] == 0)
        p_away[m] = has & ((cap[hi] - cap[lo]) > 0) & (bap[m] == 0)

    backed_home = b["side_home"].to_numpy() == 1
    out = pd.DataFrame(index=b.index)
    out["post_goal_for"] = np.where(backed_home, g_home, g_away)
    out["post_goal_against"] = np.where(backed_home, g_away, g_home)
    out["post_pp_for"] = np.where(backed_home, p_home, p_away)
    out["post_pp_against"] = np.where(backed_home, p_away, p_home)
    return out


BREAK_ORDER = ["after P1", "after P2", "after regulation (OT next)",
               "after OT (shootout next)"]


def period_break_rows(feat):
    """One row per intermission: the last event of each period, with the anchor
    pushed forward to the buzzer.

    The game clock stops on whistles, so time-to-buzzer understates the wall
    clock a little; the last event sits a median of ~9s from the buzzer in
    regulation, so the slippage is small next to a 60s lag.

    After regulation only counts if the game is level (overtime is next), and
    after overtime only if it is still level (a shootout is next).
    """
    f = feat.sort_values(["game_id", "event_idx"], kind="stable")
    last = f.groupby(["game_id", "period"], as_index=False).tail(1).copy()

    per = last["period"].to_numpy("float64")
    ps = last["period_seconds"].to_numpy("float64")
    to_buzzer = np.where(per <= 3, 1200.0 - ps, 300.0 - ps).clip(0, None)
    last["anchor_ts"] = last["wallclock_unix"].to_numpy("float64") + to_buzzer

    sd = last["score_diff"].to_numpy("float64")
    label = np.full(len(last), None, dtype=object)
    label[per == 1] = "after P1"
    label[per == 2] = "after P2"
    label[(per == 3) & (sd == 0)] = "after regulation (OT next)"
    label[(per == 4) & (sd == 0)] = "after OT (shootout next)"
    last["break_view"] = label
    return last[last["break_view"].notna()].reset_index(drop=True)


def find_fills(feat, trades, lag, window):
    """Attach the first printed trade in (anchor, anchor+window]."""
    feat = feat.sort_values(["game_id", "wallclock_unix"], kind="stable").reset_index(drop=True)
    trades = trades.sort_values(["game_id", "timestamp_unix"], kind="stable").reset_index(drop=True)

    price = np.full(len(feat), np.nan)
    count = np.full(len(feat), np.nan)
    ts = np.full(len(feat), np.nan)

    tg = trades["game_id"].to_numpy("int64")
    tt = trades["timestamp_unix"].to_numpy("float64")
    tp = trades["home_yes_price_cents"].to_numpy("float64")
    tc = trades["count_fp"].to_numpy("float64")

    starts = {}
    uniq, first_idx = np.unique(tg, return_index=True)
    for u, fi in zip(uniq, first_idx):
        starts[u] = fi
    ends = dict(zip(uniq, list(first_idx[1:]) + [len(tg)]))

    fg = feat["game_id"].to_numpy("int64")
    fw = (feat["anchor_ts"].to_numpy("float64") if "anchor_ts" in feat.columns
          else feat["wallclock_unix"].to_numpy("float64"))

    for gid in np.unique(fg):
        if gid not in starts:
            continue
        a, b = starts[gid], ends[gid]
        sub_t = tt[a:b]
        rows = np.flatnonzero(fg == gid)
        anchor = fw[rows] + lag
        pos = np.searchsorted(sub_t, anchor, side="left")
        ok = (pos < len(sub_t))
        pc = np.clip(pos, 0, max(len(sub_t) - 1, 0))
        hit = ok & (sub_t[pc] <= anchor + window)
        idx = rows[hit]
        src = a + pc[hit]
        price[idx] = tp[src]
        count[idx] = tc[src]
        ts[idx] = tt[src]

    feat["fill_home_cents"] = price
    feat["fill_count"] = count
    feat["fill_ts"] = ts
    return feat


# ----------------------------------------------------------------------------


def bucket_edge(e):
    bins = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 1.01]
    labels = ["2-3%", "3-5%", "5-7.5%", "7.5-10%", "10-15%", "15%+"]
    return pd.cut(e, bins=bins, labels=labels, right=False)


def bucket_price(p):
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
    labels = ["1-9c", "10-19c", "20-29c", "30-39c", "40-49c", "50-59c",
              "60-69c", "70-79c", "80-89c", "90-100c"]
    return pd.cut(p, bins=bins, labels=labels, right=False)


def bucket_scorediff(d):
    d = np.asarray(d)
    out = np.empty(len(d), dtype=object)
    out[:] = "tied"
    out[d == -1] = "trail 1"
    out[d == -2] = "trail 2"
    out[d == -3] = "trail 3"
    out[d <= -4] = "trail 4+"
    out[d == 1] = "lead 1"
    out[d == 2] = "lead 2"
    out[d == 3] = "lead 3"
    out[d >= 4] = "lead 4+"
    return out


SCORE_ORDER = ["trail 4+", "trail 3", "trail 2", "trail 1", "tied",
               "lead 1", "lead 2", "lead 3", "lead 4+"]


def summarise(bets, by, order=None, title=""):
    g = bets.groupby(by, observed=True, dropna=False).agg(
        bets=("pnl", "size"),
        games=("game_id", "nunique"),
        staked=("stake", "sum"),
        pnl=("pnl", "sum"),
        avg_price=("price_cents", "mean"),
        avg_size=("contracts", "mean"),
        win_pct=("won", "mean"),
    )
    g["roi_pct"] = 100.0 * g["pnl"] / g["staked"].replace(0, np.nan)
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    tot = pd.DataFrame({
        "bets": [len(bets)],
        "games": [bets["game_id"].nunique()],
        "staked": [bets["stake"].sum()],
        "pnl": [bets["pnl"].sum()],
        "avg_price": [bets["price_cents"].mean()],
        "avg_size": [bets["contracts"].mean()],
        "win_pct": [bets["won"].mean()],
        "roi_pct": [100.0 * bets["pnl"].sum() / max(bets["stake"].sum(), 1e-9)],
    }, index=["TOTAL"])
    g = pd.concat([g, tot])
    print(f"\n=== {title or by} ===")
    print(g.to_string(formatters={
        "bets": "{:,.0f}".format, "games": "{:,.0f}".format,
        "staked": "${:,.0f}".format, "pnl": "${:,.0f}".format,
        "avg_price": "{:.1f}".format, "avg_size": "{:,.0f}".format,
        "win_pct": "{:.3f}".format,
        "roi_pct": "{:+.2f}".format,
    }))
    return g.assign(view=title or str(by)).reset_index().rename(columns={"index": "bucket"})


# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2026])
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--mode", choices=["all", "breaks"], default="all",
                    help="all = every play-by-play event; breaks = only the "
                         "intermissions after P1, P2, regulation (if the game "
                         "is tied) and OT (if it is going to a shootout)")
    ap.add_argument("--lag", type=float, default=None,
                    help="seconds after the anchor before we look for a print "
                         "(default 10, or 60 in breaks mode)")
    ap.add_argument("--window", type=float, default=None,
                    help="how long to keep looking (default 5, or 60 in breaks mode)")
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--contracts", type=int, default=100,
                    help="AVERAGE contracts per bet; individual bets are scaled by edge")
    ap.add_argument("--size-slope", type=float, default=1.0,
                    help="contracts multiplier = 1 + slope * z(edge). 0 = flat sizing")
    ap.add_argument("--size-min", type=float, default=0.20,
                    help="floor on the size multiplier before rescaling")
    ap.add_argument("--size-max", type=float, default=3.00,
                    help="cap on the size multiplier before rescaling")
    ap.add_argument("--cap-to-print", action="store_true",
                    help="cap size at the matched print's contract count. Off by "
                         "default: the book is deep and a cent wide, so a small "
                         "print means someone wanted a small size, not that size "
                         "was unavailable.")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--post-window", type=float, default=30.0,
                    help="seconds after the fill to look for what happened next")
    ap.add_argument("--base-calib-seasons", nargs="*", type=int, default=[],
                    help="fit a shrink on baseline_logit using these seasons "
                         "(training seasons only) and apply it to the backtest")
    ap.add_argument("--base-calib-by", choices=["none", "period", "state", "tails"],
                    default="none",
                    help="fit the shrink per period, per (|score diff|, time "
                         "left) cell, or as a smooth function of distance from "
                         "even money")
    ap.add_argument("--cell-prior-rows", type=float, default=50_000.0,
                    help="pseudo-rows pulling each cell's fit toward the global one")
    ap.add_argument("--cell-min-rows", type=int, default=5_000,
                    help="cells thinner than this just use the global fit")
    ap.add_argument("--base-only", action="store_true",
                    help="price off the analytic Skellam baseline, no trees")
    args = ap.parse_args()
    if args.lag is None:
        args.lag = 60.0 if args.mode == "breaks" else 10.0
    if args.window is None:
        args.window = 60.0 if args.mode == "breaks" else 5.0

    if args.base_only:
        meta, models = None, None
        print("[model] BASELINE ONLY - analytic Skellam, no trees")
    else:
        meta, models = load_model(args.root, args.tag)
        print(f"[model] tag={args.tag} members={len(models)} "
              f"features={len(meta['features'])} calib={meta['calibrator']['kind']} "
              f"margin={meta['base_margin']}")

    fdir = os.path.join(args.root, "features")
    frames = []
    for s in args.seasons:
        fp = os.path.join(fdir, f"nhl_wp_features_{s}.parquet")
        if not os.path.exists(fp):
            print(f"[warn] missing {fp}")
            continue
        frames.append(pd.read_parquet(fp))
    if not frames:
        sys.exit("ERROR: no feature files")
    feat = pd.concat(frames, ignore_index=True)
    feat = feat[feat["wallclock_unix"].notna()].reset_index(drop=True)
    print(f"[feat] {len(feat):,} timestamped rows / {feat['game_id'].nunique():,} games")

    if args.base_only:
        feat["p_home"] = feat["baseline_wp"].astype("float64")
    else:
        feat["p_home"] = predict(feat, meta, models)

    end_map = finish_types(feat)
    print(f"[games] finish type: {pd.Series(end_map).value_counts().to_dict()}")

    if args.mode == "breaks":
        feat = period_break_rows(feat)
        print(f"[breaks] {len(feat):,} intermissions across "
              f"{feat['game_id'].nunique():,} games")
        print(feat["break_view"].value_counts().reindex(BREAK_ORDER).to_string())

    if args.base_calib_seasons:
        overlap = set(args.base_calib_seasons) & set(args.seasons)
        if overlap:
            sys.exit(f"ERROR: --base-calib-seasons overlaps the backtest seasons {sorted(overlap)}")
        a, b, by_period, by_cell, tails = fit_base_shrink(
            args.root, args.base_calib_seasons, args.cell_min_rows, args.cell_prior_rows,
            want_tails=(args.base_calib_by == "tails"))
        z = logit(feat["p_home"].to_numpy("float64"))
        if args.base_calib_by == "tails" and tails is not None:
            ta, tc, tb = tails
            feat["p_home"] = sigmoid(ta * z + tc * z * np.abs(z) + tb)
        elif args.base_calib_by == "state":
            cl = state_cell(feat["score_diff"], feat["secs_left_reg"], feat["period"])
            av = np.full(len(z), a)
            bv = np.full(len(z), b)
            for k, (ca, cb) in by_cell.items():
                m = cl == k
                av[m] = ca
                bv[m] = cb
            feat["p_home"] = sigmoid(av * z + bv)
        elif args.base_calib_by == "period":
            pb = period_bucket(feat["period"].to_numpy("float64"))
            av = np.full(len(z), a)
            bv = np.full(len(z), b)
            for k, (pa, pbi) in by_period.items():
                m = pb == k
                av[m] = pa
                bv[m] = pbi
            feat["p_home"] = sigmoid(av * z + bv)
        else:
            feat["p_home"] = sigmoid(a * z + b)

    trades = load_kalshi(args.root)
    games = feat.drop_duplicates("game_id")[["game_id", "game_date", "home_abbr", "away_abbr"]]
    trades = map_kalshi_to_games(trades, games)

    # goal wallclock times per game, captured BEFORE filtering to filled rows,
    # so we can tell which fills straddle a scoring event
    goal_times = {}
    gsub = feat[feat["event_type"] == "GOAL"]
    for gid, grp in gsub.groupby("game_id"):
        goal_times[int(gid)] = np.sort(grp["wallclock_unix"].to_numpy("float64"))

    feat_all = feat
    feat = find_fills(feat, trades, args.lag, args.window)
    filled = feat["fill_home_cents"].notna()
    print(f"[fill] {filled.mean():.3%} of decision points matched a print "
          f"({filled.sum():,} of {len(feat):,})")
    feat = feat[filled].copy()
    if feat.empty:
        sys.exit("ERROR: no fills")

    hp = feat["fill_home_cents"].to_numpy("float64") / 100.0
    apr = (101.0 - feat["fill_home_cents"].to_numpy("float64")) / 100.0
    p_home = feat["p_home"].to_numpy("float64")

    edge_h = p_home - hp - FEE_RATE * hp * (1.0 - hp)
    edge_a = (1.0 - p_home) - apr - FEE_RATE * apr * (1.0 - apr)

    take_home = edge_h >= edge_a
    edge = np.where(take_home, edge_h, edge_a)
    keep = edge >= args.min_edge

    b = feat[keep].copy()
    th = take_home[keep]
    b["side_home"] = th.astype(int)
    b["edge"] = edge[keep]
    b["price"] = np.where(th, hp[keep], apr[keep])
    b["price_cents"] = np.round(b["price"] * 100.0, 0)
    b["model_p"] = np.where(th, p_home[keep], 1.0 - p_home[keep])

    n, zscore = edge_scaled_size(b["edge"].to_numpy("float64"),
                                 float(args.contracts), args.size_slope,
                                 args.size_min, args.size_max)
    if args.cap_to_print:
        n = np.minimum(n, np.nan_to_num(b["fill_count"].to_numpy("float64")))
    n = np.floor(n)
    b["edge_z"] = zscore
    print(f"[size] edge-scaled, avg {args.contracts} contracts per bet"
          + (" capped at the printed size" if args.cap_to_print else " (uncapped)")
          + f"; slope {args.size_slope:g}, multiplier clipped to "
            f"[{args.size_min:.2f}, {args.size_max:.2f}]")
    print(f"[size] mean {n.mean():.1f}  min {n.min():.0f}  "
          f"median {np.median(n):.0f}  max {n.max():.0f}")
    b["contracts"] = n
    b = b[b["contracts"] > 0].copy()

    nc = b["contracts"].to_numpy("float64")
    pr = b["price"].to_numpy("float64")
    fee = fee_dollars(nc, pr)
    won = np.where(b["side_home"].to_numpy() == 1,
                   b["home_won"].to_numpy("float64"),
                   1.0 - b["home_won"].to_numpy("float64"))
    b["won"] = won
    b["stake"] = nc * pr
    b["fee"] = fee
    b["pnl"] = np.where(won == 1, nc * (1.0 - pr), -nc * pr) - fee

    # ---- view columns -------------------------------------------------------
    b["edge_bucket"] = bucket_edge(b["edge"].to_numpy())
    b["price_bucket"] = bucket_price(b["price_cents"].to_numpy())
    b["side"] = np.where(b["side_home"] == 1, "home", "away")
    ph_close = b["p_home_close"].to_numpy("float64")
    home_fav = ph_close >= 0.5
    bet_fav = np.where(b["side_home"].to_numpy() == 1, home_fav, ~home_fav)
    b["pregame"] = np.where(bet_fav, "favorite", "underdog")
    b["role"] = np.where(b["side_home"] == 1,
                         np.where(home_fav, "home fav", "home dog"),
                         np.where(home_fav, "road dog", "road fav"))
    sd_home = b["score_diff"].to_numpy("float64")
    sd_bet = np.where(b["side_home"].to_numpy() == 1, sd_home, -sd_home)
    b["score_bucket"] = bucket_scorediff(sd_bet)
    b["period_view"] = np.where(b["period"] >= 4, "OT", b["period"].astype(int).astype(str))

    # did the score change between the model's evaluation and the fill?
    straddle = np.zeros(len(b), dtype=bool)
    bg = b["game_id"].to_numpy("int64")
    t0 = b["wallclock_unix"].to_numpy("float64")
    t1 = b["fill_ts"].to_numpy("float64")
    for gid in np.unique(bg):
        gt = goal_times.get(int(gid))
        if gt is None or len(gt) == 0:
            continue
        m = bg == gid
        lo = np.searchsorted(gt, t0[m], side="right")
        hi = np.searchsorted(gt, t1[m], side="right")
        straddle[m] = hi > lo
    b["state_changed"] = np.where(straddle, "goal before fill", "state held")
    b["game_end"] = b["game_id"].map(end_map)

    pe = post_bet_events(feat_all, b, args.post_window)
    for c in pe.columns:
        b[c] = np.where(pe[c].to_numpy(), "yes", "no")
    print(f"[post] within {args.post_window:.0f}s of the fill: "
          + ", ".join(f"{c.replace('post_', '')} {int((pe[c]).sum()):,}"
                      for c in pe.columns))
    print(f"[stale] {straddle.sum():,} of {len(b):,} bets ({straddle.mean():.2%}) "
          f"had a goal land between evaluation and fill")
    dt = pd.to_datetime(b["game_date"], errors="coerce")
    b["month"] = dt.dt.strftime("%Y-%m")
    b["year"] = dt.dt.year.astype("Int64").astype(str)

    print(f"\n[bets] {len(b):,} bets / {b['game_id'].nunique():,} games "
          f"| staked ${b['stake'].sum():,.0f} | PNL ${b['pnl'].sum():,.0f} "
          f"| ROI {100 * b['pnl'].sum() / max(b['stake'].sum(), 1e-9):+.2f}% "
          f"| fees ${b['fee'].sum():,.0f}")

    views = []
    if "break_view" in b.columns:
        views.append(summarise(b, "break_view", order=BREAK_ORDER,
                               title="BY INTERMISSION"))
    views.append(summarise(b, "edge_bucket", title="BY EDGE RANGE"))
    views.append(summarise(b, "side", order=["home", "away"], title="BY HOME / AWAY"))
    views.append(summarise(b, "pregame", order=["favorite", "underdog"],
                           title="BY PREGAME FAV / DOG"))
    views.append(summarise(b, "role", order=["home fav", "home dog", "road fav", "road dog"],
                           title="BY ROLE"))
    views.append(summarise(b, "price_bucket", title="BY EXECUTION PRICE"))
    views.append(summarise(b, "period_view", order=["1", "2", "3", "OT"], title="BY PERIOD"))
    views.append(summarise(b, "score_bucket", order=SCORE_ORDER,
                           title="BY SCORE DIFFERENTIAL (team bet on)"))
    views.append(summarise(b, "game_end",
                           order=["regulation", "overtime", "shootout"],
                           title="BY HOW THE GAME FINISHED"))
    for col, title in (
        ("post_goal_for", f"GOAL FOR THE BACKED TEAM WITHIN {args.post_window:.0f}s"),
        ("post_goal_against", f"GOAL FOR THE OPPONENT WITHIN {args.post_window:.0f}s"),
        ("post_pp_for", f"POWER PLAY FOR THE BACKED TEAM WITHIN {args.post_window:.0f}s"),
        ("post_pp_against", f"POWER PLAY FOR THE OPPONENT WITHIN {args.post_window:.0f}s"),
    ):
        views.append(summarise(b, col, order=["yes", "no"], title=title))
    views.append(summarise(b, "state_changed",
                           order=["state held", "goal before fill"],
                           title="BY STATE CHANGE BEFORE FILL"))
    views.append(summarise(b, "month", title="BY MONTH"))
    views.append(summarise(b, "year", title="BY YEAR"))

    odir = os.path.join(args.root, "backtests")
    os.makedirs(odir, exist_ok=True)
    sfx = args.out_suffix or ((("breaks_" if args.mode == "breaks" else "")
                               + ("base" if args.base_only else args.tag))
                              + ("_shrunk" if args.base_calib_seasons else "")
                              + ("_by" + args.base_calib_by
                                 if args.base_calib_by != "none" else "")
                              + "_" + "_".join(str(s) for s in args.seasons))
    vfp = os.path.join(odir, f"nhl_backtest_views_{sfx}.csv")
    pd.concat(views, ignore_index=True).to_csv(vfp, index=False)

    keep_cols = [
        "game_id", "game_date", "home_abbr", "away_abbr", "period", "period_seconds",
        "home_score", "away_score", "score_diff", "wallclock_unix", "fill_ts",
        "side", "role", "pregame", "model_p", "price_cents", "edge", "edge_z",
        "contracts",
        "stake", "fee", "pnl", "won", "edge_bucket", "price_bucket", "score_bucket",
        "month", "p_home_close", "state_changed", "game_end", "break_view",
        "anchor_ts", "post_goal_for", "post_goal_against", "post_pp_for",
        "post_pp_against",
    ]
    bfp = os.path.join(odir, f"nhl_backtest_bets_{sfx}.csv")
    b[[c for c in keep_cols if c in b.columns]].to_csv(bfp, index=False)
    print(f"\n[out] {vfp}\n[out] {bfp}")

    wfp = os.path.join(args.root, "models", f"nhl_wp_{args.tag}_weights.csv")
    if not args.base_only and os.path.exists(wfp):
        w = pd.read_csv(wfp).head(30)
        print("\n=== FEATURE WEIGHTING (top 30, mean |SHAP| share) ===")
        print(w[["feature", "gain", "mean_abs_shap", "shap_share_pct"]].to_string(
            index=False,
            formatters={"gain": "{:,.0f}".format, "mean_abs_shap": "{:.5f}".format,
                        "shap_share_pct": "{:.2f}".format}))


if __name__ == "__main__":
    main()
