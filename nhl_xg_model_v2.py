#!/usr/bin/env python
"""
Fit ONE xG model across 2011-2026 from raw shot features, replacing the
source's own `xg`.

WHY
---
nhl_xg_regime_check_v1.py showed the vendor changed its xG model at the 2025
season. After scaling each season so its xg sums to its goals - which removes
any level difference - the same kinds of shot still get materially different
values across the break: tip-ins -29%, pokes +23%, shots from behind the goal
line -14%, and a spread of -13% to +10% across distance x angle buckets. That
is a reweighting, and no per-season multiplier undoes a reweighting.

Pooling seasons with the vendor's xg therefore trains on a feature whose
meaning changes partway through the sample. This script replaces it with one
model, fit on inputs that are recorded identically in every season back to
2011, so xG means the same thing in 2012 as in 2026.

NO LEAKAGE
----------
The xG for a season comes from a model fit ONLY on earlier seasons, refit
each season on an expanding window. The first --seed-seasons cannot have that
and are fit on themselves; they are marked `xg_in_sample = True` so they can
be dropped from a backtest rather than quietly trusted. With the default of 3
that is 2011-2013, and 2014 onward are all clean out-of-sample.

FEATURES - all present in every season
--------------------------------------
  shot_distance, shot_angle       100% populated on fenwick events
  secondary_type                  wrist / snap / slap / tip-in / backhand /
                                  deflected / wrap-around / poke / bat
  own_skaters, opp_skaters        from home_skaters/away_skaters, read from
                                  the SHOOTER's side, so 5v4 means the shooter
                                  is on the power play
  empty_net                       the defending net is empty
  is_rebound                      a prior unblocked attempt within
                                  --rebound-seconds
  seconds_since_last              time since the previous event, capped
  is_home, period

Blocked shots are excluded: they never reach the net and carry no distance.

OUTPUT
------
  <outdir>/xg_own/xg_{season}.parquet
      game_id, event_idx, xg_own, xg_in_sample

  Join on (game_id, event_idx). nhl_ytd_stats_v6.py picks these up
  automatically and uses them instead of the vendor column.

Setup
-----
  pip install pandas pyarrow scikit-learn numpy

Usage
-----
  python nhl_xg_model_v2.py --seasons 2011 2026 --outdir data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FENWICK = ("SHOT", "GOAL", "MISSED_SHOT")
LOAD = ["game_id", "event_idx", "event_type", "secondary_type",
        "shot_distance", "shot_angle", "x_fixed", "y_fixed",
        "home_skaters", "away_skaters", "event_team_type", "empty_net",
        "period", "period_type", "game_seconds", "season_type"]

SHOT_TYPES = ["wrist", "snap", "slap", "tip-in", "backhand", "deflected",
              "wrap-around", "poke", "bat", "between-legs", "cradle"]


def build_features(season: int, raw_dir: Path,
                   rebound_s: float) -> pd.DataFrame | None:
    path = raw_dir / f"play_by_play_{season}.parquet"
    if not path.exists():
        return None
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(path).schema_arrow.names)
    df = pd.read_parquet(path, columns=[c for c in LOAD if c in have])
    for c in LOAD:
        if c not in df.columns:
            df[c] = np.nan

    pt = df["period_type"].astype(str).str.upper()
    df = df[pt != "SHOOTOUT"]
    df = df.sort_values(["game_id", "game_seconds", "event_idx"])

    # Gaps are measured against ALL events, not just shots, so "seconds since
    # last" means what it says. Computed before filtering to fenwick.
    df["gap"] = df.groupby("game_id")["game_seconds"].diff()
    is_fen = df["event_type"].isin(FENWICK)
    prev_fen_t = (df["game_seconds"].where(is_fen)
                    .groupby(df["game_id"]).ffill().groupby(df["game_id"]).shift())
    df["since_fen"] = df["game_seconds"] - prev_fen_t

    f = df[is_fen].copy()
    if f.empty:
        return None

    home = f["event_team_type"].astype(str).eq("home")
    hs = pd.to_numeric(f["home_skaters"], errors="coerce")
    as_ = pd.to_numeric(f["away_skaters"], errors="coerce")
    f["own_skaters"] = np.where(home, hs, as_)
    f["opp_skaters"] = np.where(home, as_, hs)
    f["is_home"] = home.astype(int)

    dist = pd.to_numeric(f["shot_distance"], errors="coerce")
    ang = pd.to_numeric(f["shot_angle"], errors="coerce").abs()
    # Fall back to geometry when the feed omits them. Distance is to the goal
    # line centre at x = 89.
    x = pd.to_numeric(f["x_fixed"], errors="coerce").abs()
    y = pd.to_numeric(f["y_fixed"], errors="coerce")
    gx = (89.0 - x)
    f["dist"] = dist.fillna(np.sqrt(gx ** 2 + y ** 2))
    f["angle"] = ang.fillna(np.degrees(np.arctan2(y.abs(), gx.clip(lower=0.01))))

    f["empty_net"] = (f["empty_net"].astype("object").where(
        f["empty_net"].notna(), False).astype(bool).astype(int))
    f["is_rebound"] = (f["since_fen"] <= rebound_s).fillna(False).astype(int)
    f["since_last"] = f["gap"].fillna(60.0).clip(0, 60)
    f["period_c"] = pd.to_numeric(f["period"], errors="coerce").fillna(1).clip(1, 4)

    st = f["secondary_type"].astype(str).str.lower()
    for t in SHOT_TYPES:
        f[f"st_{t.replace('-', '_')}"] = (st == t).astype(int)

    f["goal"] = f["event_type"].eq("GOAL").astype(int)
    f["season"] = season
    return f


FEATS = (["dist", "angle", "own_skaters", "opp_skaters", "empty_net",
          "is_rebound", "since_last", "period_c", "is_home"]
         + [f"st_{t.replace('-', '_')}" for t in SHOT_TYPES])


def fit(train: pd.DataFrame, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1,
        random_state=seed)
    m.fit(train[FEATS].to_numpy(dtype="float64"), train["goal"].to_numpy())
    return m


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    default=[2011, 2026])
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    ap.add_argument("--train-seasons", type=int, default=6,
                    help="TRAILING window of prior seasons used to fit each "
                         "season's model. v1 used every prior season, and "
                         "calibration decayed from 1.003 in 2014 to 0.930 in "
                         "2026 while AUC fell 0.792 to 0.749: shot LOCATION "
                         "recording has changed, so old seasons teach a "
                         "relationship that no longer holds. 0 restores the "
                         "old expanding behaviour.")
    ap.add_argument("--no-recalibrate", action="store_true",
                    help="Skip the level correction described below.")
    ap.add_argument("--seed-seasons", type=int, default=3,
                    help="How many opening seasons are fit on themselves "
                         "because nothing earlier exists. They are marked "
                         "xg_in_sample. Default 3.")
    ap.add_argument("--rebound-seconds", type=float, default=3.0)
    ap.add_argument("--random-state", type=int, default=0)
    args = ap.parse_args()

    raw_dir = args.outdir / "nhl_pbp_raw"
    out_dir = args.outdir / "xg_own"
    out_dir.mkdir(parents=True, exist_ok=True)
    first, last = args.seasons

    print("Building shot features:")
    per_season: dict[int, pd.DataFrame] = {}
    for s in range(first, last + 1):
        f = build_features(s, raw_dir, args.rebound_seconds)
        if f is None:
            print(f"  {s}: no source file")
            continue
        per_season[s] = f
        print(f"  {s}: {len(f):,} unblocked attempts, "
              f"{f['goal'].mean():.4f} goal rate")
    if not per_season:
        print("nothing to fit", file=sys.stderr)
        return 1

    seasons = sorted(per_season)
    seed_set = set(seasons[:args.seed_seasons])
    seed_model = fit(pd.concat([per_season[s] for s in seed_set],
                               ignore_index=True), args.random_state)

    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
    rows = []
    print("\nPer season - the model for a season sees only EARLIER seasons:")
    print(f"  {'season':>7} {'shots':>9} {'trained on':>18} {'cal':>6} "
          f"{'sum xg':>9} {'goals':>7} {'ratio':>7} {'AUC':>6} {'Brier':>7} "
          f"{'logloss':>8}")
    for s in seasons:
        f = per_season[s]
        if s in seed_set:
            model, trained, insample = seed_model, "itself (seed)", True
        else:
            pri = [q for q in seasons if q < s]
            if args.train_seasons > 0:
                pri = pri[-args.train_seasons:]
            model = fit(pd.concat([per_season[q] for q in pri],
                                  ignore_index=True), args.random_state)
            trained = f"{pri[0]}-{pri[-1]}"
            insample = False

        p = model.predict_proba(f[FEATS].to_numpy(dtype="float64"))[:, 1]

        # Level correction, learned OUT OF SAMPLE. The model's shape is now
        # consistent across seasons, so what remains is a level offset caused
        # by shot recording drifting between the training window and the
        # season being scored. The correction is this model's own miss on the
        # most recent season it did NOT train on - a single scalar from a
        # prior season, never from this one.
        cal = 1.0
        if not args.no_recalibrate and not insample and len(pri) >= 2:
            probe = per_season[pri[-1]]
            probe_p = fit(pd.concat([per_season[q] for q in pri[:-1]],
                                    ignore_index=True), args.random_state
                          ).predict_proba(
                probe[FEATS].to_numpy(dtype="float64"))[:, 1]
            cal = float(probe["goal"].sum()) / max(float(probe_p.sum()), 1e-9)
            p = np.clip(p * cal, 1e-6, 1 - 1e-6)

        y = f["goal"].to_numpy()
        auc = roc_auc_score(y, p) if y.sum() else np.nan
        out = pd.DataFrame({
            "game_id": f["game_id"].to_numpy(),
            "event_idx": f["event_idx"].to_numpy(),
            "xg_own": p.astype("float32"),
            "xg_in_sample": insample,
        })
        out.to_parquet(out_dir / f"xg_{s}.parquet", index=False)
        ratio = y.sum() / max(p.sum(), 1e-9)
        rows.append((s, ratio, auc, brier_score_loss(y, p), log_loss(y, p),
                     insample))
        print(f"  {s:>7} {len(f):>9,} {trained:>18} {cal:>6.3f} "
              f"{p.sum():>9,.0f} {y.sum():>7,} {ratio:>7.4f} {auc:>6.3f} "
              f"{brier_score_loss(y, p):>7.4f} {log_loss(y, p):>8.4f}")

    r = pd.DataFrame(rows, columns=["season", "ratio", "auc", "brier",
                                    "logloss", "in_sample"])
    oos = r[~r["in_sample"]]
    print("\nCalibration - goals divided by summed xG, out-of-sample seasons:")
    print(f"  median {oos['ratio'].median():.4f}   "
          f"range {oos['ratio'].min():.4f} to {oos['ratio'].max():.4f}")
    print("  A model that is calibrated sits at 1.000. The vendor's own column "
          "sat between 0.70 and 0.95 and moved with its model version - that "
          "drift is what this replaces.")
    print(f"\nDiscrimination: AUC median {oos['auc'].median():.3f}")
    if args.seed_seasons:
        print(f"\nSeasons {seasons[:args.seed_seasons]} are fit on themselves "
              f"and marked xg_in_sample - drop them from any backtest.")
    print(f"\nWrote {len(seasons)} files to {out_dir}. "
          f"nhl_ytd_stats_v6.py picks them up automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
