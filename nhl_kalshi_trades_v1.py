#!/usr/bin/env python
"""
Stage 1 - pull every executed Kalshi trade on NHL game-winner markets
(KXNHLGAME) for the 2025-26 REGULAR SEASON and write the backtest price tape.

Port of nba_kalshi_trades.py. Same auth, same paging, same shard/resume model,
same output columns. Three things are NHL-specific:

  1. SERIES = "KXNHLGAME". Confirmed from live market tickers, e.g.
         KXNHLGAME-26MAR28WSHVGK-VGK
         KXNHLGAME-25DEC31NJCBJ-NJ
     Same shape as NBA: <SERIES>-<yy><MON><dd><AWAY><HOME>-<YES>, no start-time
     segment. (KXNHL is the Stanley Cup futures series and KXNHLSERIES is the
     playoff series-winner series - neither is game markets.)

  2. Kalshi uses ESPN-style hockey abbreviations, which are NOT the nflverse /
     hockey-reference 3-letter codes. Four teams are TWO letters:
         LA (Kings), NJ (Devils), SJ (Sharks), TB (Lightning)
     plus WSH (not WAS), UTA (Mammoth), VGK, CBJ, MTL, WPG.
     A 2-letter code makes the away+home blob ambiguous under brute-force
     splitting, so the split is anchored on the ticker's own YES suffix
     instead: the YES leg is one of the two teams, so the blob either starts
     with it (YES = away) or ends with it (YES = home). The abbreviation table
     is only used to validate the result and as a fallback. See _split_teams.

  3. Season window. The 2025-26 regular season ran 2025-10-07 through
     2026-04-16. The playoffs began 2026-04-18. There was no 2026 All-Star
     Game - the league took an Olympic break instead (roughly Feb 6-24), so
     no All-Star market exists to exclude. The default --start/--end therefore
     covers regular season only; nothing else is filtered by name.

Anything that fails to parse is COUNTED AND LOGGED, never silently dropped.
If Kalshi lists non-NHL hockey under this series (Olympic country codes, for
instance), it fails the abbreviation check and shows up in the unparsed list.

Output columns
--------------
  game_date             date parsed from the ticker
  away_team, home_team  Kalshi abbreviations (LA, NJ, SJ, TB stay 2-letter)
  event_ticker, ticker  Kalshi identifiers
  is_home_yes           1 if this market's YES resolves for the home team
  trade_id              Kalshi trade id, used for dedupe
  timestamp             ISO-8601 UTC
  timestamp_unix        Unix seconds as float; sub-second precision is kept
  yes_price_cents       executed YES price of THIS ticker, fractional cents
  home_yes_price_cents  same trade restated as implied P(home win), cents
  no_price_cents        Kalshi's own NO price, kept for a consistency check
  count_fp              contracts traded, fractional
  taker_side            'yes' | 'no'
  taker_outcome_side    'yes' | 'no'
  taker_book_side       'bid' | 'ask'   <- the fill-side column
  is_block_trade        bool

WHAT THIS PRICE IS. An executed trade is a taker crossing the spread, so it
prints AT the bid or AT the ask - never the midpoint, never fair value. With
taker_book_side you can reconstruct which side was hit, but you still cannot
see the resting quote on the other side. The fill assumption belongs in the
backtester and must be stated there explicitly.

Setup
-----
  pip install requests cryptography pandas pyarrow

Usage
-----
  python nhl_kalshi_trades_v1.py --api-key KEY --private-key <path-to-key.pem>

Defaults are the full 2025-26 regular season, so no date flags are needed.
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("nhl_trades")

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXNHLGAME"

# 2025-26 regular season. Playoffs start 2026-04-18 and are excluded by --end.
SEASON_START = "2025-10-07"
SEASON_END = "2026-04-16"

# Kalshi's own codes, as they appear in live KXNHLGAME tickers.
VALID_NHL_ABBRS = {
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
    "EDM", "FLA", "LA", "MIN", "MTL", "NJ", "NSH", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SEA", "SJ", "STL", "TB", "TOR", "UTA", "VAN", "VGK",
    "WPG", "WSH",
}

# Standard hockey codes mapped onto Kalshi's. Only a guard in case Kalshi
# changes convention mid-season; none of these are used today.
ABBR_ALIASES = {
    "LAK": "LA", "NJD": "NJ", "SJS": "SJ", "TBL": "TB",
    "WAS": "WSH", "WSG": "WSH",
    "UTAH": "UTA", "ARI": "UTA", "PHX": "UTA",
    "VEG": "VGK", "VGS": "VGK", "LV": "VGK",
    "WIN": "WPG", "WPJ": "WPG",
    "CLB": "CBJ", "CBS": "CBJ",
    "MON": "MTL", "CAL": "CGY", "ANH": "ANA", "NAS": "NSH", "TAM": "TB",
}

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

TICKER_RE = re.compile(
    r"^(?P<series>[A-Z0-9]+)-"
    r"(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})"
    r"(?P<hhmm>\d{4})?"
    r"(?P<teams>[A-Z]{4,10})"
    r"-(?P<yes>[A-Z]{2,4})$"
)


# -- auth ---------------------------------------------------------------------

class KalshiAuth:
    def __init__(self, api_key: str, private_key_path: str):
        self.api_key = api_key
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
        with open(private_key_path, "rb") as f:
            self._key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
        log.info(f"Private key loaded from {private_key_path}")

    def headers(self, method: str, path: str) -> dict:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric import padding as ap
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
        ts = str(int(datetime.now().timestamp() * 1000))
        msg = (ts + method.upper() + path).encode()
        if isinstance(self._key, RSAPrivateKey):
            sig = self._key.sign(
                msg,
                ap.PSS(mgf=ap.MGF1(hashes.SHA256()), salt_length=ap.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
        else:
            sig = self._key.sign(msg, ec.ECDSA(hashes.SHA256()))
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
        }


def _get(session: requests.Session, auth: KalshiAuth, api_path: str,
         url: str, params: dict, retries: int = 6) -> Optional[dict]:
    for attempt in range(retries):
        try:
            r = session.get(url, headers=auth.headers("GET", api_path),
                            params=params, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(30.0, 0.5 * (2 ** attempt)))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            wait = min(30.0, 0.5 * (2 ** attempt))
            log.warning(f"  {api_path} failed ({e}) - retry in {wait:.1f}s")
            time.sleep(wait)
    log.error(f"  Gave up on {api_path} after {retries} attempts")
    return None


# -- market discovery ---------------------------------------------------------

def fetch_markets(auth: KalshiAuth, session: requests.Session,
                  limit: int = 1000) -> list[dict]:
    """
    Every KXNHLGAME market across both tiers.

    Settled markets age out of /markets into /historical/markets. Pulling only
    one tier silently loses most of the season, so both are fetched and merged
    on ticker.
    """
    by_ticker: dict[str, dict] = {}
    for api_path, url in (
        ("/trade-api/v2/markets", f"{KALSHI_BASE}/markets"),
        ("/trade-api/v2/historical/markets", f"{KALSHI_BASE}/historical/markets"),
    ):
        cursor, page, added = None, 0, 0
        while True:
            params = {"series_ticker": SERIES, "limit": limit}
            if cursor:
                params["cursor"] = cursor
            data = _get(session, auth, api_path, url, params)
            if data is None:
                break
            batch = data.get("markets", [])
            for m in batch:
                t = m.get("ticker")
                if t and t not in by_ticker:
                    by_ticker[t] = m
                    added += 1
            cursor = data.get("cursor")
            page += 1
            if not cursor or len(batch) < limit:
                break
            time.sleep(0.12)
        log.info(f"  {api_path}: +{added:,} new markets ({len(by_ticker):,} unique)")
    return list(by_ticker.values())


# -- ticker parsing -----------------------------------------------------------

def _norm_abbr(code: str) -> Optional[str]:
    c = ABBR_ALIASES.get(code.upper(), code.upper())
    return c if c in VALID_NHL_ABBRS else None


def _table_split(blob: str) -> list[tuple[str, str]]:
    """Brute-force every split point; both halves must be real NHL codes."""
    hits = []
    for i in range(2, len(blob) - 1):
        a, h = _norm_abbr(blob[:i]), _norm_abbr(blob[i:])
        if a and h and a != h:
            hits.append((a, h))
    return hits


def _split_teams(blob: str, yes_raw: str) -> Optional[tuple[str, str]]:
    """
    Split the concatenated away+home blob into (away, home).

    Anchored on the YES suffix rather than brute-forced, because NHL codes are
    a mix of 2- and 3-letter and brute force is ambiguous. The YES leg is one
    of the two teams by construction, so blob either starts with it (YES is
    the away team) or ends with it (YES is the home team).

    Both halves still have to be real Kalshi NHL codes - that check is what
    keeps non-NHL hockey markets out of the tape. If the anchor matches at
    both ends (only possible for short codes) the table split breaks the tie,
    and if it cannot, None is returned rather than guessed.
    """
    anchored: list[tuple[str, str]] = []
    if len(blob) - len(yes_raw) >= 2:
        if blob.startswith(yes_raw):
            anchored.append((yes_raw, blob[len(yes_raw):]))
        if blob.endswith(yes_raw):
            anchored.append((blob[:-len(yes_raw)], yes_raw))

    valid = []
    for a, h in anchored:
        na, nh = _norm_abbr(a), _norm_abbr(h)
        if na and nh and na != nh:
            valid.append((na, nh))

    if len(valid) == 1:
        return valid[0]

    hits = _table_split(blob)
    if len(valid) == 2:
        agree = [c for c in valid if c in hits]
        return agree[0] if len(agree) == 1 else None
    return hits[0] if len(hits) == 1 else None


def parse_ticker(ticker: str) -> Optional[dict]:
    m = TICKER_RE.match(ticker.strip().upper())
    if not m:
        return None
    if m.group("series") != SERIES:
        return None
    mon = MONTH_MAP.get(m.group("mon"))
    if not mon:
        return None
    try:
        game_date = date(2000 + int(m.group("yy")), mon, int(m.group("dd")))
    except ValueError:
        return None

    yes_raw = m.group("yes")
    teams = _split_teams(m.group("teams"), yes_raw)
    if not teams:
        return None
    away, home = teams
    yes = _norm_abbr(yes_raw)
    if yes is None or yes not in (away, home):
        return None
    return {
        "game_date": game_date,
        "away_team": away,
        "home_team": home,
        "is_home_yes": int(yes == home),
    }


# -- trades -------------------------------------------------------------------

def fetch_trades_for_ticker(ticker: str, auth: KalshiAuth,
                            session: requests.Session) -> list[dict]:
    """
    Every trade for one market, no cap.

    Kalshi returns trades NEWEST-FIRST, so any cap discards the OLDEST prints -
    puck drop and the first period, exactly what a live-betting backtest needs.
    There is deliberately no cap parameter.

    Settled markets live in the archived tier, open ones in the live tier.
    Archived is tried first.
    """
    for api_path, url in (
        ("/trade-api/v2/historical/trades", f"{KALSHI_BASE}/historical/trades"),
        ("/trade-api/v2/markets/trades", f"{KALSHI_BASE}/markets/trades"),
    ):
        trades: list[dict] = []
        cursor = None
        while True:
            params = {"ticker": ticker, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            data = _get(session, auth, api_path, url, params)
            if data is None:
                break
            batch = data.get("trades", [])
            trades.extend(batch)
            cursor = data.get("cursor")
            if not cursor or len(batch) < 1000:
                break
            time.sleep(0.12)
        if trades:
            return trades
    return []


def _to_cents(v) -> Optional[float]:
    """'0.6500' -> 65.0. Fractional cents are preserved, not rounded."""
    if v is None:
        return None
    try:
        return float(v) * 100.0
    except (TypeError, ValueError):
        return None


def trades_to_rows(info: dict, market: dict, trades: list[dict],
                   stats: dict) -> list[dict]:
    rows = []
    is_home_yes = info["is_home_yes"]

    for t in trades:
        ts_str = t.get("created_time", "")
        if not ts_str:
            stats["no_timestamp"] += 1
            continue

        yes_c = _to_cents(t.get("yes_price_dollars"))
        if yes_c is None:
            # The live tier historically used an integer-cent `yes_price`.
            # Kept as a fallback so a schema change on either tier degrades
            # loudly rather than silently emptying the price column.
            raw = t.get("yes_price")
            yes_c = float(raw) if raw is not None else None
        if yes_c is None:
            stats["no_price"] += 1
            continue

        no_c = _to_cents(t.get("no_price_dollars"))

        try:
            ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            stats["bad_timestamp"] += 1
            continue

        try:
            cnt = float(t.get("count_fp"))
        except (TypeError, ValueError):
            cnt = float("nan")
            stats["no_count"] += 1

        rows.append({
            "game_date": info["game_date"],
            "away_team": info["away_team"],
            "home_team": info["home_team"],
            "event_ticker": market.get("event_ticker", ""),
            "ticker": market.get("ticker", ""),
            "is_home_yes": is_home_yes,
            "trade_id": t.get("trade_id", ""),
            "timestamp": ts_dt.astimezone(timezone.utc).isoformat(),
            "timestamp_unix": ts_dt.timestamp(),
            "yes_price_cents": yes_c,
            "home_yes_price_cents": yes_c if is_home_yes else 100.0 - yes_c,
            "no_price_cents": no_c,
            "count_fp": cnt,
            "taker_side": t.get("taker_side", ""),
            "taker_outcome_side": t.get("taker_outcome_side", ""),
            "taker_book_side": t.get("taker_book_side", ""),
            "is_block_trade": bool(t.get("is_block_trade", False)),
        })
    return rows


# -- main ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key", default=None,
                    help="Falls back to the KALSHI_API_KEY env var.")
    ap.add_argument("--private-key", default=None,
                    help="Falls back to the KALSHI_PRIVATE_KEY_PATH env var.")
    ap.add_argument("--start", default=SEASON_START,
                    help="YYYY-MM-DD, inclusive. Default 2025-26 opening night.")
    ap.add_argument("--end", default=SEASON_END,
                    help="YYYY-MM-DD, inclusive. Default the last day of the "
                         "2025-26 regular season; playoffs began 2026-04-18.")
    ap.add_argument("--outdir", type=Path, default=Path("data") / "kalshi")
    ap.add_argument("--shard-dir", type=Path, default=None,
                    help="Per-market parquet shards. Default <outdir>/nhl_shards. "
                         "Existing shards are skipped, so an interrupted run resumes.")
    ap.add_argument("--refetch", action="store_true",
                    help="Ignore existing shards and re-pull every market.")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("KALSHI_API_KEY", "")
    key_path = args.private_key or os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
    if not api_key or not key_path:
        print("Need --api-key and --private-key (or KALSHI_API_KEY and "
              "KALSHI_PRIVATE_KEY_PATH env vars).", file=sys.stderr)
        return 1

    auth = KalshiAuth(api_key, key_path)
    session = requests.Session()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    log.info(f"Series {SERIES}, window {start} to {end} (regular season only)")

    shard_dir = args.shard_dir or (args.outdir / "nhl_shards")
    shard_dir.mkdir(parents=True, exist_ok=True)

    markets = fetch_markets(auth, session)
    log.info(f"{len(markets):,} unique markets")

    kept, unparsed, out_of_window = [], [], 0
    for m in markets:
        info = parse_ticker(m.get("ticker", ""))
        if not info:
            unparsed.append(m.get("ticker", ""))
            continue
        if start <= info["game_date"] <= end:
            kept.append((info, m))
        else:
            out_of_window += 1

    log.info(f"{len(kept):,} markets in window; {out_of_window:,} outside the "
             f"window (preseason/playoffs); {len(unparsed):,} tickers unparsed")
    if unparsed:
        log.warning(f"  unparsed examples: {unparsed[:10]}")
    if not kept:
        print("No markets in the window.", file=sys.stderr)
        return 1

    stats = {"no_timestamp": 0, "no_price": 0, "bad_timestamp": 0, "no_count": 0}
    empty_markets = 0
    kept.sort(key=lambda x: (x[0]["game_date"], x[1].get("ticker", "")))
    kept_tickers = {m.get("ticker", "") for _, m in kept}

    for i, (info, m) in enumerate(kept, 1):
        ticker = m.get("ticker", "")
        shard = shard_dir / f"{ticker}.parquet"
        if shard.exists() and not args.refetch:
            continue

        trades = fetch_trades_for_ticker(ticker, auth, session)
        if not trades:
            empty_markets += 1
        rows = trades_to_rows(info, m, trades, stats)
        # An empty shard is still written, so a market with genuinely no
        # trades is not re-fetched on every resume.
        pd.DataFrame(rows).to_parquet(shard, index=False)

        if i % 50 == 0 or i == len(kept):
            log.info(f"  [{i}/{len(kept)}] {ticker}: {len(rows):,} trades")
        time.sleep(0.12)

    # -- assemble -------------------------------------------------------------
    # Only shards belonging to the current window are read. A shard left over
    # from a wider earlier run cannot leak playoff games into the output.
    shards = [s for s in sorted(shard_dir.glob("*.parquet"))
              if s.stem in kept_tickers]
    log.info(f"Assembling {len(shards):,} shards")
    frames = []
    for s in shards:
        df_s = pd.read_parquet(s)
        if len(df_s):
            frames.append(df_s)
    if not frames:
        print("All shards empty - no trades found.", file=sys.stderr)
        return 1

    df = pd.concat(frames, ignore_index=True)
    del frames

    before = len(df)
    df = df[df["trade_id"].astype(str) != ""]
    df = df.drop_duplicates(subset=["trade_id"]).reset_index(drop=True)
    dupes = before - len(df)

    df = df.sort_values(
        ["game_date", "home_team", "timestamp_unix"]
    ).reset_index(drop=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "nhl_kalshi_trades.parquet"
    df.to_parquet(out, index=False)

    # -- report ---------------------------------------------------------------
    per_game = df.groupby(["game_date", "away_team", "home_team"]).size()

    print(f"\nWrote {out}: {len(df):,} trades across {len(per_game):,} games")
    print(f"  date range              : {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"  duplicate trade_ids     : {dupes:,} dropped")
    print(f"  markets with zero trades: {empty_markets:,} of {len(kept):,}")
    print(f"  contracts traded        : {df['count_fp'].sum():,.1f}")
    print(f"  block trades            : {int(df['is_block_trade'].sum()):,}")

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    print(f"\nteams seen ({len(teams)}): {' '.join(teams)}")

    print("\ngames per month:")
    gm = per_game.reset_index()
    gm["month"] = pd.to_datetime(gm["game_date"]).dt.to_period("M").astype(str)
    print(gm.groupby("month").size().to_string())

    print("\ntaker_book_side (which side of the book the print hit):")
    print(df["taker_book_side"].value_counts(dropna=False).to_string())

    print("\ncount_fp:")
    print(df["count_fp"].describe(percentiles=[0.5, 0.95]).round(2).to_string())

    print("\nhome_yes_price_cents:")
    print(df["home_yes_price_cents"].describe(
        percentiles=[0.05, 0.5, 0.95]).round(1).to_string())

    print("\nTrades per game:")
    print(per_game.describe(percentiles=[0.05, 0.5, 0.95]).round(1).to_string())

    # yes + no should sum to 100 cents. Drift means the two price fields are
    # not the two legs of the same print and the tape cannot be trusted.
    both = df[df["no_price_cents"].notna()]
    if len(both):
        s = both["yes_price_cents"] + both["no_price_cents"]
        bad = int((s.sub(100.0).abs() > 0.01).sum())
        print(f"\nyes+no = 100 check: {len(both) - bad:,} of {len(both):,} rows consistent")
        if bad:
            print(f"  {bad:,} rows where yes+no != 100 - inspect before trusting the tape")

    bad_rows = {k: v for k, v in stats.items() if v}
    if bad_rows:
        print(f"\nSkipped/degraded rows: {bad_rows}")

    thin = per_game[per_game < 50]
    if len(thin):
        print(f"\n{len(thin):,} games have fewer than 50 trades - these are where "
              f"forward-filling the last print does the most work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
