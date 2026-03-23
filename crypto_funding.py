#!/usr/bin/env python3
"""
Crypto perpetual futures funding rate toolkit.

Subcommands:
  download   Fetch funding rate history from Kraken, Binance, Deribit, Bybit,
             OKX, Bitfinex, and KuCoin into a local SQLite database.
  simulate   Simulate a basis-trade investment (long spot, short perp) by
             compounding collected funding rates, and plot performance.

Usage:
  python crypto_funding.py download [--days N] [--db PATH]
  python crypto_funding.py simulate [--days N] [--db PATH] [--investment N]

Dependencies:
  pip install python-kraken-sdk pandas requests matplotlib numpy
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
from kraken.futures import Market

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------
TABLE = "funding_rates"
OUTPUT_COLS = ["exchange", "symbol", "timestamp", "fundingRate", "relativeFundingRate"]

KRAKEN_SYMBOLS = [
    "PF_XBTUSD", "PF_ETHUSD",   # linear (multi-collateral)
    "PI_XBTUSD", "PI_ETHUSD",   # inverse (single-collateral)
]
BINANCE_USDM_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
BINANCE_COINM_SYMBOLS = ["BTCUSD_PERP", "ETHUSD_PERP"]
DERIBIT_SYMBOLS = ["BTC-PERPETUAL", "ETH-PERPETUAL"]
BYBIT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
OKX_SYMBOLS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]
BITFINEX_SYMBOLS = ["tBTCF0:USTF0", "tETHF0:USTF0"]
KUCOIN_SYMBOLS = ["XBTUSDTM", "ETHUSDTM"]

SIMULATE_PAIRS_BY_ASSET: dict[str, dict[tuple[str, str], str]] = {
    "BTC": {
        ("KRAKEN", "PF_XBTUSD"):          "BTC Perp (Kraken Linear)",
        ("KRAKEN", "PI_XBTUSD"):          "BTC Perp (Kraken Inverse)",
        ("BINANCE", "BTCUSDT"):           "BTC Perp (Binance USDT-M)",
        ("BINANCE_COINM", "BTCUSD_PERP"): "BTC Perp (Binance COIN-M)",
        ("DERIBIT", "BTC-PERPETUAL"):     "BTC Perp (Deribit)",
        ("BYBIT", "BTCUSDT"):            "BTC Perp (Bybit USDT-M)",
        ("OKX", "BTC-USDT-SWAP"):       "BTC Perp (OKX)",
        ("BITFINEX", "tBTCF0:USTF0"):   "BTC Perp (Bitfinex)",
        ("KUCOIN", "XBTUSDTM"):        "BTC Perp (KuCoin)",
    },
    "ETH": {
        ("KRAKEN", "PF_ETHUSD"):          "ETH Perp (Kraken Linear)",
        ("KRAKEN", "PI_ETHUSD"):          "ETH Perp (Kraken Inverse)",
        ("BINANCE", "ETHUSDT"):           "ETH Perp (Binance USDT-M)",
        ("BINANCE_COINM", "ETHUSD_PERP"): "ETH Perp (Binance COIN-M)",
        ("DERIBIT", "ETH-PERPETUAL"):     "ETH Perp (Deribit)",
        ("BYBIT", "ETHUSDT"):            "ETH Perp (Bybit USDT-M)",
        ("OKX", "ETH-USDT-SWAP"):       "ETH Perp (OKX)",
        ("BITFINEX", "tETHF0:USTF0"):   "ETH Perp (Bitfinex)",
        ("KUCOIN", "ETHUSDTM"):        "ETH Perp (KuCoin)",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _log(tag: str, msg: str, error: bool = False) -> None:
    ts = _utc_now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}", file=sys.stderr if error else sys.stdout, flush=True)


def _iso_z(dt: datetime | pd.Timestamp | pd.Series) -> str | pd.Series:
    if isinstance(dt, pd.Series):
        return dt.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if isinstance(dt, (pd.Timestamp, datetime)):
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    raise TypeError(f"Unsupported type for _iso_z: {type(dt)}")


def _cutoff(days: int) -> datetime:
    return _utc_now() - timedelta(days=days)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLS)


def _normalize_generic(rows: list[dict], symbol: str, exchange: str, days: int,
                        ts_field: str = "fundingTime", ts_unit: str = "ms",
                        rate_field: str = "fundingRate") -> pd.DataFrame:
    """Shared normalizer for Binance, Bybit, OKX, KuCoin, and Deribit data."""
    if not rows:
        return _empty_df()
    df = pd.DataFrame([r for r in rows if isinstance(r, dict)])
    if ts_field not in df.columns:
        return _empty_df()
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df[ts_field], errors="coerce"), unit=ts_unit, utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[df["timestamp"] >= _cutoff(days)]
    df["fundingRate"] = pd.to_numeric(df[rate_field], errors="coerce")
    df["relativeFundingRate"] = pd.NA
    df["timestamp"] = _iso_z(df["timestamp"])
    df["exchange"] = exchange
    df["symbol"] = symbol
    return df[OUTPUT_COLS].drop_duplicates(["exchange", "symbol", "timestamp"])


# ---------------------------------------------------------------------------
# Fetchers — Kraken
# ---------------------------------------------------------------------------

def _fetch_kraken(symbol: str) -> list[dict]:
    with Market() as market:
        data = market.get_historical_funding_rates(symbol=symbol)
    if not isinstance(data, dict) or "rates" not in data:
        raise RuntimeError(f"Unexpected Kraken response for {symbol}: {data!r}")
    return data["rates"]


def _normalize_kraken(rates: list[dict], symbol: str, days: int) -> pd.DataFrame:
    if not rates:
        return _empty_df()
    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ("fundingRate", "relativeFundingRate"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce") if col in df.columns else pd.NA
    df = df.dropna(subset=["timestamp"])
    df = df[df["timestamp"] >= _cutoff(days)]
    df["timestamp"] = _iso_z(df["timestamp"])
    df["exchange"] = "KRAKEN"
    df["symbol"] = symbol
    return df[OUTPUT_COLS].drop_duplicates(["exchange", "symbol", "timestamp"])


# ---------------------------------------------------------------------------
# Fetchers — Binance (unified for USDM and COINM)
# ---------------------------------------------------------------------------

_BINANCE_URLS = {
    "BINANCE":      "https://fapi.binance.com/fapi/v1/fundingRate",
    "BINANCE_COINM": "https://dapi.binance.com/dapi/v1/fundingRate",
}


def _fetch_binance(symbol: str, exchange: str, days: int) -> list[dict]:
    url = _BINANCE_URLS[exchange]
    start = int(_cutoff(days).timestamp() * 1000)
    end = int(_utc_now().timestamp() * 1000)
    out: list[dict] = []
    cur = start
    while True:
        r = requests.get(url, params={"symbol": symbol, "startTime": cur, "endTime": end, "limit": 1000}, timeout=20)
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        out.extend(page)
        next_start = int(page[-1]["fundingTime"]) + 1
        if next_start >= end or next_start == cur:
            break
        cur = next_start
        time.sleep(0.2)
    return out


# ---------------------------------------------------------------------------
# Fetchers — Deribit
# ---------------------------------------------------------------------------
_DERIBIT_BASE = "https://www.deribit.com"
_DERIBIT_CHUNK_DAYS = 31
_DERIBIT_SESSION: requests.Session | None = None


def _get_deribit_session() -> requests.Session:
    global _DERIBIT_SESSION
    if _DERIBIT_SESSION is None:
        _DERIBIT_SESSION = requests.Session()
        _DERIBIT_SESSION.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json",
        })
    return _DERIBIT_SESSION


def _fetch_deribit(instrument: str, days: int) -> list[dict]:
    start_ms = int(_cutoff(days).timestamp() * 1000)
    end_ms = int(_utc_now().timestamp() * 1000)
    chunk = timedelta(days=_DERIBIT_CHUNK_DAYS)
    out: list[dict] = []
    cur = start_ms
    session = _get_deribit_session()
    max_retries = 5
    while cur <= end_ms:
        cur_end = min(cur + int(chunk.total_seconds() * 1000) - 1, end_ms)
        page: list[dict] = []
        for attempt in range(max_retries):
            try:
                r = session.get(
                    f"{_DERIBIT_BASE}/api/v2/public/get_funding_rate_history",
                    params={"instrument_name": instrument, "start_timestamp": cur, "end_timestamp": cur_end},
                    timeout=30,
                )
                r.raise_for_status()
                result = r.json().get("result")
                page = result if isinstance(result, list) else (result.get("data", []) if isinstance(result, dict) else [])
                break
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status == 400:
                    break
                if attempt < max_retries - 1:
                    wait = min(2 ** (attempt + 1), 16)
                    _log("DERIBIT", f"{instrument}: retry {attempt + 1}/{max_retries} after HTTP {status} (wait {wait}s)")
                    time.sleep(wait)
                else:
                    _log("DERIBIT", f"{instrument}: all {max_retries} retries failed (HTTP {status})", error=True)
            except requests.ConnectionError:
                if attempt < max_retries - 1:
                    wait = min(2 ** (attempt + 1), 16)
                    _log("DERIBIT", f"{instrument}: connection error, retry {attempt + 1}/{max_retries} (wait {wait}s)")
                    time.sleep(wait)
                else:
                    _log("DERIBIT", f"{instrument}: all {max_retries} retries failed (connection error)", error=True)
        if page:
            out.extend(page)
        cur = cur_end + 1
        time.sleep(0.3)
    return out


def _normalize_deribit(rows: list[dict], instrument: str, days: int) -> pd.DataFrame:
    if not rows:
        return _empty_df()
    df = pd.DataFrame([r for r in rows if isinstance(r, dict)])
    if "timestamp" not in df.columns:
        return _empty_df()
    rate_col = "interest_8h" if "interest_8h" in df.columns else ("funding_rate" if "funding_rate" in df.columns else None)
    if rate_col is None:
        return _empty_df()
    return _normalize_generic(rows, instrument, "DERIBIT", days, ts_field="timestamp", rate_field=rate_col)


# ---------------------------------------------------------------------------
# Fetchers — Bybit
# ---------------------------------------------------------------------------
_BYBIT_BASE = "https://api.bybit.com"


def _fetch_bybit(symbol: str, days: int) -> list[dict]:
    start_ms = int(_cutoff(days).timestamp() * 1000)
    end_ms = int(_utc_now().timestamp() * 1000)
    out: list[dict] = []
    while True:
        try:
            r = requests.get(
                f"{_BYBIT_BASE}/v5/market/funding/history",
                params={"category": "linear", "symbol": symbol, "endTime": end_ms, "limit": 200},
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            r2 = requests.get(
                f"{_BYBIT_BASE}/derivatives/v3/public/funding/history",
                params={"category": "linear", "symbol": symbol, "endTime": end_ms, "limit": 200},
                timeout=20,
            )
            r2.raise_for_status()
            data = r2.json()

        result = data.get("result") or {}
        lst = result.get("list") or result.get("data") if isinstance(result, dict) else None
        if lst is None:
            lst = data.get("list") or data.get("data")
        if not lst:
            break

        earliest_ms = None
        for item in lst:
            if not isinstance(item, dict):
                continue
            ts = item.get("fundingRateTimestamp") or item.get("fundingTime") or item.get("timestamp")
            fr = item.get("fundingRate")
            if ts is None or fr is None:
                continue
            ts_int = int(ts)
            out.append({"fundingTime": ts_int, "fundingRate": fr})
            if earliest_ms is None or ts_int < earliest_ms:
                earliest_ms = ts_int

        if earliest_ms is None or earliest_ms <= start_ms:
            break
        end_ms = earliest_ms - 1
        time.sleep(0.2)
    return out


# ---------------------------------------------------------------------------
# Fetchers — OKX
# ---------------------------------------------------------------------------
_OKX_BASE = "https://www.okx.com"


def _fetch_okx(symbol: str, days: int) -> list[dict]:
    start_ms = int(_cutoff(days).timestamp() * 1000)
    out: list[dict] = []
    after = ""
    while True:
        params: dict[str, Any] = {"instId": symbol, "limit": "100"}
        if after:
            params["after"] = after
        r = requests.get(f"{_OKX_BASE}/api/v5/public/funding-rate-history", params=params, timeout=20)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        for item in data:
            ts = int(item["fundingTime"])
            if ts < start_ms:
                return out
            out.append({"fundingTime": ts, "fundingRate": item["fundingRate"]})
        after = data[-1]["fundingTime"]
        time.sleep(0.25)
    return out


# ---------------------------------------------------------------------------
# Fetchers — Bitfinex
# ---------------------------------------------------------------------------
_BITFINEX_BASE = "https://api-pub.bitfinex.com"


def _fetch_bitfinex(symbol: str, days: int) -> list[dict]:
    start_ms = int(_cutoff(days).timestamp() * 1000)
    end_ms = int(_utc_now().timestamp() * 1000)
    out: list[dict] = []
    cur = start_ms
    while cur < end_ms:
        r = requests.get(
            f"{_BITFINEX_BASE}/v2/status/deriv/{symbol}/hist",
            params={"start": cur, "end": end_ms, "limit": 5000, "sort": 1},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        if not data or not isinstance(data, list):
            break
        for row in data:
            if not isinstance(row, list) or len(row) < 12:
                continue
            ts, funding_rate = row[0], row[11]
            if ts is not None and funding_rate is not None:
                out.append({"fundingTime": int(ts), "fundingRate": funding_rate})
        last_ts = int(data[-1][0])
        if last_ts <= cur:
            break
        cur = last_ts + 1
        time.sleep(0.3)
    return out


def _normalize_bitfinex(rows: list[dict], symbol: str, days: int) -> pd.DataFrame:
    """Bitfinex returns per-minute snapshots; resample to 8h windows."""
    if not rows:
        return _empty_df()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["fundingTime"], errors="coerce"), unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[df["timestamp"] >= _cutoff(days)]
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.set_index("timestamp").sort_index()
    df_8h = df["fundingRate"].resample("8h").mean().dropna().reset_index()
    df_8h.columns = ["timestamp", "fundingRate"]
    df_8h["relativeFundingRate"] = pd.NA
    df_8h["timestamp"] = _iso_z(df_8h["timestamp"])
    df_8h["exchange"] = "BITFINEX"
    df_8h["symbol"] = symbol
    return df_8h[OUTPUT_COLS].drop_duplicates(["exchange", "symbol", "timestamp"])


# ---------------------------------------------------------------------------
# Fetchers — KuCoin
# ---------------------------------------------------------------------------
_KUCOIN_BASE = "https://api-futures.kucoin.com"


def _fetch_kucoin(symbol: str, days: int) -> list[dict]:
    start_ms = int(_cutoff(days).timestamp() * 1000)
    end_ms = int(_utc_now().timestamp() * 1000)
    out: list[dict] = []
    cur_from = start_ms
    while cur_from < end_ms:
        cur_to = min(cur_from + 31 * 24 * 3600 * 1000, end_ms)
        r = requests.get(
            f"{_KUCOIN_BASE}/api/v1/contract/funding-rates",
            params={"symbol": symbol, "from": cur_from, "to": cur_to},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        lst = data.get("dataList", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        if not lst:
            cur_from = cur_to + 1
            time.sleep(0.25)
            continue
        for item in lst:
            ts = item.get("timePoint") or item.get("timepoint") or item.get("fundingTime")
            fr = item.get("fundingRate")
            if ts is not None and fr is not None:
                out.append({"fundingTime": int(ts), "fundingRate": fr})
        cur_from = cur_to + 1
        time.sleep(0.25)
    return out


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

def _ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE,))
    if cur.fetchone() is None:
        conn.execute(f"""
            CREATE TABLE {TABLE} (
                exchange TEXT NOT NULL, symbol TEXT NOT NULL, timestamp TEXT NOT NULL,
                fundingRate REAL, relativeFundingRate REAL,
                PRIMARY KEY (exchange, symbol, timestamp)
            )
        """)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_ts ON {TABLE}(timestamp)")
        conn.commit()
        return

    # Migrate legacy schema (no exchange column) if needed
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({TABLE})").fetchall()}
    if "exchange" in cols:
        return
    tmp = f"{TABLE}_v2"
    conn.execute(f"""
        CREATE TABLE {tmp} (
            exchange TEXT NOT NULL, symbol TEXT NOT NULL, timestamp TEXT NOT NULL,
            fundingRate REAL, relativeFundingRate REAL,
            PRIMARY KEY (exchange, symbol, timestamp)
        )
    """)
    conn.execute(f"INSERT INTO {tmp} SELECT 'KRAKEN', symbol, timestamp, fundingRate, relativeFundingRate FROM {TABLE}")
    conn.execute(f"DROP TABLE {TABLE}")
    conn.execute(f"ALTER TABLE {tmp} RENAME TO {TABLE}")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_ts ON {TABLE}(timestamp)")
    conn.commit()


def _write_df(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    clean = df.astype(object).where(pd.notna(df), None)
    rows = list(clean.itertuples(index=False, name=None))
    conn.executemany(f"""
        INSERT INTO {TABLE} (exchange, symbol, timestamp, fundingRate, relativeFundingRate)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(exchange, symbol, timestamp) DO UPDATE SET
            fundingRate = excluded.fundingRate,
            relativeFundingRate = excluded.relativeFundingRate
    """, rows)
    conn.commit()


# ---------------------------------------------------------------------------
# Download command
# ---------------------------------------------------------------------------

def _fetch_all(days: int) -> list[pd.DataFrame]:
    jobs: list[tuple[str, str, Any]] = []

    for sym in KRAKEN_SYMBOLS:
        jobs.append(("KRAKEN", sym, lambda s=sym: _normalize_kraken(_fetch_kraken(s), s, days)))
    for sym in BINANCE_USDM_SYMBOLS:
        jobs.append(("BINANCE USDM", sym, lambda s=sym: _normalize_generic(_fetch_binance(s, "BINANCE", days), s, "BINANCE", days)))
    for sym in BINANCE_COINM_SYMBOLS:
        jobs.append(("BINANCE COINM", sym, lambda s=sym: _normalize_generic(_fetch_binance(s, "BINANCE_COINM", days), s, "BINANCE_COINM", days)))
    for sym in DERIBIT_SYMBOLS:
        jobs.append(("DERIBIT", sym, lambda s=sym: _normalize_deribit(_fetch_deribit(s, days), s, days)))
    for sym in BYBIT_SYMBOLS:
        jobs.append(("BYBIT", sym, lambda s=sym: _normalize_generic(_fetch_bybit(s, days), s, "BYBIT", days)))
    for sym in OKX_SYMBOLS:
        jobs.append(("OKX", sym, lambda s=sym: _normalize_generic(_fetch_okx(s, days), s, "OKX", days)))
    for sym in BITFINEX_SYMBOLS:
        jobs.append(("BITFINEX", sym, lambda s=sym: _normalize_bitfinex(_fetch_bitfinex(s, days), s, days)))
    for sym in KUCOIN_SYMBOLS:
        jobs.append(("KUCOIN", sym, lambda s=sym: _normalize_generic(_fetch_kucoin(s, days), s, "KUCOIN", days)))

    def _run_job(tag: str, sym: str, fetch_fn: Any) -> pd.DataFrame | None:
        _log(tag, f"Fetching {sym} ...")
        try:
            df = fetch_fn()
            _log(tag, f"{sym}: {len(df)} records after filtering to last {days} days.")
            return df if not df.empty else None
        except Exception as e:
            _log(tag, f"{sym}: {e}", error=True)
            return None

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_run_job, tag, sym, fn): (tag, sym) for tag, sym, fn in jobs}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                frames.append(result)

    return frames


def cmd_download(args: argparse.Namespace) -> None:
    t0 = time.monotonic()
    frames = _fetch_all(args.days)
    if not frames:
        _log("DOWNLOAD", "No data collected from any exchange.", error=True)
        sys.exit(1)

    df_all = pd.concat(frames, ignore_index=True).sort_values(["exchange", "symbol", "timestamp"]).reset_index(drop=True)
    conn = sqlite3.connect(args.db)
    try:
        _ensure_schema(conn)
        _write_df(conn, df_all)
    finally:
        conn.close()

    _log("DOWNLOAD", f"Saved {len(df_all)} rows to {args.db} in {time.monotonic() - t0:.1f}s.")


# ---------------------------------------------------------------------------
# Simulation command
# ---------------------------------------------------------------------------

def _query_rates(conn: sqlite3.Connection, exchange: str, symbol: str, start_z: str, end_z: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        f"SELECT * FROM {TABLE} WHERE exchange=? AND symbol=? AND timestamp>=? AND timestamp<? ORDER BY timestamp",
        conn, params=(exchange, symbol, start_z, end_z),
    )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ("fundingRate", "relativeFundingRate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _expand_8h_to_hourly(df_8h: pd.DataFrame, rate_col: str) -> pd.DataFrame:
    """Convert 8-hour funding rates to hourly via r_hour = (1 + r_8h)^(1/8) - 1, vectorized."""
    df = df_8h.dropna(subset=[rate_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "rate_hourly"])
    one_plus = 1.0 + df[rate_col]
    df = df[one_plus > 0].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "rate_hourly"])
    r_hour = one_plus[df.index].pow(1.0 / 8.0) - 1.0

    # Vectorized expansion: repeat each row 8 times with hour offsets
    ts = df["timestamp"].values
    rates = r_hour.values
    offsets = np.arange(7, -1, -1)  # 7,6,5,...,0
    expanded_ts = np.repeat(ts, 8) - np.tile(offsets, len(ts)) * np.timedelta64(1, "h")
    expanded_rates = np.repeat(rates, 8)

    out = pd.DataFrame({"timestamp": expanded_ts, "rate_hourly": expanded_rates})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _to_hourly(exchange: str, df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw rates to a uniform hourly series [timestamp, rate_hourly]."""
    if df.empty:
        return df
    if exchange == "KRAKEN":
        out = df[["timestamp"]].copy()
        out["rate_hourly"] = pd.to_numeric(df["relativeFundingRate"], errors="coerce")
        return out.dropna(subset=["rate_hourly"]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # All other exchanges use 8h funding rates
    return _expand_8h_to_hourly(df, "fundingRate")


def _simulate_compounding(rates: pd.DataFrame, initial: float) -> pd.DataFrame:
    """Compound hourly rates using vectorized numpy cumprod."""
    values = initial * np.cumprod(1.0 + rates["rate_hourly"].values)
    out = rates.copy()
    out["portfolio_value"] = values
    return out


def _simulate_asset(asset: str, conn: sqlite3.Connection, args: argparse.Namespace,
                    start_z: str, end_z: str, start_date: str, end_date: str) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Run simulation for a single asset, return (histories, finals)."""
    simulate_pairs = SIMULATE_PAIRS_BY_ASSET.get(asset)
    if simulate_pairs is None:
        _log("SIMULATE", f"Unknown asset '{asset}'. Choose from: {', '.join(SIMULATE_PAIRS_BY_ASSET.keys())}", error=True)
        return {}, {}

    histories: dict[str, pd.DataFrame] = {}
    finals: dict[str, float] = {}

    for (exchange, symbol), label in simulate_pairs.items():
        raw = _query_rates(conn, exchange, symbol, start_z, end_z)
        if raw.empty:
            _log("SIMULATE", f"Skip — no data for ({exchange}, {symbol}) in range.")
            continue
        hourly = _to_hourly(exchange, raw)
        mask = (hourly["timestamp"] >= pd.to_datetime(start_z)) & (hourly["timestamp"] < pd.to_datetime(end_z))
        hourly = hourly.loc[mask].drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if hourly.empty:
            _log("SIMULATE", f"Skip — no hourly rates for ({exchange}, {symbol}).")
            continue
        hist = _simulate_compounding(hourly, args.investment)
        histories[label] = hist
        finals[label] = hist["portfolio_value"].iloc[-1]

    if finals:
        _log("SIMULATE", f"  [{asset}] Investment per leg: {args.investment:,.2f}")
        _log("SIMULATE", f"  [{asset}] Period: {start_date} to {end_date} ({args.days} days)")
        for label, fv in finals.items():
            ret_pct = (fv / args.investment - 1) * 100
            pa_yield = ((fv / args.investment) ** (365.0 / args.days) - 1) * 100
            _log("SIMULATE", f"    {label}: {fv:,.2f}  ({ret_pct:+.2f}% / {pa_yield:+.2f}% p.a.)")

    return histories, finals


def _plot_asset(asset: str, histories: dict[str, pd.DataFrame], finals: dict[str, float],
                investment: float, days: int, start_date: str, end_date: str, fig_num: int) -> None:
    """Create a chart for a single asset."""
    import matplotlib.pyplot as plt

    plt.figure(fig_num, figsize=(11, 6))
    for label, fv in finals.items():
        hist = histories[label]
        ret_pct = (fv / investment - 1) * 100
        pa_yield = ((fv / investment) ** (365.0 / days) - 1) * 100
        plt.plot(hist["timestamp"], hist["portfolio_value"],
                 label=f"{label}  ({ret_pct:+.2f}% / {pa_yield:+.2f}% p.a.)")
    plt.xlabel("Date (UTC)")
    plt.ylabel("Value (USD)")
    plt.title(f"{asset} Basis Trade: Funding Rate Compounding — {days} days ({start_date} to {end_date})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def cmd_simulate(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    start_date = (datetime.now(timezone.utc).date() - timedelta(days=args.days)).isoformat()
    end_date = datetime.now(timezone.utc).date().isoformat()
    start_z = pd.to_datetime(start_date, utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    end_z = (pd.to_datetime(end_date, utc=True).normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    asset = args.asset.upper()
    assets = ["BTC", "ETH"] if asset == "BOTH" else [asset]

    conn = sqlite3.connect(args.db)
    any_data = False

    for i, a in enumerate(assets):
        histories, finals = _simulate_asset(a, conn, args, start_z, end_z, start_date, end_date)
        if finals:
            any_data = True
            _plot_asset(a, histories, finals, args.investment, args.days, start_date, end_date, fig_num=i + 1)

    conn.close()

    if not any_data:
        _log("SIMULATE", "No data available for any selected pair in the range.", error=True)
        sys.exit(1)

    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import config

    parser = argparse.ArgumentParser(description="Crypto perpetual funding rate tracker & simulator")
    parser.add_argument("--db", default=config.DB, help=f"SQLite database path (default: {config.DB})")
    parser.add_argument("--days", type=int, default=config.DAYS, help=f"Lookback period in days (default: {config.DAYS})")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("download", help="Download funding rates from all exchanges")

    sim = sub.add_parser("simulate", help="Simulate basis-trade investment and plot results")
    sim.add_argument("--investment", type=float, default=config.INVESTMENT, help=f"Initial investment amount (default: {config.INVESTMENT})")
    sim.add_argument("--asset", default=config.ASSET, help=f"Asset to simulate: BTC or ETH (default: {config.ASSET})")

    args = parser.parse_args()

    if args.command is None:
        args.command = config.MODE
        if not hasattr(args, "investment"):
            args.investment = config.INVESTMENT
        if not hasattr(args, "asset"):
            args.asset = config.ASSET

    if args.command == "download":
        cmd_download(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "both":
        cmd_download(args)
        if not hasattr(args, "investment"):
            args.investment = config.INVESTMENT
        cmd_simulate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
