"""AKShare downloader for public A-share daily data."""

from __future__ import annotations

import time
from typing import Iterable

import pandas as pd

from src.utils.io import save_parquet


FALLBACK_A_SHARE_UNIVERSE = [
    "600000.SH", "600009.SH", "600010.SH", "600015.SH", "600016.SH", "600018.SH", "600019.SH", "600028.SH",
    "600030.SH", "600031.SH", "600036.SH", "600048.SH", "600050.SH", "600061.SH", "600085.SH", "600089.SH",
    "600104.SH", "600111.SH", "600150.SH", "600176.SH", "600196.SH", "600276.SH", "600309.SH", "600346.SH",
    "600406.SH", "600436.SH", "600438.SH", "600519.SH", "600547.SH", "600570.SH", "600584.SH", "600585.SH",
    "600660.SH", "600690.SH", "600745.SH", "600760.SH", "600795.SH", "600809.SH", "600837.SH", "600887.SH",
    "600893.SH", "600900.SH", "600905.SH", "600918.SH", "600919.SH", "600926.SH", "600941.SH", "600958.SH",
    "600999.SH", "601006.SH", "601012.SH", "601066.SH", "601088.SH", "601111.SH", "601138.SH", "601166.SH",
    "601186.SH", "601211.SH", "601225.SH", "601288.SH", "601318.SH", "601319.SH", "601328.SH", "601336.SH",
    "601390.SH", "601398.SH", "601601.SH", "601628.SH", "601668.SH", "601688.SH", "601728.SH", "601766.SH",
    "601800.SH", "601816.SH", "601818.SH", "601857.SH", "601888.SH", "601898.SH", "601899.SH", "601901.SH",
    "601919.SH", "601985.SH", "601988.SH", "601989.SH", "601995.SH", "601998.SH", "603259.SH", "603288.SH",
    "603501.SH", "603799.SH", "603986.SH", "000001.SZ", "000002.SZ", "000063.SZ", "000066.SZ", "000069.SZ",
    "000100.SZ", "000157.SZ", "000166.SZ", "000301.SZ", "000333.SZ", "000338.SZ", "000408.SZ", "000425.SZ",
    "000538.SZ", "000568.SZ", "000596.SZ", "000625.SZ", "000651.SZ", "000661.SZ", "000708.SZ", "000725.SZ",
    "000776.SZ", "000786.SZ", "000858.SZ", "000876.SZ", "000895.SZ", "000938.SZ", "000963.SZ", "000977.SZ",
    "001979.SZ", "002001.SZ", "002007.SZ", "002008.SZ", "002027.SZ", "002049.SZ", "002050.SZ", "002074.SZ",
    "002129.SZ", "002142.SZ", "002179.SZ", "002230.SZ", "002241.SZ", "002271.SZ", "002304.SZ", "002311.SZ",
    "002352.SZ", "002371.SZ", "002410.SZ", "002415.SZ", "002459.SZ", "002460.SZ", "002466.SZ", "002475.SZ",
    "002493.SZ", "002594.SZ", "002601.SZ", "002602.SZ", "002607.SZ", "002648.SZ", "002709.SZ", "002714.SZ",
    "002736.SZ", "002812.SZ", "002821.SZ", "002841.SZ", "002916.SZ", "002938.SZ", "003816.SZ", "300014.SZ",
    "300015.SZ", "300033.SZ", "300059.SZ", "300122.SZ", "300124.SZ", "300274.SZ", "300308.SZ", "300316.SZ",
    "300347.SZ", "300408.SZ", "300413.SZ", "300433.SZ", "300450.SZ", "300454.SZ", "300496.SZ", "300498.SZ",
    "300502.SZ", "300601.SZ", "300628.SZ", "300661.SZ", "300750.SZ", "300760.SZ", "300782.SZ",
]


def _suffix_code(symbol: str) -> str:
    """Attach exchange suffix to a six-digit A-share code."""
    symbol = str(symbol).zfill(6)
    return f"{symbol}.SH" if symbol.startswith(("5", "6", "9")) else f"{symbol}.SZ"


def _normalize_akshare_hist(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Map AKShare Chinese columns to the project schema."""
    rename_map = {
        "日期": "date",
        "股票代码": "code",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover_rate",
        "turnover": "turnover_rate",
    }
    out = df.rename(columns=rename_map).copy()
    if not {"date", "open", "high", "low", "close", "volume", "amount"}.issubset(out.columns) and len(out.columns) >= 8:
        cols = list(out.columns)
        positional_map = {
            cols[0]: "date",
            cols[1]: "code",
            cols[2]: "open",
            cols[3]: "close",
            cols[4]: "high",
            cols[5]: "low",
            cols[6]: "volume",
            cols[7]: "amount",
        }
        if len(cols) >= 12:
            positional_map[cols[11]] = "turnover_rate"
        out = df.rename(columns=positional_map).copy()
    out["code"] = code
    if "market_cap" not in out.columns and {"close", "outstanding_share"}.issubset(out.columns):
        out["market_cap"] = pd.to_numeric(out["close"], errors="coerce") * pd.to_numeric(out["outstanding_share"], errors="coerce")
    keep_cols = ["date", "code", "open", "high", "low", "close", "volume", "amount", "turnover_rate", "market_cap"]
    missing = [col for col in ["date", "open", "high", "low", "close", "volume", "amount"] if col not in out.columns]
    if missing:
        raise ValueError(f"AKShare response for {code} missing columns after normalization: {missing}")
    out = out[[col for col in keep_cols if col in out.columns]]
    out["date"] = pd.to_datetime(out["date"])
    for col in out.columns:
        if col not in {"date", "code"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def get_akshare_universe(universe_size: int = 100) -> list[str]:
    """Return a medium-size A-share universe, preferring AKShare spot data when available."""
    try:
        import akshare as ak

        spot = ak.stock_zh_a_spot_em()
        code_col = "代码" if "代码" in spot.columns else "股票代码"
        amount_col = "成交额" if "成交额" in spot.columns else None
        spot = spot.copy()
        spot[code_col] = spot[code_col].astype(str).str.zfill(6)
        spot = spot[spot[code_col].str.match(r"^(000|001|002|003|300|600|601|603|605|688)")]
        if amount_col is not None:
            spot[amount_col] = pd.to_numeric(spot[amount_col], errors="coerce")
            spot = spot.sort_values(amount_col, ascending=False)
        return [_suffix_code(code) for code in spot[code_col].head(universe_size).tolist()]
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] AKShare spot universe unavailable; using built-in universe: {exc}")
        return FALLBACK_A_SHARE_UNIVERSE[:universe_size]


def download_akshare_daily(
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    out_path: str,
    adjust: str = "qfq",
    source: str = "sina",
    retry_times: int = 3,
    retry_sleep_seconds: float = 1.5,
    min_valid_stocks: int = 1,
) -> pd.DataFrame:
    """Download daily A-share OHLCV data through AKShare with per-stock retries."""
    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError("akshare is required for real-data download. Install with `pip install akshare`.") from exc

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for i, code in enumerate(codes, start=1):
        symbol = code.split(".")[0]
        last_error: Exception | None = None
        for attempt in range(1, retry_times + 1):
            try:
                if source == "sina":
                    sina_symbol = ("sh" if code.endswith(".SH") else "sz") + symbol
                    hist = ak.stock_zh_a_daily(
                        symbol=sina_symbol,
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust,
                    )
                else:
                    hist = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust,
                        timeout=20,
                    )
                normalized = _normalize_akshare_hist(hist, code)
                if normalized.empty:
                    raise ValueError("empty historical data")
                frames.append(normalized)
                print(f"[INFO] {i:03d} downloaded {code}: {len(normalized)} rows")
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < retry_times:
                    time.sleep(retry_sleep_seconds)
        else:
            failed.append(code)
            print(f"[WARN] Failed to download {code}: {last_error}")

    if len(frames) < min_valid_stocks:
        raise RuntimeError(
            f"Only {len(frames)} valid stocks downloaded, below min_valid_stocks={min_valid_stocks}. "
            "AKShare/network data is not stable enough for this run."
        )
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "code"])
    save_parquet(df, out_path)
    print(f"[INFO] AKShare download complete: {df['code'].nunique()} valid stocks, {len(df)} rows, {len(failed)} failed.")
    return df
