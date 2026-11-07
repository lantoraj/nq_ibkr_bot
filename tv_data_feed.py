"""
TradingView Real-Time / Hybrid Futures 2-Minute OHLCV Data Feed for NQ / MNQ.

Why this exists:
- IBKR Paper Trading (`DUA...`) without a paid CME Level 1 subscription delays `MNQ` / `NQ`
  historical and streaming bars by 10 minutes (`600s`), while executing orders against real-time prices.
- TradingView's WebSocket (`wss://data.tradingview.com/socket.io/websocket`) provides:
  1. Direct real-time CME futures (`CME_MINI:MNQZ2026` / `CME_MINI:MNQ1!`) when `TRADINGVIEW_AUTH_TOKEN`
     is configured for a TradingView account with CME data, OR
  2. Unauthenticated Hybrid Real-Time mode (`0-second delay`):
     - Requests standard `1`-minute bars (since unauthenticated TradingView restricts `2`-minute
       `custom_resolution`, whereas `1`-minute is standard and streams in <0.5s) and aggregates
       them into exact `2`-minute OHLCV bars (`closed='left', label='left'`).
     - Uses true `CME_MINI:MNQZ2026` (or `CME_MINI:MNQ1!`) bars for all history up to `T - 10 min`
       (preserving exact CME Session VWAP, volume profile, and PDH/PDL), AND
     - Uses TradingView's 0-second real-time Nasdaq-100 feed (`CAPITALCOM:US100`, with automatic fallback
       to `FOREXCOM:NSXUSD`) for the most recent 10 minutes (`T - 10 min` to `T`), dynamically adjusted
       by the rolling 30-bar median futures fair-value basis (`median(CME_close - CFD_close)`) rounded
       to `0.25` CME ticks and volume-scaled to CME contract volume.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
import socket
import ssl
import string
import struct
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from indicators import round_to_tick

logger = logging.getLogger("nq_ibkr_bot")


def _random_session_id(prefix: str) -> str:
    chars = string.ascii_lowercase + string.digits
    return f"{prefix}_{''.join(random.choices(chars, k=12))}"


def _tv_pack(func: str, params: list) -> str:
    payload = json.dumps({"m": func, "p": params}, separators=(",", ":"))
    return f"~m~{len(payload)}~m~{payload}"


def _tv_parse_messages(raw: str) -> List[str]:
    """Extract all payload strings from TradingView `~m~<len>~m~<payload>` framing."""
    pattern = re.compile(r"~m~(\d+)~m~")
    pos = 0
    out: List[str] = []
    while pos < len(raw):
        m = pattern.search(raw, pos)
        if not m:
            break
        length = int(m.group(1))
        start = m.end()
        out.append(raw[start : start + length])
        pos = start + length
    return out


class _TVWebSocket:
    """Minimal zero-dependency TLS WebSocket client tailored for TradingView's data socket."""

    def __init__(self, timeout: float = 6.0) -> None:
        self.timeout = timeout
        self.sock: Optional[ssl.SSLSocket] = None
        self._buf = bytearray()

    def connect(self) -> None:
        ctx = ssl.create_default_context()
        raw_sock = socket.create_connection(("data.tradingview.com", 443), timeout=self.timeout)
        self.sock = ctx.wrap_socket(raw_sock, server_hostname="data.tradingview.com")
        self.sock.settimeout(self.timeout)

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            "GET /socket.io/websocket?from=chart%2F&type=chart HTTP/1.1\r\n"
            "Host: data.tradingview.com\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "Origin: https://www.tradingview.com\r\n"
            "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)\r\n\r\n"
        )
        self.sock.sendall(req.encode("ascii"))

        raw = bytearray()
        while b"\r\n\r\n" not in raw:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("TradingView WebSocket closed during HTTP upgrade handshake")
            raw.extend(chunk)

        sep_idx = raw.index(b"\r\n\r\n") + 4
        header_text = raw[:sep_idx].decode("latin1", errors="ignore")
        self._buf = bytearray(raw[sep_idx:])
        if "101" not in header_text.splitlines()[0]:
            raise ConnectionError(f"TradingView WebSocket handshake failed: {header_text.splitlines()[0]}")

    def send_text(self, text: str) -> None:
        if self.sock is None:
            raise ConnectionError("WebSocket is not connected")
        data = text.encode("utf-8")
        mask = os.urandom(4)
        header = bytearray([0x81])
        n = len(data)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", n))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", n))
        header.extend(mask)
        masked = bytearray(b ^ mask[i % 4] for i, b in enumerate(data))
        self.sock.sendall(header + masked)

    def send_cmd(self, func: str, params: list) -> None:
        self.send_text(_tv_pack(func, params))

    def _recv_exact(self, n: int) -> bytes:
        if self.sock is None:
            raise ConnectionError("WebSocket is not connected")
        while len(self._buf) < n:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("TradingView WebSocket closed while reading frame")
            self._buf.extend(chunk)
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def recv_text(self) -> str:
        while True:
            hdr = self._recv_exact(2)
            opcode = hdr[0] & 0x0F
            masked = (hdr[1] & 0x80) != 0
            length = hdr[1] & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv_exact(8))[0]
            mask = self._recv_exact(4) if masked else b""
            payload = self._recv_exact(length)
            if masked:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

            if opcode == 0x8:  # Close frame
                raise ConnectionError("TradingView WebSocket closed by remote peer")
            if opcode == 0x9:  # Ping -> send Pong
                if self.sock is not None:
                    pong = bytearray([0x8A, 0x80, 0, 0, 0, 0])
                    self.sock.sendall(pong)
                continue
            if opcode in (0x1, 0x2):
                return payload.decode("utf-8", errors="ignore")

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


def ibkr_symbol_to_tv_cme_symbol(local_symbol: Optional[str], base_symbol: str = "MNQ") -> str:
    """
    Convert an IBKR localSymbol such as `'MNQZ6'` or `'NQH7'` into TradingView's CME symbol
    (e.g. `'CME_MINI:MNQZ2026'`), falling back to `'CME_MINI:MNQ1!'`.
    """
    if local_symbol:
        m = re.match(r"^([A-Z]+?)([FGHJKMNQUVXZ])(\d)$", local_symbol.strip().upper())
        if m:
            root, month_code, year_digit = m.groups()
            decade = (datetime.now(timezone.utc).year // 10) * 10
            year = decade + int(year_digit)
            if year < datetime.now(timezone.utc).year - 1:
                year += 10
            return f"CME_MINI:{root}{month_code}{year}"
    return f"CME_MINI:{base_symbol.upper()}1!"


def resample_1m_to_target(df_1m: pd.DataFrame, target_minutes: int = 2) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV bars (`timestamp` in UTC) into exact `target_minutes` (e.g. 2-minute)
    even-minute-aligned bars (`closed='left', label='left'`), matching IBKR's 2-minute bar timestamps.
    """
    if df_1m is None or df_1m.empty or target_minutes <= 1:
        return df_1m.copy() if df_1m is not None else pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    tmp = df_1m.sort_values("timestamp").set_index("timestamp")
    resampled = (
        tmp.resample(f"{target_minutes}min", closed="left", label="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return resampled[["timestamp", "open", "high", "low", "close", "volume"]]


class TradingViewDataFeed:
    """
    Fetches 2-minute OHLCV bars from TradingView for NQ/MNQ futures with automatic
    real-time basis stitching when CME futures are delayed by 10 minutes.
    """

    def __init__(
        self,
        auth_token: Optional[str] = None,
        realtime_proxy_symbol: str = "CAPITALCOM:US100",
        fallback_proxy_symbol: str = "FOREXCOM:NSXUSD",
        tick_size: float = 0.25,
        timeout: float = 6.0,
    ) -> None:
        self.auth_token = (
            auth_token
            or os.getenv("TRADINGVIEW_AUTH_TOKEN", "").strip()
            or "unauthorized_user_token"
        )
        self.realtime_proxy_symbol = realtime_proxy_symbol
        self.fallback_proxy_symbol = fallback_proxy_symbol
        self.tick_size = tick_size
        self.timeout = timeout
        self.last_basis_points: float = 0.0
        self.last_cme_update_mode: str = "unknown"
        self.last_stitched_bars_count: int = 0

    def _fetch_dual_series(
        self,
        cme_symbol: str,
        proxy_symbol: str,
        bar_size: str = "2 mins",
        n_bars: int = 600,
    ) -> Tuple[pd.DataFrame, str, pd.DataFrame, str]:
        """
        Fetch `cme_symbol` and `proxy_symbol` simultaneously over a single TradingView WebSocket.
        Requests `'1'` (1-minute) when `bar_size` is `'2 mins'` (since unauthenticated TradingView
        blocks `'2'` as `custom_resolution`) and resamples to exact 2-minute bars.
        Returns `(cme_df, cme_update_mode, proxy_df, proxy_update_mode)`.
        """
        m_int = re.search(r"(\d+)", bar_size)
        target_minutes = int(m_int.group(1)) if m_int else 2
        if target_minutes == 2:
            tv_interval = "1"
            req_cme_bars = min(4500, n_bars * 2)
            req_prx_bars = 240
        else:
            tv_interval = str(target_minutes)
            req_cme_bars = n_bars
            req_prx_bars = 120

        ws = _TVWebSocket(timeout=self.timeout)
        cs_cme = _random_session_id("cs_cme")
        cs_prx = _random_session_id("cs_prx")

        cme_bars: Dict[int, Tuple[float, float, float, float, float]] = {}
        prx_bars: Dict[int, Tuple[float, float, float, float, float]] = {}
        cme_mode = "unknown"
        prx_mode = "unknown"
        cme_done = False
        prx_done = False

        try:
            ws.connect()
            init_raw = ws.recv_text()
            for msg in _tv_parse_messages(init_raw):
                if msg.startswith("~h~"):
                    ws.send_text(f"~m~{len(msg)}~m~{msg}")

            ws.send_cmd("set_auth_token", [self.auth_token])

            ws.send_cmd("chart_create_session", [cs_cme, ""])
            ws.send_cmd("switch_timezone", [cs_cme, "Etc/UTC"])
            ws.send_cmd(
                "resolve_symbol",
                [
                    cs_cme,
                    "sds_sym_1",
                    "=" + json.dumps({"symbol": cme_symbol, "adjustment": "splits", "session": "extended"}),
                ],
            )
            ws.send_cmd("create_series", [cs_cme, "sds_1", "s1", "sds_sym_1", tv_interval, req_cme_bars, ""])

            ws.send_cmd("chart_create_session", [cs_prx, ""])
            ws.send_cmd("switch_timezone", [cs_prx, "Etc/UTC"])
            ws.send_cmd(
                "resolve_symbol",
                [
                    cs_prx,
                    "sds_sym_2",
                    "=" + json.dumps({"symbol": proxy_symbol, "adjustment": "splits", "session": "extended"}),
                ],
            )
            ws.send_cmd("create_series", [cs_prx, "sds_2", "s2", "sds_sym_2", tv_interval, req_prx_bars, ""])

            for _ in range(80):
                if cme_done and prx_done:
                    break
                try:
                    raw = ws.recv_text()
                except (socket.timeout, TimeoutError):
                    break
                for payload in _tv_parse_messages(raw):
                    if payload.startswith("~h~"):
                        ws.send_text(f"~m~{len(payload)}~m~{payload}")
                        continue
                    try:
                        obj = json.loads(payload)
                    except Exception:
                        continue
                    m = obj.get("m")
                    p = obj.get("p", [])
                    if not p:
                        continue
                    sess = p[0]
                    if m == "symbol_resolved" and len(p) >= 3 and isinstance(p[2], dict):
                        mode = str(p[2].get("update_mode", "unknown"))
                        if sess == cs_cme:
                            cme_mode = mode
                        elif sess == cs_prx:
                            prx_mode = mode
                    elif m in ("timescale_update", "du") and len(p) >= 2 and isinstance(p[1], dict):
                        target = cme_bars if sess == cs_cme else (prx_bars if sess == cs_prx else None)
                        if target is not None:
                            for s_key in ("sds_1", "sds_2"):
                                s_obj = p[1].get(s_key)
                                if isinstance(s_obj, dict) and "s" in s_obj:
                                    for item in s_obj["s"]:
                                        v = item.get("v", [])
                                        if len(v) >= 5:
                                            ts_sec = int(v[0])
                                            vol = float(v[5]) if len(v) >= 6 else 0.0
                                            target[ts_sec] = (
                                                float(v[1]),
                                                float(v[2]),
                                                float(v[3]),
                                                float(v[4]),
                                                vol,
                                            )
                    elif m in ("series_completed", "symbol_error", "series_error"):
                        if sess == cs_cme:
                            cme_done = True
                        elif sess == cs_prx:
                            prx_done = True
        finally:
            ws.close()

        cme_df = self._dict_to_df(cme_bars)
        prx_df = self._dict_to_df(prx_bars)
        if target_minutes == 2:
            cme_df = resample_1m_to_target(cme_df, target_minutes=2)
            prx_df = resample_1m_to_target(prx_df, target_minutes=2)

        return cme_df, cme_mode, prx_df, prx_mode

    @staticmethod
    def _dict_to_df(bars_dict: Dict[int, Tuple[float, float, float, float, float]]) -> pd.DataFrame:
        if not bars_dict:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        rows = []
        for ts_sec in sorted(bars_dict.keys()):
            o, h, l, c, v = bars_dict[ts_sec]
            rows.append(
                {
                    "timestamp": pd.Timestamp(ts_sec, unit="s", tz="UTC"),
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                }
            )
        return pd.DataFrame(rows)

    def fetch_bars(
        self,
        local_symbol: Optional[str] = "MNQZ6",
        base_symbol: str = "MNQ",
        bar_size: str = "2 mins",
        n_bars: int = 600,
    ) -> pd.DataFrame:
        """
        Return a standardized DataFrame with columns `['timestamp', 'open', 'high', 'low', 'close', 'volume']`
        containing closed bars PLUS the currently forming bar as the final row (`iloc[-1]`), matching
        IBKR's `reqHistoricalData` format so `_bars_to_dataframe` (`df.iloc[:-1]`) works identically.
        """
        cme_sym = ibkr_symbol_to_tv_cme_symbol(local_symbol, base_symbol=base_symbol)

        cme_df, cme_mode, prx_df, _ = self._fetch_dual_series(
            cme_symbol=cme_sym,
            proxy_symbol=self.realtime_proxy_symbol,
            bar_size=bar_size,
            n_bars=n_bars,
        )

        # Fallback to continuous front-month CME_MINI:MNQ1! if specific contract month wasn't found
        if cme_df.empty and cme_sym != f"CME_MINI:{base_symbol.upper()}1!":
            cme_sym = f"CME_MINI:{base_symbol.upper()}1!"
            cme_df, cme_mode, prx_df, _ = self._fetch_dual_series(
                cme_symbol=cme_sym,
                proxy_symbol=self.realtime_proxy_symbol,
                bar_size=bar_size,
                n_bars=n_bars,
            )

        # Fallback proxy if primary proxy empty
        if prx_df.empty and self.fallback_proxy_symbol:
            _, _, prx_df, _ = self._fetch_dual_series(
                cme_symbol=cme_sym,
                proxy_symbol=self.fallback_proxy_symbol,
                bar_size=bar_size,
                n_bars=120,
            )

        self.last_cme_update_mode = cme_mode

        if cme_df.empty and prx_df.empty:
            raise RuntimeError("TradingView returned no bars for both CME futures and real-time proxy")

        if cme_df.empty:
            self.last_stitched_bars_count = len(prx_df)
            return prx_df.reset_index(drop=True)

        if prx_df.empty or cme_mode == "streaming":
            self.last_stitched_bars_count = 0
            return cme_df.reset_index(drop=True)

        # Compute rolling basis between true CME futures and real-time Nasdaq-100 proxy on overlapping bars.
        # Drop the very last CME row first in case TradingView marks the delayed forming bar as incomplete.
        cme_closed = cme_df.iloc[:-1].copy() if len(cme_df) > 2 else cme_df.copy()
        merged = pd.merge(
            cme_closed,
            prx_df,
            on="timestamp",
            how="inner",
            suffixes=("_cme", "_prx"),
        )

        if not merged.empty:
            recent_overlap = merged.tail(30)
            diffs = (recent_overlap["close_cme"] - recent_overlap["close_prx"]).to_numpy(dtype=float)
            basis = float(np.median(diffs))
            vol_ratios = (
                recent_overlap["volume_cme"] / recent_overlap["volume_prx"].clip(lower=1.0)
            ).to_numpy(dtype=float)
            vol_scale = float(np.median(vol_ratios)) if len(vol_ratios) > 0 else 1.0
        else:
            basis = float(cme_closed["close"].iloc[-1] - prx_df["close"].iloc[-1])
            vol_scale = 1.0

        self.last_basis_points = round_to_tick(basis, self.tick_size)

        last_cme_ts = cme_closed["timestamp"].iloc[-1]
        newer_prx = prx_df[prx_df["timestamp"] > last_cme_ts].copy()
        if newer_prx.empty:
            self.last_stitched_bars_count = 0
            return cme_df.reset_index(drop=True)

        # Apply basis adjustment and tick rounding to the real-time bars from (T - 10m) to T
        stitched_rows = []
        prev_close = float(cme_closed["close"].iloc[-1])
        for idx, row in newer_prx.reset_index(drop=True).iterrows():
            o = round_to_tick(float(row["open"]) + basis, self.tick_size)
            h = round_to_tick(float(row["high"]) + basis, self.tick_size)
            l = round_to_tick(float(row["low"]) + basis, self.tick_size)
            c = round_to_tick(float(row["close"]) + basis, self.tick_size)
            if idx == 0 and abs(o - prev_close) <= 3.0:
                o = prev_close
            h = max(h, o, c)
            l = min(l, o, c)
            v = max(1.0, round(float(row["volume"]) * vol_scale))
            stitched_rows.append(
                {
                    "timestamp": row["timestamp"],
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                }
            )
            prev_close = c

        stitched_df = pd.DataFrame(stitched_rows)
        self.last_stitched_bars_count = len(stitched_df)
        combined = pd.concat([cme_closed, stitched_df], ignore_index=True)
        return combined.reset_index(drop=True)
