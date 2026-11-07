"""
Live Trade Journal for Live-vs-Backtest Reconciliation.

Appends one JSON object per line to `journal/events/YYYY-MM-DD.jsonl` (ET trading date) for:
- every closed bar exactly as the live bot saw it (incl. TradingView basis / stitched-bar metadata)
- every signal (placed or blocked by risk rules), order, fill, stop move, cancel and exit

Design rules:
- Never raises: journaling must never interrupt order management.
- No strategy logic lives here; it only records what happened.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger("nq_ibkr_bot")

DEFAULT_JOURNAL_DIR = os.getenv("JOURNAL_DIR", "journal")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, float):
        return round(value, 4)
    if hasattr(value, "item"):  # numpy scalars
        try:
            return _to_jsonable(value.item())
        except Exception:
            return str(value)
    return value


class TradeJournal:
    """Thread-safe, append-only JSONL event journal partitioned by ET trading date."""

    def __init__(self, base_dir: str = DEFAULT_JOURNAL_DIR, timezone: str = "America/New_York", enabled: bool = True) -> None:
        self.base_dir = Path(base_dir)
        self.timezone = timezone
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            try:
                (self.base_dir / "events").mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning("Trade journal disabled (cannot create %s): %s", self.base_dir, exc)
                self.enabled = False

    def _path_for(self, ts_utc: pd.Timestamp) -> Path:
        day = ts_utc.tz_convert(self.timezone).strftime("%Y-%m-%d")
        return self.base_dir / "events" / f"{day}.jsonl"

    def log(self, event: str, bar_ts: Optional[pd.Timestamp] = None, **fields: Any) -> None:
        """Append an event. `bar_ts` (if given) decides the trading-date file and is stored as ISO UTC."""
        if not self.enabled:
            return
        try:
            now_utc = pd.Timestamp.now(tz="UTC")
            record: dict[str, Any] = {"event": event, "logged_at": now_utc.isoformat()}
            ref_ts = now_utc
            if bar_ts is not None:
                b = pd.Timestamp(bar_ts)
                b = b.tz_localize("UTC") if b.tzinfo is None else b.tz_convert("UTC")
                record["bar_ts"] = b.isoformat()
                ref_ts = b
            for k, v in fields.items():
                record[k] = _to_jsonable(v)
            line = json.dumps(record, default=str)
            with self._lock:
                with open(self._path_for(ref_ts), "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as exc:  # journaling must never break trading
            logger.warning("Trade journal write failed (%s): %s", event, exc)


def load_journal(day: str, base_dir: str = DEFAULT_JOURNAL_DIR) -> list[dict]:
    """Load all events for an ET trading date `YYYY-MM-DD`."""
    path = Path(base_dir) / "events" / f"{day}.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out
