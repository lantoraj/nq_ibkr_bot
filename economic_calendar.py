"""
economic_calendar.py — ForexFactory Economic Calendar Feed & News Blackout Engine

Fetches, caches, and classifies weekly macro economic events from ForexFactory
(`https://nfs.faireconomy.media/ff_calendar_thisweek.json`) so the live trading
bot can:
  1. Block new trade entries before and immediately after Tier-1 / Tier-2 USD
     High-Impact releases where 2-minute candle ranges routinely exceed the
     40 pt max stop-loss cap.
  2. Tighten stop-losses on open positions shortly ahead of an imminent release.
  3. Export `ff_calendar.json` for the live bilingual HTML strategy dashboard.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz

logger = logging.getLogger("nq_bot.calendar")

FF_THIS_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FF_NEXT_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_nextweek.json"

# Tier-1 US macro releases that historically produce extreme 2-minute spikes
# (80–280 pts on NQ/MNQ), blowing through normal 20–40 pt stop-losses.
TIER1_KEYWORDS: Tuple[str, ...] = (
    "fomc",
    "federal funds rate",
    "powell",
    "non-farm",
    "nonfarm",
    "cpi",
    "ppi",
    "pce",
    "gdp",
    "retail sales",
    "ism manufacturing",
    "ism services",
    "jolts",
    "adp non-farm",
    "unemployment rate",
    "employment change",
)


@dataclass(frozen=True)
class EconomicEvent:
    """Normalized representation of a single ForexFactory calendar event."""

    title: str
    country: str
    date: str
    dt_utc: datetime
    dt_et: datetime
    impact: str
    forecast: str = ""
    previous: str = ""
    tier: int = 0

    def to_dict(self, cet_tz: Optional[pytz.BaseTzInfo] = None) -> Dict[str, Any]:
        """Serialize event with ET and CET localized fields for web export."""
        cet = cet_tz or pytz.timezone("Europe/Prague")
        dt_cet = self.dt_utc.astimezone(cet)
        return {
            "title": self.title,
            "country": self.country,
            "date": self.date,
            "impact": self.impact,
            "forecast": self.forecast,
            "previous": self.previous,
            "tier": self.tier,
            "date_et": self.dt_et.strftime("%Y-%m-%d"),
            "time_et": self.dt_et.strftime("%H:%M"),
            "date_cet": dt_cet.strftime("%Y-%m-%d"),
            "time_cet": dt_cet.strftime("%H:%M"),
        }


class EconomicCalendar:
    """Fetches, caches, and evaluates ForexFactory macro blackout windows."""

    def __init__(
        self,
        cache_path: str = "/app/journal/ff_calendar_cache.json",
        report_path: Optional[str] = "/app/report/ff_calendar.json",
        timezone_name: str = "America/New_York",
        tier1_pre_mins: int = 15,
        tier1_post_mins: int = 30,
        tier2_pre_mins: int = 10,
        tier2_post_mins: int = 15,
        refresh_interval_hours: float = 4.0,
        blackout_countries: Tuple[str, ...] = ("USD",),
    ) -> None:
        self.cache_path = Path(cache_path)
        self.report_path = Path(report_path) if report_path else None
        self.tz = pytz.timezone(timezone_name)
        self.cet_tz = pytz.timezone("Europe/Prague")
        self.tier1_pre_mins = int(tier1_pre_mins)
        self.tier1_post_mins = int(tier1_post_mins)
        self.tier2_pre_mins = int(tier2_pre_mins)
        self.tier2_post_mins = int(tier2_post_mins)
        self.refresh_interval = timedelta(hours=float(refresh_interval_hours))
        self.blackout_countries = tuple(c.upper() for c in blackout_countries)

        self.events: List[EconomicEvent] = []
        self.raw_events: List[Dict[str, Any]] = []
        self._last_fetch_utc: Optional[datetime] = None

    @staticmethod
    def _parse_iso_datetime(date_str: str) -> datetime:
        """Parse ISO-8601 timestamp string into a UTC-aware datetime."""
        cleaned = str(date_str).strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(pytz.UTC)

    def _to_utc_datetime(self, ts: Any) -> datetime:
        """Convert datetime or pandas Timestamp to a UTC-aware datetime."""
        if isinstance(ts, pd.Timestamp):
            dt = ts.to_pydatetime()
        elif isinstance(ts, datetime):
            dt = ts
        else:
            dt = pd.Timestamp(ts).to_pydatetime()

        if dt.tzinfo is None:
            dt = self.tz.localize(dt)
        return dt.astimezone(pytz.UTC)

    def classify_tier(self, title: str, country: str, impact: str) -> int:
        """
        Classify event blackout tier:
          1 = Tier-1 Mega Shock (selected USD High-Impact macro releases)
          2 = Tier-2 High-Impact event in a blackout country
          0 = Informational only (no entry blackout)
        """
        if country.upper() not in self.blackout_countries:
            return 0
        if impact.strip().lower() != "high":
            return 0
        lowered = title.lower()
        for kw in TIER1_KEYWORDS:
            if kw in lowered:
                return 1
        return 2

    def load_events_from_list(self, raw_events: List[Dict[str, Any]]) -> int:
        """Parse and index raw ForexFactory event dictionaries in memory."""
        dedup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        parsed: List[EconomicEvent] = []

        for item in raw_events or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            country = str(item.get("country", "")).strip().upper()
            date_str = str(item.get("date", "")).strip()
            impact = str(item.get("impact", "")).strip()
            if not title or not date_str:
                continue
            key = (date_str, country, title)
            if key in dedup:
                continue
            try:
                dt_utc = self._parse_iso_datetime(date_str)
            except Exception as exc:
                logger.debug("Skipping unparseable FF event %r (%s): %s", title, date_str, exc)
                continue

            dedup[key] = dict(item)
            dt_et = dt_utc.astimezone(self.tz)
            tier = self.classify_tier(title, country, impact)
            parsed.append(
                EconomicEvent(
                    title=title,
                    country=country,
                    date=date_str,
                    dt_utc=dt_utc,
                    dt_et=dt_et,
                    impact=impact,
                    forecast=str(item.get("forecast", "") or "").strip(),
                    previous=str(item.get("previous", "") or "").strip(),
                    tier=tier,
                )
            )

        parsed.sort(key=lambda e: e.dt_utc)
        self.events = parsed
        self.raw_events = list(dedup.values())
        return len(self.events)

    def _download_feed(self, url: str, timeout: float = 10.0) -> Optional[List[Dict[str, Any]]]:
        """Download JSON array from a ForexFactory mirror URL; returns None on error."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NQIBKRBot/4.2)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if getattr(resp, "status", 200) != 200:
                    return None
                payload = resp.read().decode("utf-8", errors="replace")
            data = json.loads(payload)
            return data if isinstance(data, list) else None
        except urllib.error.HTTPError as exc:
            # nextweek endpoint commonly returns 404 on weekends; treat softly
            logger.debug("FF feed %s HTTP %s", url, exc.code)
            return None
        except Exception as exc:
            logger.warning("Failed fetching ForexFactory feed %s: %s", url, exc)
            return None

    def load_from_cache(self) -> bool:
        """Load cached ForexFactory events from disk if present."""
        if not self.cache_path.exists():
            return False
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("events"), list):
                raw_list = data["events"]
                fetched_str = data.get("fetched_at_utc")
                if fetched_str:
                    self._last_fetch_utc = self._parse_iso_datetime(fetched_str)
            elif isinstance(data, list):
                raw_list = data
            else:
                return False

            count = self.load_events_from_list(raw_list)
            logger.info("Loaded %d calendar events from cache %s", count, self.cache_path)
            return count > 0
        except Exception as exc:
            logger.warning("Failed reading calendar cache %s: %s", self.cache_path, exc)
            return False

    def _is_cache_fresh(self, now_utc: datetime) -> bool:
        if self._last_fetch_utc is not None:
            return (now_utc - self._last_fetch_utc) < self.refresh_interval
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                fetched_str = data.get("fetched_at_utc") if isinstance(data, dict) else None
                if fetched_str:
                    fetched_dt = self._parse_iso_datetime(fetched_str)
                    return (now_utc - fetched_dt) < self.refresh_interval
            except Exception:
                return False
        return False

    def fetch_and_cache(self, force: bool = False, include_next_week: bool = True) -> bool:
        """
        Fetch latest weekly ForexFactory events, update disk cache and optional
        dashboard report JSON. Falls back to existing disk cache if offline.
        """
        now_utc = datetime.now(timezone.utc)
        if not force and self._is_cache_fresh(now_utc):
            if not self.events:
                self.load_from_cache()
            if self.report_path and self.events and not self.report_path.exists():
                self.export_report_json(str(self.report_path))
            return len(self.events) > 0

        combined: List[Dict[str, Any]] = []
        this_week = self._download_feed(FF_THIS_WEEK_URL)
        if this_week:
            combined.extend(this_week)

        if include_next_week:
            next_week = self._download_feed(FF_NEXT_WEEK_URL)
            if next_week:
                combined.extend(next_week)

        if not combined:
            logger.warning("ForexFactory live fetch empty; falling back to local cache")
            loaded = self.load_from_cache()
            if loaded and self.report_path and not self.report_path.exists():
                self.export_report_json(str(self.report_path))
            return loaded

        count = self.load_events_from_list(combined)
        self._last_fetch_utc = now_utc

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_payload = {
                "fetched_at_utc": now_utc.isoformat(),
                "source": FF_THIS_WEEK_URL,
                "count": count,
                "events": self.raw_events,
            }
            self.cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not write calendar cache %s: %s", self.cache_path, exc)

        if self.report_path:
            self.export_report_json(str(self.report_path))

        logger.info(
            "ForexFactory calendar synced: %d total events (%d Tier-1, %d Tier-2 USD High)",
            count,
            sum(1 for e in self.events if e.tier == 1),
            sum(1 for e in self.events if e.tier == 2),
        )
        return count > 0

    def refresh_if_needed(self) -> bool:
        """Refresh calendar from ForexFactory if refresh_interval has elapsed."""
        now_utc = datetime.now(timezone.utc)
        if self._last_fetch_utc is None or (now_utc - self._last_fetch_utc) >= self.refresh_interval:
            return self.fetch_and_cache(force=True)
        return False

    def get_blackout_window(self, ev: EconomicEvent) -> Optional[Tuple[datetime, datetime, int, int]]:
        """Return (start_utc, end_utc, pre_mins, post_mins) for Tier-1/Tier-2 events."""
        if ev.tier == 1:
            pre, post = self.tier1_pre_mins, self.tier1_post_mins
        elif ev.tier == 2:
            pre, post = self.tier2_pre_mins, self.tier2_post_mins
        else:
            return None
        return (
            ev.dt_utc - timedelta(minutes=pre),
            ev.dt_utc + timedelta(minutes=post),
            pre,
            post,
        )

    def is_in_news_blackout(self, ts: Any) -> Tuple[bool, str]:
        """Return (True, reason) if `ts` falls inside an active Tier-1 or Tier-2 blackout."""
        if not self.events:
            return False, ""

        ts_utc = self._to_utc_datetime(ts)
        for ev in self.events:
            if ev.tier == 0:
                continue
            window = self.get_blackout_window(ev)
            if window is None:
                continue
            w_start, w_end, pre, post = window
            if w_start <= ts_utc <= w_end:
                reason = (
                    f"Tier-{ev.tier} USD High-Impact news blackout: "
                    f"{ev.title} ({ev.dt_et.strftime('%H:%M')} ET, -{pre}m/+{post}m)"
                )
                return True, reason
        return False, ""

    def upcoming_event_within(self, ts: Any, minutes: int = 4) -> Optional[EconomicEvent]:
        """Return the soonest Tier-1/Tier-2 event occurring within [0, minutes] ahead of `ts`."""
        if not self.events or minutes <= 0:
            return None

        ts_utc = self._to_utc_datetime(ts)
        max_sec = minutes * 60.0
        for ev in self.events:
            if ev.tier == 0:
                continue
            delta_sec = (ev.dt_utc - ts_utc).total_seconds()
            if 0.0 <= delta_sec <= max_sec:
                return ev
        return None

    def get_today_events(
        self,
        ts: Optional[Any] = None,
        countries: Tuple[str, ...] = ("USD",),
        impacts: Tuple[str, ...] = ("High",),
    ) -> List[EconomicEvent]:
        """Return all calendar events on the ET session date of `ts` matching filters."""
        if not self.events:
            return []
        ref_utc = self._to_utc_datetime(ts) if ts is not None else datetime.now(timezone.utc)
        target_date = ref_utc.astimezone(self.tz).date()
        allowed_c = {c.upper() for c in countries}
        allowed_i = {i.lower() for i in impacts}

        return [
            ev
            for ev in self.events
            if ev.dt_et.date() == target_date
            and (not allowed_c or ev.country.upper() in allowed_c)
            and (not allowed_i or ev.impact.lower() in allowed_i)
        ]

    @staticmethod
    def format_events_summary(events: List[EconomicEvent]) -> str:
        """Build compact single-line summary string for logs and Discord notifications."""
        if not events:
            return "None"
        parts = []
        for ev in events:
            tier_tag = f"T{ev.tier}" if ev.tier in (1, 2) else ev.impact
            parts.append(f"{ev.dt_et.strftime('%H:%M')} ET {ev.title} [{tier_tag}]")
        return " | ".join(parts)

    def export_report_json(self, report_path: str) -> None:
        """Write normalized calendar feed JSON for the nginx strategy dashboard."""
        out = Path(report_path)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": FF_THIS_WEEK_URL,
                "rules": {
                    "tier1_pre_mins": self.tier1_pre_mins,
                    "tier1_post_mins": self.tier1_post_mins,
                    "tier2_pre_mins": self.tier2_pre_mins,
                    "tier2_post_mins": self.tier2_post_mins,
                },
                "events": [ev.to_dict(self.cet_tz) for ev in self.events],
            }
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not export report calendar JSON %s: %s", out, exc)
