"""
Kalshi paper test for namaste (BT, 2026-10-03).

Reads candidate Kalshi markets from the PUBLIC market-data API (no Kalshi account,
no Kalshi key), runs namaste's own forecasting pipeline on each one, and logs
our probability next to the market price at that moment. No orders are placed.
Rows are appended to hammer1/kalshi_paper_log.csv on BT's server through the door,
where the resolver scores them after each market settles.

Run: poetry run python kalshi_paper.py [--max 8] [--dry]
Env: OPENROUTER_API_KEY (forecasts), BT_KEY (logging). Nothing else.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import logging
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import dotenv

from bot_helpers import silence_noisy_dependencies

silence_noisy_dependencies()

from forecasting_tools import BinaryQuestion, GeneralLlm  # noqa: E402

from main import FallTemplateBot2026  # noqa: E402

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
BT_DOOR = "https://boostersplace.com/bt"
LOG_PATH = "hammer1/kalshi_paper_log.csv"

# Our lane: count and base-rate markets that resolve on public records.
# Everything else (crypto, commodities, rate decisions, sports, geopolitics) is out.
ALLOWED_PREFIXES = (
    "KXSPACEXCOUNT", "KXLAUNCHES", "KXPRESSBRIEFINGCOUNT", "KXFEDMENTION",
    "KXDATACENTCON", "KXTRUMPSTATESMON", "KXNYTHEAD", "KXBILLSSIGNED", "KXRT",
    "KXPARDONSTRUMP", "KXTRUMPSAYCOMPANY", "KXTRUMPSAYCOUNTRY", "KXTRUMPMEETM",
    "KXGEMINI", "KXOAIPERSONALAGENT", "KXGTATRAILER", "KXRAIN", "KXPRESSSECANNOUNCE",
)
MIN_VOLUME = 1000
MIN_PRICE, MAX_PRICE = 0.15, 0.85
MIN_DAYS, MAX_DAYS = 1, 30


def kget(path: str, **params):
    url = KALSHI + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"User-Agent": "namaste-paper/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def door(method_path: str, body: str = "") -> str:
    key = os.getenv("BT_KEY", "").strip().lstrip("\ufeff")
    req = urllib.request.Request(
        BT_DOOR + method_path, data=body.encode("utf-8"), method="POST",
        headers={"X-BT-Key": key, "Content-Type": "text/plain; charset=utf-8", "User-Agent": "namaste-paper/0.1"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def already_logged() -> set[str]:
    try:
        raw = json.loads(door(f"/read?path={LOG_PATH}")).get("content", "")
    except Exception as e:  # first run, or door unreachable: log nothing twice anyway
        logger.warning(f"could not read existing log: {e!r}")
        return set()
    seen = set()
    for row in csv.DictReader(io.StringIO(raw)):
        if row.get("ticker"):
            seen.add(row["ticker"])
    return seen


def candidates(max_n: int, seen: set[str]) -> list[dict]:
    now = datetime.now(timezone.utc)
    out = []
    for prefix in ALLOWED_PREFIXES:
        try:
            ms = kget("/markets", series_ticker=prefix, status="open", limit=100).get("markets", [])
        except Exception as e:
            logger.warning(f"{prefix}: {e!r}")
            continue
        for m in ms:
            if m.get("market_type") != "binary" or m["ticker"] in seen:
                continue
            try:
                vol = float(m.get("volume_fp") or 0)
                bid = float(m.get("yes_bid_dollars") or 0)
                ask = float(m.get("yes_ask_dollars") or 0)
                days = (datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")) - now).total_seconds() / 86400
            except Exception:
                continue
            mid = (bid + ask) / 2
            if vol < MIN_VOLUME or not (MIN_PRICE <= mid <= MAX_PRICE) or not (MIN_DAYS <= days <= MAX_DAYS):
                continue
            if ask - bid > 0.10:  # too wide to learn anything from the price
                continue
            out.append(dict(ticker=m["ticker"], title=m.get("title", ""), sub=m.get("yes_sub_title", ""),
                            rules=m.get("rules_primary", ""), fine=m.get("rules_secondary", ""),
                            bid=bid, ask=ask, volume=vol, close=m["close_time"], days=days))
    out.sort(key=lambda x: -x["volume"])
    return out[:max_n]


def to_question(c: dict) -> BinaryQuestion:
    text = c["title"] if not c["sub"] or c["sub"] in c["title"] else f'{c["title"]} ({c["sub"]})'
    qid = int(hashlib.sha1(c["ticker"].encode()).hexdigest()[:8], 16)
    return BinaryQuestion(
        question_text=text,
        id_of_post=qid,
        page_url=f"https://kalshi.com/markets/{c['ticker']}",
        background_info=f"Kalshi market {c['ticker']}. Closes {c['close']}.",
        resolution_criteria=c["rules"] or "Resolves YES if the stated condition is met per Kalshi's rules.",
        fine_print=c["fine"] or "",
    )


async def run(max_n: int, dry: bool) -> int:
    seen = set() if dry else already_logged()
    cands = candidates(max_n, seen)
    print(f"candidates: {len(cands)}")
    for c in cands:
        print(f"  {c['ticker']}: {c['title'][:70]} | {c['sub'][:30]} | {c['bid']:.2f}/{c['ask']:.2f} | {c['days']:.1f}d")
    if not cands:
        return 0

    bot = FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=4,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        extra_metadata_in_explanation=False,
        llms={
            "default": GeneralLlm(model="openrouter/google/gemini-3.8-flash", temperature=0.3, timeout=120, allowed_tries=6),
            "default2": GeneralLlm(model="openrouter/anthropic/claude-haiku-4.5", temperature=0.3, timeout=120, allowed_tries=6),
            "summarizer": "openrouter/google/gemini-3.5-flash-lite",
            "researcher": GeneralLlm(model="openrouter/perplexity/sonar", temperature=0.1, timeout=120, allowed_tries=6),
            "parser": "openrouter/google/gemini-3.5-flash-lite",
        },
    )
    questions = [to_question(c) for c in cands]
    reports = await bot.forecast_questions(questions, return_exceptions=True)

    rows = []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    for c, rep in zip(cands, reports):
        if isinstance(rep, BaseException):
            print(f"  FAILED {c['ticker']}: {type(rep).__name__}: {str(rep)[:160]}")
            continue
        p = float(rep.prediction)
        mid = (c["bid"] + c["ask"]) / 2
        rows.append([stamp, c["ticker"], c["title"].replace(",", ";")[:120] + (f" [{c['sub']}]" if c["sub"] else ""),
                     f"{p:.3f}", f"{c['bid']:.2f}", f"{c['ask']:.2f}", "namaste", f"edge={p - mid:+.3f} close={c['close'][:10]}"])
        print(f"  {c['ticker']}: ours {p:.2f} vs market {mid:.2f} (edge {p - mid:+.2f})")
    if rows and not dry:
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows(rows)
        print(door(f"/append?path={LOG_PATH}", buf.getvalue()))
    return len(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=8)
    ap.add_argument("--dry", action="store_true", help="forecast but do not log")
    a = ap.parse_args()
    if not os.getenv("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY missing"); sys.exit(1)
    n = asyncio.run(run(a.max, a.dry))
    print(f"logged {n} forecast(s)")
