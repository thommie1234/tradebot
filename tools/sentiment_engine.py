#!/usr/bin/env python3
"""
SOVEREIGN SENTIMENT ENGINE — News scoring pipeline
====================================================

Scrapes crypto/forex news headlines from RSS feeds and CryptoPanic API,
scores them via Ollama (Llama 3.1 8B), and stores sentiment in SQLite
for the trading bot to query.

Runs as a standalone systemd service (5-min loop).

Usage:
    python3 sentiment_engine.py           # Run continuous loop
    python3 sentiment_engine.py --once    # Single scrape+score, then exit

Author: Thomas (HP Z440)
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

try:
    import feedparser
except ImportError:
    print("[FATAL] feedparser not installed. Run: pip install feedparser")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = str(SCRIPT_DIR / "sovereign_log.db")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

SCRAPE_INTERVAL = 300       # 5 minutes between scrape cycles
BATCH_SIZE = 15             # Headlines per LLM call (5-20 sweet spot)
SEEN_CACHE_MAX = 5000       # Max URLs to keep in dedup set

# Discord config (optional — loaded from discord_config.json)
DISCORD_WEBHOOK = None
try:
    _dcfg_path = SCRIPT_DIR / "discord_config.json"
    if _dcfg_path.exists():
        with open(_dcfg_path) as _f:
            _dcfg = json.load(_f)
        if _dcfg.get("enabled"):
            DISCORD_WEBHOOK = _dcfg.get("webhook_url")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Symbol keyword mapping
# ---------------------------------------------------------------------------

SYMBOL_KEYWORDS: dict[str, list[str]] = {
    # Crypto
    "BTCUSD": ["bitcoin", "btc", "halving", "saylor", "grayscale"],
    "ETHUSD": ["ethereum", "eth"],
    "SOLUSD": ["solana", "sol"],
    "XMRUSD": ["monero", "xmr"],
    "AAVUSD": ["aave"],
    "LNKUSD": ["chainlink", "link"],
    "DOGUSD": ["dogecoin", "doge"],
    "ADAUSD": ["cardano", "ada"],
    "DOTUSD": ["polkadot", "dot"],
    "XLMUSD": ["stellar", "xlm"],
    "AVXUSD": ["avalanche", "avax"],
    "MATUSD": ["polygon", "matic"],
    "UNIUSD": ["uniswap", "uni"],
    "XRPUSD": ["ripple", "xrp"],
    "BNBUSD": ["binance coin", "bnb"],
    "LTCUSD": ["litecoin", "ltc"],
    # Indices
    "JP225.cash": ["nikkei", "japan stock", "boj", "ueda", "tokyo stock", "yen", "jpy", "bank of japan"],
    "US500.cash": ["s&p 500", "s&p", "sp500"],
    "US100.cash": ["nasdaq", "tech stocks"],
    "US30.cash": ["dow jones", "dow"],
    "DE40.cash": ["dax", "german stock"],
    # Commodities
    "XAUUSD": ["gold", "xau"],
    "XAGUSD": ["silver", "xag"],
    "XPD_USD": ["palladium", "norilsk"],
    "XPT_USD": ["platinum"],
    "WHEAT.c": ["wheat", "grain", "agriculture"],
    "USOIL.cash": ["oil", "crude", "wti", "brent"],
    "NGAS.cash": ["natural gas"],
    # Forex — wildcard patterns (match any pair containing this currency)
    "EUR_*": ["ecb", "eurozone", "euro area"],
    "USD_*": ["fed", "fomc", "dollar", "treasury", "us economy", "federal reserve",
              "powell", "nfp", "dxy"],
    "GBP_*": ["boe", "sterling", "pound", "bank of england"],
    "JPY_*": ["boj", "yen", "bank of japan"],
    "AUD_*": ["rba", "aussie", "australian dollar"],
    "NZD_*": ["rbnz", "kiwi", "new zealand dollar"],
    "CAD_*": ["boc", "loonie", "bank of canada"],
    "CHF_*": ["snb", "swiss franc", "swiss national bank"],
}

BROAD_KEYWORDS: dict[str, list[str]] = {
    "crypto": ["crypto", "cryptocurrency", "defi", "sec crypto", "crypto regulation",
               "stablecoin", "cbdc", "digital asset"],
    "forex": ["interest rate", "inflation", "central bank", "gdp", "employment",
              "nonfarm", "cpi", "ppi", "monetary policy"],
    "risk_off": ["recession", "crash", "crisis", "war", "sanctions", "default",
                 "contagion", "bank run", "systemic risk"],
}

# ---------------------------------------------------------------------------
# RSS / API Feed sources
# ---------------------------------------------------------------------------

RSS_FEEDS: list[dict] = [
    {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "category": "crypto"},
    {"name": "CoinTelegraph", "url": "https://cointelegraph.com/rss", "category": "crypto"},
    {"name": "TheBlock", "url": "https://www.theblock.co/rss.xml", "category": "crypto"},
    {"name": "Investing.com", "url": "https://www.investing.com/rss/news.rss", "category": "broad"},
    {"name": "FXStreet", "url": "https://www.fxstreet.com/rss/news", "category": "forex"},
    {"name": "CNBC Markets", "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069", "category": "broad"},
]

CRYPTOPANIC_URL = "https://cryptopanic.com/api/free/v1/posts/?public=true&kind=news"

# ---------------------------------------------------------------------------
# LLM Scoring Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial sentiment classifier. Score each headline from -1.0 (extreme fear/bearish) to +1.0 (extreme greed/bullish). Return ONLY valid JSON array.

Rules:
- Regulatory crackdown, hack, insolvency = -0.7 to -1.0
- Rate hike, inflation up, recession signal = -0.5 to -0.8
- ETF approval, institutional adoption = +0.7 to +1.0
- Rate cut, stimulus, easing = +0.5 to +0.8
- Neutral/irrelevant news = 0.0
- Minor positive/negative = ±0.1 to ±0.3

Return ONLY a JSON array like: [{"id":1,"score":0.5,"reason":"brief reason"},...]
No markdown, no explanation, just the JSON array."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("SentimentEngine")

# ---------------------------------------------------------------------------
# SentimentEngine
# ---------------------------------------------------------------------------


class SentimentEngine:
    """Fetch headlines, score via Ollama, store in SQLite."""

    def __init__(self, db_path: str = DB_PATH, ollama_host: str = OLLAMA_HOST):
        self.db_path = db_path
        self.ollama_host = ollama_host
        self._seen_urls: set[str] = set()
        self._init_db()
        self._load_seen_urls()

    # ---- DB setup ----

    def _init_db(self):
        """Create sentiment_scores table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    headline TEXT NOT NULL,
                    source TEXT NOT NULL,
                    url TEXT UNIQUE,
                    score REAL NOT NULL,
                    reason TEXT,
                    symbols TEXT,
                    category TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_ts
                ON sentiment_scores(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_symbol
                ON sentiment_scores(symbols)
            """)
            conn.commit()
        finally:
            conn.close()

    def _load_seen_urls(self):
        """Load recent URLs from DB to avoid re-scoring on restart."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT url FROM sentiment_scores WHERE timestamp > ? AND url IS NOT NULL",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()
        self._seen_urls = {r[0] for r in rows}
        log.info("Loaded %d seen URLs from DB", len(self._seen_urls))

    # ---- News fetching ----

    def _fetch_rss(self, feed: dict) -> list[dict]:
        """Fetch headlines from a single RSS feed."""
        headlines = []
        try:
            parsed = feedparser.parse(feed["url"])
            for entry in parsed.entries[:30]:
                url = getattr(entry, "link", "") or ""
                title = getattr(entry, "title", "") or ""
                if not title or not url:
                    continue
                headlines.append({
                    "title": title.strip(),
                    "url": url.strip(),
                    "source": feed["name"],
                    "category": feed["category"],
                })
        except Exception as e:
            log.warning("RSS fetch failed for %s: %s", feed["name"], e)
        return headlines

    def _fetch_cryptopanic(self) -> list[dict]:
        """Fetch headlines from CryptoPanic free API."""
        headlines = []
        try:
            resp = requests.get(CRYPTOPANIC_URL, timeout=15, headers={
                "User-Agent": "SovereignSentimentEngine/1.0",
            })
            if resp.status_code != 200:
                log.warning("CryptoPanic returned %d", resp.status_code)
                return headlines
            data = resp.json()
            for post in data.get("results", [])[:30]:
                title = post.get("title", "")
                url = post.get("url", "")
                if not title or not url:
                    continue
                # Extract currency symbols from CryptoPanic metadata
                currencies = [c.get("code", "").upper()
                              for c in post.get("currencies", []) if c.get("code")]
                headlines.append({
                    "title": title.strip(),
                    "url": url.strip(),
                    "source": "CryptoPanic",
                    "category": "crypto",
                    "currencies": currencies,
                })
        except Exception as e:
            log.warning("CryptoPanic fetch failed: %s", e)
        return headlines

    def _fetch_all_sources(self) -> list[dict]:
        """Fetch from all configured sources, return deduplicated list."""
        all_headlines: list[dict] = []

        # RSS feeds
        for feed in RSS_FEEDS:
            all_headlines.extend(self._fetch_rss(feed))

        # CryptoPanic
        all_headlines.extend(self._fetch_cryptopanic())

        log.info("Fetched %d total headlines from all sources", len(all_headlines))
        return all_headlines

    # ---- Symbol matching ----

    def _match_symbols(self, headline: dict) -> tuple[list[str], str]:
        """Match headline text to trading symbols. Returns (symbols, category)."""
        text = headline["title"].lower()
        matched: list[str] = []
        category = headline.get("category", "broad")

        # Direct symbol matching
        for symbol, keywords in SYMBOL_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    if symbol.endswith("_*"):
                        # Wildcard forex — expand to known pairs
                        base = symbol.replace("_*", "")
                        matched.append(f"{base}_BROAD")
                    else:
                        matched.append(symbol)
                    break

        # CryptoPanic currency codes → symbols
        for code in headline.get("currencies", []):
            sym = f"{code}USD"
            if sym not in matched and any(sym == s for s in SYMBOL_KEYWORDS):
                matched.append(sym)

        # Broad keyword matching
        for broad_cat, keywords in BROAD_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    matched.append(f"_BROAD_{broad_cat.upper()}")
                    if broad_cat == "crypto":
                        category = "crypto"
                    elif broad_cat == "forex":
                        category = "forex"
                    break

        # Deduplicate
        matched = list(dict.fromkeys(matched))
        return matched, category

    # ---- LLM scoring ----

    def _batch_score(self, headlines: list[dict]) -> list[dict]:
        """Score a batch of headlines via Ollama. Returns headlines with 'score' and 'reason' added."""
        scored = []

        # Process in chunks of BATCH_SIZE
        for i in range(0, len(headlines), BATCH_SIZE):
            batch = headlines[i:i + BATCH_SIZE]
            prompt_lines = ["Score these headlines:"]
            for idx, h in enumerate(batch, 1):
                prompt_lines.append(f'{idx}. "{h["title"]}"')
            prompt_lines.append("")
            prompt_lines.append('Return JSON: [{"id":1,"score":0.5,"reason":"brief reason"},...]')
            prompt = "\n".join(prompt_lines)

            try:
                resp = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "system": SYSTEM_PROMPT,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 1500,
                        },
                    },
                    timeout=120,
                )
                if resp.status_code != 200:
                    log.warning("Ollama returned %d", resp.status_code)
                    continue

                raw = resp.json().get("response", "")
                scores = self._parse_scores(raw, len(batch))

                for idx, h in enumerate(batch):
                    s = scores.get(idx + 1, {"score": 0.0, "reason": "parse_fail"})
                    symbols, category = self._match_symbols(h)
                    scored.append({
                        **h,
                        "score": max(-1.0, min(1.0, s["score"])),
                        "reason": s.get("reason", ""),
                        "symbols": ",".join(symbols) if symbols else "",
                        "category": category,
                    })

            except requests.exceptions.ConnectionError:
                log.error("Cannot connect to Ollama at %s — is it running?", self.ollama_host)
                break
            except Exception as e:
                log.error("Ollama scoring failed: %s", e)
                continue

        return scored

    def _parse_scores(self, raw: str, expected: int) -> dict[int, dict]:
        """Parse LLM JSON response into {id: {score, reason}} dict."""
        scores: dict[int, dict] = {}

        # Try to extract JSON array from response
        # LLM might wrap it in markdown code blocks
        raw = raw.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict) and "id" in item and "score" in item:
                        scores[int(item["id"])] = {
                            "score": float(item["score"]),
                            "reason": str(item.get("reason", "")),
                        }
        except json.JSONDecodeError:
            # Fallback: try to find individual JSON objects
            for m in re.finditer(r'\{[^{}]*"id"\s*:\s*(\d+)[^{}]*"score"\s*:\s*(-?[\d.]+)[^{}]*\}', raw):
                try:
                    idx = int(m.group(1))
                    score = float(m.group(2))
                    # Try to extract reason
                    reason_m = re.search(r'"reason"\s*:\s*"([^"]*)"', m.group(0))
                    reason = reason_m.group(1) if reason_m else ""
                    scores[idx] = {"score": score, "reason": reason}
                except (ValueError, IndexError):
                    continue

        if len(scores) < expected:
            log.warning("Parsed %d/%d scores from LLM response", len(scores), expected)

        return scores

    # ---- Storage ----

    def _store(self, scored_headlines: list[dict]):
        """Store scored headlines in SQLite."""
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        conn = sqlite3.connect(self.db_path)
        try:
            for h in scored_headlines:
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO sentiment_scores
                           (timestamp, headline, source, url, score, reason, symbols, category)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (now, h["title"], h["source"], h["url"],
                         h["score"], h.get("reason", ""), h.get("symbols", ""),
                         h.get("category", "broad")),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass  # Duplicate URL
            conn.commit()
        finally:
            conn.close()
        log.info("Stored %d/%d scored headlines", inserted, len(scored_headlines))

    # ---- Discord alerts ----

    def _alert_extreme(self, scored_headlines: list[dict]):
        """Send Discord alert for extreme sentiment (|score| > 0.7)."""
        if not DISCORD_WEBHOOK:
            return

        extreme = [h for h in scored_headlines if abs(h.get("score", 0)) >= 0.7]
        if not extreme:
            return

        lines = []
        for h in extreme[:5]:  # Max 5 alerts per batch
            emoji = "🔴" if h["score"] < 0 else "🟢"
            lines.append(f'{emoji} **{h["score"]:+.1f}** — {h["title"][:100]}')
            if h.get("symbols"):
                lines.append(f'   Symbols: `{h["symbols"]}`')

        try:
            requests.post(DISCORD_WEBHOOK, json={
                "embeds": [{
                    "title": "Sentiment Alert",
                    "description": "\n".join(lines),
                    "color": 0xFF0000 if any(h["score"] < -0.7 for h in extreme) else 0x00FF00,
                }],
            }, timeout=10)
        except Exception as e:
            log.warning("Discord alert failed: %s", e)

    # ---- Query API (called by sovereign_bot.py) ----

    def _prune_seen_cache(self):
        """Keep seen URLs cache bounded."""
        if len(self._seen_urls) > SEEN_CACHE_MAX:
            # Just clear it — URLs also checked against DB via UNIQUE constraint
            self._seen_urls.clear()
            self._load_seen_urls()

    # ---- Main loop ----

    def run(self, interval: int = SCRAPE_INTERVAL):
        """Main loop: scrape → score → store. Every 5 min."""
        log.info("Starting sentiment engine (interval=%ds, model=%s)", interval, OLLAMA_MODEL)

        while True:
            try:
                headlines = self._fetch_all_sources()
                new = [h for h in headlines if h["url"] not in self._seen_urls]
                log.info("New headlines to score: %d (of %d fetched)", len(new), len(headlines))

                if new:
                    scored = self._batch_score(new)
                    if scored:
                        self._store(scored)
                        self._alert_extreme(scored)
                    for h in new:
                        self._seen_urls.add(h["url"])
                    self._prune_seen_cache()

            except Exception as e:
                log.error("Scrape cycle failed: %s", e, exc_info=True)

            time.sleep(interval)

    def run_once(self):
        """Single scrape+score cycle (for testing)."""
        log.info("Running single scrape cycle (model=%s)", OLLAMA_MODEL)
        headlines = self._fetch_all_sources()
        new = [h for h in headlines if h["url"] not in self._seen_urls]
        log.info("New headlines to score: %d (of %d fetched)", len(new), len(headlines))

        if new:
            scored = self._batch_score(new)
            if scored:
                self._store(scored)
                self._alert_extreme(scored)
                for h in scored:
                    log.info("  [%+.1f] %s — %s", h["score"], h["title"][:80], h.get("reason", ""))
            for h in new:
                self._seen_urls.add(h["url"])
            return scored
        else:
            log.info("No new headlines found")
            return []


# ---------------------------------------------------------------------------
# Sentiment Query API (imported by sovereign_bot.py)
# ---------------------------------------------------------------------------

def get_sentiment(symbol: str, hours: int = 4, db_path: str = DB_PATH) -> float:
    """Get weighted average sentiment for a symbol over last N hours.

    Returns 0.0 if no data (neutral = no effect on trading).
    Recent headlines weighted more (exponential decay, half-life = 1 hour).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    try:
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                """SELECT score, timestamp FROM sentiment_scores
                   WHERE timestamp > ? AND (
                       symbols LIKE ? OR symbols LIKE ? OR symbols LIKE ? OR symbols = ?
                   )
                   ORDER BY timestamp DESC""",
                (cutoff,
                 f"%{symbol},%",     # symbol at start/middle
                 f"%,{symbol}%",     # symbol at end
                 f"%{symbol}%",      # symbol anywhere (broad match)
                 symbol),            # exact match
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return 0.0

    if not rows:
        return 0.0

    now = datetime.now(timezone.utc)
    half_life_secs = 3600.0  # 1 hour half-life
    weighted_sum = 0.0
    weight_total = 0.0

    for score, ts_str in rows:
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_secs = (now - ts).total_seconds()
            weight = math.exp(-0.693 * age_secs / half_life_secs)  # ln(2) ≈ 0.693
            weighted_sum += score * weight
            weight_total += weight
        except (ValueError, TypeError):
            continue

    if weight_total == 0:
        return 0.0

    return round(weighted_sum / weight_total, 3)


def get_sentiment_bulk(symbols: list[str], hours: int = 4, db_path: str = DB_PATH) -> dict[str, float]:
    """Get sentiment for multiple symbols in one call. Returns {symbol: score}."""
    return {sym: get_sentiment(sym, hours, db_path) for sym in symbols}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = SentimentEngine()

    if "--once" in sys.argv:
        engine.run_once()
    else:
        engine.run()
