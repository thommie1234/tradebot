"""
FTMO Email Monitor — reads FTMO emails via IMAP + parses with local LLM.

Extracts:
  - Restricted trading events (news blackout windows)
  - Account warnings (drawdown, inactivity, rule changes)
  - General FTMO announcements

Stores parsed events in a JSON file that the bot reads for news blackout.

Usage:
  python3 common/tools/ftmo_email_monitor.py           # one-shot check
  python3 common/tools/ftmo_email_monitor.py --daemon   # run every 5 min
"""
from __future__ import annotations

import email
import imaplib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from email.header import decode_header
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
ENV_FILE = SCRIPT_DIR / ".env.email"
EVENTS_FILE = REPO_ROOT / "common" / "config" / "ftmo_restricted_events.json"
PARSED_LOG = REPO_ROOT / "common" / "audit" / "parsed_emails.json"

# Ollama config
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# ── Load credentials ──────────────────────────────────────────────────

def _load_env() -> dict[str, str]:
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


# ── IMAP email fetcher ─────────────────────────────────────────────────

def fetch_unread_emails(env: dict) -> list[dict]:
    """Fetch all unread emails from the mailbox via IMAP."""
    host = env.get("FTMO_EMAIL_IMAP_HOST", "outlook.office365.com")
    port = int(env.get("FTMO_EMAIL_IMAP_PORT", "993"))
    user = env["FTMO_EMAIL_ADDRESS"]
    password = env["FTMO_EMAIL_PASSWORD"]

    emails = []
    try:
        mail = imaplib.IMAP4_SSL(host, port)
        mail.login(user, password)
        mail.select("INBOX")

        status, data = mail.search(None, "UNSEEN")
        if status != "OK" or not data[0]:
            mail.logout()
            return []

        for num in data[0].split():
            status, msg_data = mail.fetch(num, "(RFC822)")
            if status != "OK":
                continue
            raw = msg_data[0][1]
            msg = email.message_from_bytes(raw)

            # Decode subject
            subject_parts = decode_header(msg["Subject"] or "")
            subject = ""
            for part, encoding in subject_parts:
                if isinstance(part, bytes):
                    subject += part.decode(encoding or "utf-8", errors="replace")
                else:
                    subject += part

            # Decode sender
            sender = msg.get("From", "")

            # Decode date
            date_str = msg.get("Date", "")

            # Extract body
            body = _extract_body(msg)

            emails.append({
                "subject": subject,
                "sender": sender,
                "date": date_str,
                "body": body[:5000],  # Limit body size for LLM
            })

        mail.logout()
    except Exception as e:
        print(f"[EMAIL_MONITOR] IMAP error: {e}")

    return emails


def _extract_body(msg: email.message.Message) -> str:
    """Extract plain text body from email message."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
            elif content_type == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    html = payload.decode(charset, errors="replace")
                    return _html_to_text(html)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
    return ""


def _html_to_text(html: str) -> str:
    """Simple HTML to text conversion."""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── LLM parser ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an email parser for a prop trading firm (FTMO).
You extract structured data from FTMO emails. Always respond with valid JSON only, no other text.

For each email, classify it and extract relevant data:

1. "restricted_event" — trading restriction around a news event
   Extract: event_name, date (ISO 8601), currency, restricted_symbols (if mentioned), window_minutes_before, window_minutes_after

2. "account_warning" — drawdown warning, inactivity notice, rule violation
   Extract: warning_type, severity (low/medium/high/critical), details

3. "announcement" — general FTMO news, rule changes, platform updates
   Extract: topic, summary, action_required (bool)

4. "irrelevant" — marketing, newsletters, promotional content
   Just classify as irrelevant.

Respond with a JSON object:
{
  "type": "restricted_event" | "account_warning" | "announcement" | "irrelevant",
  "data": { ... extracted fields ... },
  "summary": "one-line human-readable summary"
}"""


def parse_email_with_llm(email_data: dict) -> dict | None:
    """Send email content to Ollama for structured parsing."""
    import requests

    prompt = f"""Parse this FTMO email:

Subject: {email_data['subject']}
From: {email_data['sender']}
Date: {email_data['date']}

Body:
{email_data['body']}

Respond with JSON only."""

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_predict": 500,
                },
            },
            timeout=60,
        )
        if resp.status_code == 200:
            raw = resp.json().get("response", "").strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    return json.loads(match.group())
                print(f"[EMAIL_MONITOR] LLM returned invalid JSON: {raw[:200]}")
                return None
        else:
            print(f"[EMAIL_MONITOR] Ollama HTTP {resp.status_code}")
            return None
    except Exception as e:
        print(f"[EMAIL_MONITOR] Ollama error: {e}")
        return None


# ── Event storage ──────────────────────────────────────────────────────

def _load_existing_events() -> list[dict]:
    if EVENTS_FILE.exists():
        try:
            return json.loads(EVENTS_FILE.read_text())
        except Exception:
            return []
    return []


def _save_events(events: list[dict]):
    EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVENTS_FILE.write_text(json.dumps(events, indent=2))


def _log_parsed(email_data: dict, parsed: dict):
    """Append to parsed emails log for audit."""
    log = []
    if PARSED_LOG.exists():
        try:
            log = json.loads(PARSED_LOG.read_text())
        except Exception:
            log = []
    log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "subject": email_data["subject"],
        "sender": email_data["sender"],
        "parsed": parsed,
    })
    # Keep last 200 entries
    log = log[-200:]
    PARSED_LOG.parent.mkdir(parents=True, exist_ok=True)
    PARSED_LOG.write_text(json.dumps(log, indent=2))


# ── Discord notification ───────────────────────────────────────────────

def _notify_discord(parsed: dict, email_data: dict):
    """Send important email notifications to Discord."""
    try:
        sys.path.insert(0, str(REPO_ROOT / "common"))
        from tools.discord_notifier import DiscordNotifier
        discord = DiscordNotifier()
    except Exception:
        return

    email_type = parsed.get("type", "unknown")
    summary = parsed.get("summary", email_data["subject"])

    if email_type == "restricted_event":
        data = parsed.get("data", {})
        discord.send(
            f"FTMO Restricted Event",
            f"**{data.get('event_name', 'Unknown')}**\n"
            f"Date: {data.get('date', '?')}\n"
            f"Currency: {data.get('currency', '?')}\n"
            f"Window: {data.get('window_minutes_before', '?')}min before / "
            f"{data.get('window_minutes_after', '?')}min after\n"
            f"_{summary}_",
            "red",
        )
    elif email_type == "account_warning":
        severity = parsed.get("data", {}).get("severity", "medium")
        color = "red" if severity in ("high", "critical") else "orange"
        discord.send(f"FTMO Warning [{severity.upper()}]", summary, color)
    elif email_type == "announcement":
        if parsed.get("data", {}).get("action_required"):
            discord.send("FTMO Announcement", summary, "blue")


# ── Main loop ──────────────────────────────────────────────────────────

def check_emails():
    """One-shot: fetch unread emails, parse with LLM, store events."""
    env = _load_env()
    if "FTMO_EMAIL_ADDRESS" not in env:
        print("[EMAIL_MONITOR] No email credentials configured")
        return

    emails = fetch_unread_emails(env)
    if not emails:
        print(f"[EMAIL_MONITOR] No unread emails")
        return

    print(f"[EMAIL_MONITOR] Found {len(emails)} unread email(s)")

    events = _load_existing_events()
    now = datetime.now(timezone.utc)

    for em in emails:
        print(f"  Parsing: {em['subject'][:60]}")
        parsed = parse_email_with_llm(em)
        if not parsed:
            print(f"    LLM parse failed, skipping")
            continue

        _log_parsed(em, parsed)
        print(f"    Type: {parsed.get('type')} — {parsed.get('summary', '')[:60]}")

        # Handle restricted events → add to events file for bot
        if parsed.get("type") == "restricted_event":
            data = parsed.get("data", {})
            event = {
                "event_name": data.get("event_name", "Unknown"),
                "date": data.get("date", ""),
                "currency": data.get("currency", ""),
                "restricted_symbols": data.get("restricted_symbols", []),
                "window_before_min": data.get("window_minutes_before", 2),
                "window_after_min": data.get("window_minutes_after", 2),
                "source": "ftmo_email",
                "added_at": now.isoformat(),
            }
            events.append(event)
            print(f"    Added restricted event: {event['event_name']}")

        # Notify Discord for important emails
        if parsed.get("type") != "irrelevant":
            _notify_discord(parsed, em)

    # Clean up expired events (older than 1 day)
    events = [
        e for e in events
        if not e.get("date") or _parse_date(e["date"]) > now.timestamp() - 86400
    ]
    _save_events(events)
    print(f"[EMAIL_MONITOR] Active restricted events: {len(events)}")


def _parse_date(date_str: str) -> float:
    """Try to parse an ISO date string to timestamp."""
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return 0.0


def daemon(interval_sec: int = 300):
    """Run email check every interval_sec (default 5 min)."""
    print(f"[EMAIL_MONITOR] Daemon started (interval={interval_sec}s)")
    while True:
        try:
            check_emails()
        except Exception as e:
            print(f"[EMAIL_MONITOR] Error: {e}")
        time.sleep(interval_sec)


if __name__ == "__main__":
    if "--daemon" in sys.argv:
        daemon()
    else:
        check_emails()
