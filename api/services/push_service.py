"""Expo Push Notifications sender."""
from __future__ import annotations

import logging

import httpx

from api.services.db_service import get_active_push_tokens

logger = logging.getLogger("api.push")

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


async def send_push(title: str, body: str, data: dict | None = None):
    tokens = get_active_push_tokens()
    if not tokens:
        return

    messages = [
        {
            "to": token,
            "sound": "default",
            "title": title,
            "body": body,
            "data": data or {},
        }
        for token in tokens
    ]

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(EXPO_PUSH_URL, json=messages)
            if resp.status_code != 200:
                logger.warning(f"Push send failed: {resp.status_code} {resp.text}")
            else:
                logger.info(f"Push sent to {len(tokens)} devices: {title}")
    except Exception as e:
        logger.warning(f"Push send error: {e}")
