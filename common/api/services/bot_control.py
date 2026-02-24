"""Bot control — systemctl commands for sovereign-bot."""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("api.control")


async def _run_cmd(cmd: list[str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    output = (stdout or b"").decode().strip()
    if proc.returncode != 0:
        output += "\n" + (stderr or b"").decode().strip()
    return proc.returncode, output


async def is_bot_running() -> bool:
    code, _ = await _run_cmd(["systemctl", "--user", "is-active", "sovereign-bot"])
    return code == 0


async def get_bot_uptime() -> str | None:
    code, output = await _run_cmd([
        "systemctl", "--user", "show", "sovereign-bot",
        "--property=ActiveEnterTimestamp", "--value",
    ])
    if code == 0 and output:
        return output
    return None


async def restart_bot() -> tuple[bool, str]:
    logger.info("Restarting sovereign-bot")
    code, output = await _run_cmd(["systemctl", "--user", "restart", "sovereign-bot"])
    return code == 0, output or "Restart issued"


async def stop_bot() -> tuple[bool, str]:
    logger.info("Stopping sovereign-bot")
    code, output = await _run_cmd(["systemctl", "--user", "stop", "sovereign-bot"])
    return code == 0, output or "Stop issued"
