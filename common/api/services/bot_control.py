"""Bot control — NSSM service commands for sovereign-bot (Windows)."""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("api.control")

SVC_MAP = {
    "ftmo_100k": "sovereign-bot-ftmo",
    "bright_100k": "sovereign-bot-bf",
    "ttp_demo": "sovereign-bot-ttp",
}


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


async def is_bot_running(account_id: str = "ftmo_100k") -> bool:
    svc = SVC_MAP.get(account_id, "sovereign-bot-ftmo")
    code, output = await _run_cmd(["nssm", "status", svc])
    return "SERVICE_RUNNING" in output


async def get_bot_uptime(account_id: str = "ftmo_100k") -> str | None:
    svc = SVC_MAP.get(account_id, "sovereign-bot-ftmo")
    code, output = await _run_cmd(["nssm", "status", svc])
    if code == 0:
        return output
    return None


async def restart_bot(account_id: str = "ftmo_100k") -> tuple[bool, str]:
    svc = SVC_MAP.get(account_id, "sovereign-bot-ftmo")
    logger.info(f"Restarting {svc}")
    code, output = await _run_cmd(["nssm", "restart", svc])
    return code == 0, output or "Restart issued"


async def stop_bot(account_id: str = "ftmo_100k") -> tuple[bool, str]:
    svc = SVC_MAP.get(account_id, "sovereign-bot-ftmo")
    logger.info(f"Stopping {svc}")
    code, output = await _run_cmd(["nssm", "stop", svc])
    return code == 0, output or "Stop issued"


async def scan_bot(account_id: str) -> tuple[bool, str]:
    """Stop the bot, run --scan-now, then restart."""
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    venv_python = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
    run_bot = str(REPO_ROOT / "common" / "live" / "run_bot.py")

    gpu_map = {"ftmo_100k": "0", "bright_100k": "0", "ttp_demo": "0"}
    if account_id not in SVC_MAP:
        return False, f"Unknown account: {account_id}"

    svc_name = SVC_MAP[account_id]
    gpu = gpu_map.get(account_id, "0")

    # 1. Stop the running bot
    logger.info(f"Scan: stopping {svc_name}")
    await _run_cmd(["nssm", "stop", svc_name])

    # 2. Run scan-now
    logger.info(f"Scan: running --scan-now for {account_id}")
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["ENABLE_LIVE_TRADING"] = "1"
    proc = await asyncio.create_subprocess_exec(
        venv_python, "-u", run_bot,
        "--live", "--account-id", account_id, "--scan-now",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await proc.communicate()
    code = proc.returncode
    output = (stdout or b"").decode().strip()

    # 3. Restart the bot
    logger.info(f"Scan: restarting {svc_name}")
    await _run_cmd(["nssm", "start", svc_name])

    if code == 0:
        lines = output.split("\n")
        summary = [l for l in lines if any(k in l for k in
                   ["FORCE_SCAN", "FORCE_RESULT", "H1_CHECK", "H1_RESULT",
                    "SCAN_NOW_TOTAL", "SIGNAL", "ORDER_FILLED", "SCAN_DONE"])]
        return True, "\n".join(summary[-10:]) if summary else "Scan complete, no signals"
    else:
        return False, f"Scan failed (rc={code}): {output[-500:]}"
