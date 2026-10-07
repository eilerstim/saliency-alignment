#!/usr/bin/env python3
"""Claude Code hook: push the session's latest message to a phone or channel.

Registered in .claude/settings.json for the ``Stop`` event (Claude finished a
turn) and the ``Notification`` event (Claude is waiting for permission or
input). Reads the hook payload from stdin and posts a short text to the URL in
``CLAUDE_NOTIFY_URL``. Does nothing when that variable is unset, so the hook
is inert for anyone who has not opted in.

Supported targets (detected from the URL):
  * ntfy  -- https://ntfy.sh/<hard-to-guess-topic>; install the ntfy app and
             subscribe to the same topic. No account needed.
  * Slack -- an incoming-webhook URL (hooks.slack.com/...).
  * anything else receives a plain-text POST.

The script never fails the hook (always exits 0) and never sends secrets:
only the assistant's own text, truncated, plus host and session id.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import urllib.request

MAX_CHARS = 3500  # ntfy accepts 4 KiB bodies


def last_assistant_text(transcript_path: str) -> str:
    """Last assistant message of the session transcript (JSONL)."""
    text = ""
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") != "assistant":
                    continue
                content = entry.get("message", {}).get("content")
                if isinstance(content, str):
                    candidate = content
                elif isinstance(content, list):
                    candidate = "\n".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                else:
                    candidate = ""
                if candidate.strip():
                    text = candidate
    except OSError:
        pass
    return text.strip()


def main() -> int:
    url = os.environ.get("CLAUDE_NOTIFY_URL", "").strip()
    if not url:
        return 0
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        payload = {}

    event = payload.get("hook_event_name", "Stop")
    session = str(payload.get("session_id", ""))[:8]
    host = socket.gethostname().split(".")[0]

    if event == "Notification":
        body = str(payload.get("message", "Claude Code is waiting for you."))
        kind = str(
            payload.get("title") or payload.get("notification_type") or "needs input"
        )
        title = f"Claude Code {kind} [{host} {session}]"
    else:
        # Stop payloads carry the final message directly; the transcript is
        # the fallback for older versions.
        body = str(payload.get("last_assistant_message") or "").strip()
        if not body:
            body = last_assistant_text(str(payload.get("transcript_path", "")))
        if not body:
            return 0
        title = f"Claude Code update [{host} {session}]"
    if len(body) > MAX_CHARS:
        body = body[: MAX_CHARS - 20].rstrip() + "\n[... truncated]"

    if "hooks.slack.com" in url:
        data = json.dumps({"text": f"*{title}*\n{body}"}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
    else:
        data = body.encode("utf-8")
        headers = {
            "Title": title.encode("ascii", "ignore").decode(),
            "Content-Type": "text/plain; charset=utf-8",
        }

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15):
            pass
    except Exception as exc:  # noqa: BLE001 - a failed push must not break the session
        print(f"notify.py: push failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
