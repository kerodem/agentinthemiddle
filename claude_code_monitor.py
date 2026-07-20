"""
Read-only mitmproxy addon for monitoring Claude Code CLI API traffic.

Never modifies requests or responses. Annotates every Anthropic API call with
model name, conversation turns, active tools, and token counts.

Recommended mode — reverse proxy (no cert installation required):
    mitmproxy --mode reverse:https://api.anthropic.com -p 8080 -s claude_code_monitor.py
    # Then in another terminal:
    ANTHROPIC_BASE_URL=http://localhost:8080 claude

Forward proxy mode (requires trusting the mitmproxy CA cert):
    mitmproxy -s claude_code_monitor.py
    HTTPS_PROXY=http://localhost:8080 ANTHROPIC_UNSAFE_ALLOW_PLAINTEXT=false claude
"""

import json
import re

from mitmproxy import ctx
from mitmproxy.http import HTTPFlow

_ANTHROPIC_RE = re.compile(r"(^|\.)anthropic\.com$", re.IGNORECASE)
_LOCALHOST_RE = re.compile(r"^(localhost|127\.\d+\.\d+\.\d+|::1)$", re.IGNORECASE)


class ClaudeCodeMonitor:
    """Passive, read-only observer of Claude Code ↔ Anthropic API traffic."""

    def configure(self, updated: set) -> None:
        ctx.options.intercept = ""  # never pause or hold flows for editing

    def request(self, flow: HTTPFlow) -> None:
        # Read-only: intentionally empty — no request modifications ever.
        pass

    def response(self, flow: HTTPFlow) -> None:
        if not _is_anthropic(flow):
            return
        _annotate(flow)


def _is_anthropic(flow: HTTPFlow) -> bool:
    """Match Anthropic API calls in both forward-proxy and reverse-proxy modes."""
    host = flow.request.pretty_host
    if _LOCALHOST_RE.match(host):
        # Reverse-proxy mode: mitmproxy is the upstream, all traffic is Anthropic.
        return True
    return bool(_ANTHROPIC_RE.search(host))


def _annotate(flow: HTTPFlow) -> None:
    try:
        req = json.loads(flow.request.content)
    except (json.JSONDecodeError, ValueError):
        return

    model = req.get("model", "?")
    messages = req.get("messages", [])
    tools = req.get("tools", [])
    streaming = req.get("stream", False)

    parts = [f"model={model}", f"turns={len(messages)}"]
    if tools:
        names = [t.get("name", "?") for t in tools]
        parts.append(f"tools=[{', '.join(names)}]")

    in_tok, out_tok = _extract_usage(flow, streaming)
    if in_tok is not None:
        parts.append(f"tokens={in_tok}↑/{out_tok}↓")

    annotation = " | ".join(parts)
    flow.comment = annotation
    ctx.log.info(f"[claude-code] {flow.request.path}  {annotation}")


def _extract_usage(
    flow: HTTPFlow, streaming: bool
) -> tuple[int | None, int | None]:
    content = flow.response.content
    if not content:
        return None, None
    if streaming:
        return _parse_sse(content)
    try:
        resp = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None, None
    usage = resp.get("usage", {})
    return usage.get("input_tokens"), usage.get("output_tokens")


def _parse_sse(content: bytes) -> tuple[int | None, int | None]:
    """Extract input/output token counts from an Anthropic SSE stream."""
    in_tok = out_tok = None
    for line in content.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw.strip() == "[DONE]":
            break
        try:
            evt = json.loads(raw)
        except json.JSONDecodeError:
            continue
        match evt.get("type"):
            case "message_start":
                in_tok = evt.get("message", {}).get("usage", {}).get(
                    "input_tokens", in_tok
                )
            case "message_delta":
                out_tok = evt.get("usage", {}).get("output_tokens", out_tok)
    return in_tok, out_tok


addons = [ClaudeCodeMonitor()]
