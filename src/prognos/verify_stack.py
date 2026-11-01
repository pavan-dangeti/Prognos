"""Checks that a running compose stack is doing what the README says it does.

Three claims, each read from the running services rather than asserted:

1. **Grafana has live data.** Every Prometheus target is up, and every series a
   dashboard panel queries has a sample from the replay.
2. **A prediction is one trace across both services.** In a window that ended long
   enough ago for both exporters to have flushed, every inference request span is the
   child of a replay span in the same trace.
3. **The inference service reports the registered model version it serves.**
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

Fetch = Callable[[str, dict[str, str]], Any]

#: Long enough ago that both services' batch exporters have flushed the window.
SETTLED_SECONDS, WINDOW_SECONDS = 60, 5


def fetch_json(url: str, headers: dict[str, str]) -> Any:  # noqa: ANN401 - JSON
    """GET a URL and parse its JSON body."""
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def dashboard_series(dashboard: Path) -> list[str]:
    """Every Prometheus expression the dashboard's panels query."""
    panels = json.loads(dashboard.read_text())["panels"]
    return [
        target["expr"]
        for panel in panels
        if panel["datasource"]["type"] == "prometheus"
        for target in panel.get("targets", [])
        if target.get("expr")
    ]


def check_grafana_data(prometheus: str, dashboard: Path, fetch: Fetch) -> dict[str, Any]:
    """Targets up, and a sample behind every panel series that can have one."""
    targets = fetch(f"{prometheus}/api/v1/targets", {})["data"]["activeTargets"]
    health = {t["labels"]["instance"]: t["health"] for t in targets}
    series: dict[str, int] = {}
    for expr in dashboard_series(dashboard):
        query = urllib.parse.quote(expr)
        series[expr] = len(fetch(f"{prometheus}/api/v1/query?query={query}", {})["data"]["result"])
    # Series that only exist once something goes wrong (span errors, rejections by
    # reason) are empty on a healthy stack: reported, not required.
    required = {e: n for e, n in series.items() if "errors" not in e and "by_reason" not in e}
    return {
        "targets": health,
        "series_samples": series,
        "passed": all(h == "up" for h in health.values()) and all(required.values()),
    }


def check_traces(jaeger: str, fetch: Fetch, now: float | None = None) -> dict[str, Any]:
    """Pair inference request spans with replay client spans in a settled window."""
    end = int(((now if now is not None else time.time()) - SETTLED_SECONDS) * 1e6)
    start = end - WINDOW_SECONDS * 1_000_000
    window = f"limit=2000&start={start}&end={end}"
    replay = fetch(f"{jaeger}/api/traces?service=prognos-replay&{window}", {})["data"]
    inference = fetch(f"{jaeger}/api/traces?service=prognos-inference&{window}", {})["data"]
    clients = {
        (t["traceID"], s["spanID"])
        for t in replay
        for s in t["spans"]
        if s["operationName"] == "inference.remote_predict"
    }
    servers = [
        (t["traceID"], ref["spanID"])
        for t in inference
        for s in t["spans"]
        if s["operationName"] == "POST /predict"
        for ref in s.get("references", [])
        if ref["refType"] == "CHILD_OF"
    ]
    joined = sum((trace, parent) in clients for trace, parent in servers)
    both = sum(len({p["serviceName"] for p in t["processes"].values()}) > 1 for t in replay)
    return {
        "window_seconds": WINDOW_SECONDS,
        "replay_traces": len(replay),
        "replay_traces_spanning_both_services": both,
        "inference_request_spans": len(servers),
        "inference_spans_joined_to_a_replay_span": joined,
        "passed": bool(servers) and joined == len(servers),
    }


def check_registry(inference: str, token: str | None, fetch: Fetch) -> dict[str, Any]:
    """The version /readyz says is being served."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    ready = fetch(f"{inference}/readyz", headers)
    registry = ready.get("registry") or {}
    return {
        "status": ready.get("status"),
        "registry": registry,
        "passed": ready.get("status") == "ready" and bool(registry.get("version")),
    }


def verify(
    prometheus: str,
    jaeger: str,
    inference: str,
    dashboard: Path,
    token: str | None,
    fetch: Fetch = fetch_json,
) -> dict[str, Any]:
    """All three checks, and whether all passed."""
    checks = {
        "grafana_live_data": check_grafana_data(prometheus, dashboard, fetch),
        "one_trace_across_services": check_traces(jaeger, fetch),
        "registered_version_served": check_registry(inference, token, fetch),
    }
    return {"checks": checks, "all_passed": all(c["passed"] for c in checks.values())}


__all__ = ["check_grafana_data", "check_registry", "check_traces", "dashboard_series", "verify"]
