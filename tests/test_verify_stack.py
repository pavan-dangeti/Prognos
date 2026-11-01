"""The stack verifier, against canned responses from each service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prognos.verify_stack import check_traces, dashboard_series, verify

DASHBOARD = Path("ops/grafana/dashboards/fleet.json")


def trace(trace_id: str, spans: list[dict[str, Any]], services: list[str]) -> dict[str, Any]:
    return {
        "traceID": trace_id,
        "spans": spans,
        "processes": {f"p{i}": {"serviceName": s} for i, s in enumerate(services)},
    }


def span(name: str, span_id: str, parent: str | None = None) -> dict[str, Any]:
    refs = [{"refType": "CHILD_OF", "spanID": parent}] if parent else []
    return {"operationName": name, "spanID": span_id, "references": refs}


def fake(joined: bool, version: str | None = "4") -> Any:
    replay = [trace("t1", [span("inference.remote_predict", "c1")], ["prognos-replay"])]
    server_trace = "t1" if joined else "t9"
    inference = [trace(server_trace, [span("POST /predict", "s1", "c1")], ["prognos-inference"])]

    def fetch(url: str, headers: dict[str, str]) -> Any:
        if "/api/v1/targets" in url:
            return {
                "data": {"activeTargets": [{"labels": {"instance": "replay:8090"}, "health": "up"}]}
            }
        if "/api/v1/query" in url:
            empty = "errors" in url or "by_reason" in url
            return {"data": {"result": [] if empty else [{"value": [0, "1"]}]}}
        if "service=prognos-replay" in url:
            return {"data": replay}
        if "service=prognos-inference" in url:
            return {"data": inference}
        registry = {"version": version} if version else None
        return {"status": "ready", "registry": registry}

    return fetch


def test_the_dashboard_series_are_read_from_the_committed_file() -> None:
    assert "prognos_interval_coverage" in dashboard_series(DASHBOARD)


def test_a_healthy_stack_passes_every_check() -> None:
    report = verify("p", "j", "i", DASHBOARD, token="t", fetch=fake(joined=True))
    assert report["all_passed"], report


def test_a_trace_broken_at_the_socket_fails() -> None:
    result = check_traces("j", fake(joined=False), now=1_000_000.0)
    assert not result["passed"]
    assert result["inference_spans_joined_to_a_replay_span"] == 0


def test_a_service_without_a_registry_version_fails() -> None:
    report = verify("p", "j", "i", DASHBOARD, token=None, fetch=fake(True, version=None))
    assert not report["checks"]["registered_version_served"]["passed"]


def test_the_readme_quotes_the_committed_verification() -> None:
    import json

    result = Path("results/stack_verification.json")
    report = json.loads(result.read_text())
    readme = " ".join(Path("README.md").read_text().split())
    traces = report["checks"]["one_trace_across_services"]
    assert (
        f"{traces['inference_spans_joined_to_a_replay_span']} of "
        f"{traces['inference_request_spans']} inference request spans"
    ) in readme
    assert (
        f"{traces['replay_traces_spanning_both_services']} of {traces['replay_traces']} "
        "replay traces span both services"
    ) in readme
    targets = report["checks"]["grafana_live_data"]["targets"]
    assert f"all {len(targets)} Prometheus targets up" in readme
    version = report["checks"]["registered_version_served"]["registry"]["version"]
    assert f"`prognos-rul` version {version}" in readme
    assert report["all_passed"]
