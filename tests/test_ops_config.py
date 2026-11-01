"""The committed Grafana and compose configuration must agree with itself.

The datasource file once carried a single `datasource` key with no URL. Grafana ignores
that shape, so every panel would have had no data, and no test read the file.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

PROVISIONING = Path("ops/grafana/provisioning/datasources/prometheus.yml")
DASHBOARD = Path("ops/grafana/dashboards/fleet.json")
COMPOSE = Path("docker-compose.yml")


def test_datasources_are_a_list_grafana_reads() -> None:
    datasources = yaml.safe_load(PROVISIONING.read_text())["datasources"]
    assert all(source["url"].startswith("http://") for source in datasources)
    assert {source["type"] for source in datasources} == {"prometheus", "jaeger"}


def test_every_panel_uses_a_provisioned_datasource() -> None:
    provisioned = {s["uid"] for s in yaml.safe_load(PROVISIONING.read_text())["datasources"]}
    for panel in json.loads(DASHBOARD.read_text())["panels"]:
        assert panel["datasource"]["uid"] in provisioned, panel["title"]


def test_every_prognos_service_exports_traces_to_the_compose_backend() -> None:
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    assert "jaeger" in services
    for name in ("ingestion", "inference", "alerting", "dashboard"):
        endpoint = services[name]["environment"]["OTEL_EXPORTER_OTLP_ENDPOINT"]
        assert endpoint == "http://jaeger:4318", name


def test_every_image_is_pinned_by_digest() -> None:
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    for name, service in services.items():
        if "image" in service:
            assert "@sha256:" in service["image"], f"{name} is not pinned by digest"


def test_every_healthcheck_uses_the_unauthenticated_probe() -> None:
    """Only /healthz is exempt from the bearer token.

    A probe on anything else reads 401 once a token is set and marks a working service
    unhealthy.
    """
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    for name, service in services.items():
        check = service.get("healthcheck")
        if check is None:
            continue
        probe = " ".join(check["test"])
        assert "/healthz" in probe and "/readyz" not in probe, name


def test_the_replay_is_scraped_and_traced() -> None:
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    replay = services["replay"]
    assert replay["environment"]["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://jaeger:4318"
    assert "--inference-url" in replay["command"]
    scrape = yaml.safe_load(Path("ops/prometheus.yml").read_text())
    targets = scrape["scrape_configs"][0]["static_configs"][0]["targets"]
    assert "replay:8090" in targets
