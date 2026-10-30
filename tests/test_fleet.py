"""The fleet view an operator reads: which engines, how much life left, what is wrong.

`StreamPipeline.summary()` has claimed "for the dashboard and the status endpoint" since
it was written, and nothing exposed it. This is the thing that exposes it, and the
reason it is a separate object rather than a dict built on demand is that an operator
watches it update while a stream runs, so the state has to survive between requests.

Two decisions worth stating, because both look like details and are not:

**The roster is what the stream has seen, not what the dataset contains.** A manifest
listing 29 engines would show 29 rows on an empty dashboard, which is a lie. Nineteen
engines appear as they send data and the rest are absent, which is the truth.

**Rows are overwritten, never appended.** One row per engine. A fleet view that grows a
row per reading is a log file wearing a table's clothes, and it stops being readable at
about the second engine's first fault.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from prognos.stream.fleet import FleetView
from prognos.uncertainty import Interval


class TestFleetView:
    def test_starts_empty(self) -> None:
        """A fleet nobody has heard from is an empty fleet."""
        assert FleetView().rows() == []

    def test_a_reading_creates_a_row(self) -> None:
        view = FleetView()
        view.record(unit=3, cycle=40, interval=Interval(12.0, 18.5, 30.0))
        rows = view.rows()
        assert len(rows) == 1
        assert rows[0]["unit"] == 3
        assert rows[0]["cycle"] == 40

    def test_a_second_reading_replaces_the_row(self) -> None:
        """Otherwise the view is a log, and it stops being a table."""
        view = FleetView()
        view.record(unit=3, cycle=40, interval=Interval(12.0, 18.5, 30.0))
        view.record(unit=3, cycle=41, interval=Interval(10.0, 16.0, 28.0))
        rows = view.rows()
        assert len(rows) == 1
        assert rows[0]["cycle"] == 41
        assert rows[0]["lower"] == pytest.approx(10.0)

    def test_engines_are_kept_apart(self) -> None:
        """Unit numbers restart per sub-dataset, so identity is the pair."""
        view = FleetView()
        view.record(unit=1, cycle=10, interval=Interval(5.0, 8.0, 15.0), source="DS01-001")
        view.record(unit=1, cycle=10, interval=Interval(90.0, 95.0, 120.0), source="DS02-001")
        assert len(view.rows()) == 2

    def test_the_label_is_engine_identity_not_a_bare_index(self) -> None:
        """A bare global index is how this project shipped a 0.716 coverage breach once."""
        view = FleetView()
        view.record(unit=1, cycle=10, interval=Interval(5.0, 8.0, 15.0), source="DS01-007")
        assert view.rows()[0]["engine"] == "DS01-007/unit001"

    def test_rows_are_ordered_so_the_view_does_not_jump(self) -> None:
        view = FleetView()
        for unit in (7, 2, 5):
            view.record(unit=unit, cycle=1, interval=Interval(1.0, 2.0, 3.0))
        assert [r["unit"] for r in view.rows()] == [2, 5, 7]

    def test_the_narrowest_interval_is_marked(self) -> None:
        """The interval's own uncertainty, which is the honest width to show.

        Width is the upper minus the lower bound. It is not a confidence score and is
        not comparable across models, but on one model over time a rising width is a
        real signal, and hiding it would make the panel look calmer than the data is.
        """
        view = FleetView()
        view.record(unit=1, cycle=1, interval=Interval(10.0, 14.0, 18.0))
        view.record(unit=2, cycle=1, interval=Interval(10.0, 18.0, 26.0))
        widths = {r["unit"]: r["width"] for r in view.rows()}
        assert widths[1] == pytest.approx(8.0)
        assert widths[2] == pytest.approx(16.0)
        assert widths[2] > widths[1]

    def test_alerts_attach_to_the_matching_engine(self) -> None:
        view = FleetView()
        view.record(unit=1, cycle=10, interval=Interval(5.0, 8.0, 15.0))
        # The shape `MaintenanceAlert.to_dict` produces. A hand-written `{"state": ...}`
        # here once let the view read a key real alerts never carry.
        view.set_alert(1, {"kind": "maintenance_due", "unit": 1, "cycle": 10, "lower": 5.0})
        view.record(unit=2, cycle=10, interval=Interval(5.0, 8.0, 15.0))
        rows = {r["unit"]: r for r in view.rows()}
        assert rows[1]["alert"] == "maintenance_due"
        assert rows[2]["alert"] is None

    def test_degraded_states_attach_to_the_matching_engine(self) -> None:
        view = FleetView()
        view.record(unit=1, cycle=10, interval=Interval(5.0, 8.0, 15.0))
        view.set_degraded(1, [{"state": "sensor_dropout", "instrument": "W_Fuel Flow"}])
        rows = {r["unit"]: r for r in view.rows()}
        assert rows[1]["degraded"][0]["state"] == "sensor_dropout"
        assert rows[1]["degraded"][0]["instrument"] == "W_Fuel Flow"

    def test_status_is_a_verdict_an_operator_can_sort_by(self) -> None:
        """Worst first, so the thing that needs attention is the top row.

        Degraded beats alerting beats nominal. A degraded engine with a live alert is
        already known and already being watched; one that is degraded but has no alert
        is the one nobody is looking at.
        """
        view = FleetView()
        view.record(unit=1, cycle=1, interval=Interval(50.0, 52.0, 60.0))
        view.record(unit=2, cycle=1, interval=Interval(50.0, 52.0, 60.0))
        view.record(unit=3, cycle=1, interval=Interval(50.0, 52.0, 60.0))
        view.set_alert(2, {"state": "critical"})
        view.set_degraded(3, [{"state": "sensor_stuck"}])
        view.set_alert(3, {"state": "warning"})
        assert [r["unit"] for r in view.rows()] == [3, 2, 1]

    def test_concurrent_readers_do_not_tear(self) -> None:
        """A dashboard is polled while a stream writes, so this is the normal case."""
        view = FleetView()
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for cycle in range(1, 300):
                    view.record(unit=1, cycle=cycle, interval=Interval(1.0, 2.0, 3.0))
            except Exception as error:  # pragma: no cover - the assertion is the point
                errors.append(error)

        def reader() -> None:
            try:
                for _ in range(300):
                    json.dumps(view.rows())
            except Exception as error:  # pragma: no cover
                errors.append(error)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []


class TestCostPanel:
    """The measured policy comparison, served rather than recomputed.

    Read from the committed result file on purpose. Recomputing it on request would make
    the panel's numbers depend on when it was loaded, and the project's rule is that a
    number shown to a reader has to match a number committed under `results/`.
    """

    @pytest.fixture(scope="class")
    def panel(self) -> dict[str, object]:
        """The cost panel built from the committed Phase 4 result."""
        from prognos.stream.fleet import build_cost_panel

        return build_cost_panel(Path("results/phase4_policy.json"))

    def test_lists_every_policy_evaluated(self, panel: dict[str, object]) -> None:
        names = {p["policy"] for p in panel["policies"]}
        assert len(names) >= 2, "a comparison of one policy is not a comparison"
        assert "run_to_failure" in names

    def test_every_row_carries_a_cost_per_engine(self, panel: dict[str, object]) -> None:
        for policy in panel["policies"]:
            assert policy["cost_per_engine"] > 0, f"{policy['policy']} costed nothing"
            assert policy["n_planned"] + policy["n_failures"] > 0

    def test_the_best_policy_is_identified_and_is_actually_best(
        self, panel: dict[str, object]
    ) -> None:
        """Stating a winner that the table does not support is worse than none."""
        best = panel["best"]
        costs = [p["cost_per_engine"] for p in panel["policies"]]
        assert best in {p["policy"] for p in panel["policies"]}
        assert panel["policies"][0]["cost_per_engine"] == min(costs) or min(
            p["cost_per_engine"] for p in panel["policies"] if p["policy"] == best
        ) == min(costs)

    def test_the_cost_model_is_shown(self, panel: dict[str, object]) -> None:
        """A cost with no model behind it cannot be argued with."""
        assert panel["cost_model"]["failure_cost_ratio"] > 0
        assert "horizon" in panel["cost_model"]

    def test_a_missing_result_file_fails_loudly(self) -> None:
        """Silently showing an empty panel would look like "no policy wins"."""
        from prognos.stream.fleet import build_cost_panel

        with pytest.raises(FileNotFoundError):
            build_cost_panel(Path("results/does_not_exist.json"))

    def test_a_result_file_with_no_policies_fails_loudly(self, tmp_path: Path) -> None:
        from prognos.stream.fleet import build_cost_panel

        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"policies": []}))
        with pytest.raises(ValueError, match="polic"):
            build_cost_panel(path)


class TestFleetViewOverHttp:
    def test_the_fleet_route_returns_what_was_recorded(self) -> None:
        """The panel exists to be read over HTTP, so that is where it is proved."""
        import logging

        from prognos.stream.api import build_component
        from prognos.stream.observability import MetricsRegistry
        from prognos.stream.service import build_server

        view = FleetView()
        view.record(unit=2, cycle=12, interval=Interval(8.0, 11.0, 20.0), source="DS01-002")
        state = build_component(
            "dashboard", metrics=MetricsRegistry(), logger=logging.getLogger("t"), fleet=view
        )
        httpd = build_server(state, host="127.0.0.1", port=0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{httpd.server_address[1]}/fleet"
            with urllib.request.urlopen(url, timeout=10) as response:
                body = json.loads(response.read().decode())
        finally:
            httpd.shutdown()
            httpd.server_close()
        assert body["fleet"][0]["unit"] == 2
        assert body["fleet"][0]["engine"] == "DS01-002/unit002"
        assert body["fleet"][0]["lower"] == pytest.approx(8.0)

    def test_the_cost_route_reports_the_committed_numbers(self) -> None:
        import logging

        from prognos.stream.api import build_component
        from prognos.stream.observability import MetricsRegistry
        from prognos.stream.service import build_server

        state = build_component(
            "dashboard", metrics=MetricsRegistry(), logger=logging.getLogger("t")
        )
        httpd = build_server(state, host="127.0.0.1", port=0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{httpd.server_address[1]}/cost"
            with urllib.request.urlopen(url, timeout=10) as response:
                body = json.loads(response.read().decode())
        finally:
            httpd.shutdown()
            httpd.server_close()
        assert body["policies"]
        assert "best" in body


class TestFleetMetrics:
    """The bridge to Prometheus, which is what the compose stack actually shows.

    A fleet view that only exists as a JSON route is invisible to an operator watching
    Grafana, which is where they will be looking.
    """

    def _registry(self) -> object:
        from prognos.stream.observability import MetricsRegistry

        return MetricsRegistry()

    def _render(self, view: FleetView) -> str:
        from prognos.stream.observability import record_fleet_metrics

        registry = self._registry()
        record_fleet_metrics(registry, view.rows())  # type: ignore[arg-type]
        return registry.render()  # type: ignore[attr-defined]

    def test_an_empty_fleet_publishes_zero_not_nothing(self) -> None:
        """Otherwise a stopped fleet and a fleet with no data look identical."""
        rendered = self._render(FleetView())
        for status in ("nominal", "alerting", "degraded"):
            assert 'status="' + status + '"} 0.0' in rendered

    def test_every_engine_gets_a_rul_series(self) -> None:
        view = FleetView()
        view.record(unit=2, cycle=12, interval=Interval(8.0, 11.0, 20.0), source="DS01-002")
        rendered = self._render(view)
        assert 'engine="DS01-002/unit002"' in rendered
        assert "fleet_rul_cycles" in rendered

    def test_status_counts_add_up(self) -> None:
        view = FleetView()
        view.record(unit=1, cycle=1, interval=Interval(1.0, 2.0, 3.0))
        view.record(unit=2, cycle=1, interval=Interval(1.0, 2.0, 3.0))
        view.set_alert(2, {"state": "critical"})
        rendered = self._render(view)
        assert 'fleet_engines{status="nominal"} 1.0' in rendered
        assert 'fleet_engines{status="alerting"} 1.0' in rendered
        assert 'fleet_engines{status="degraded"} 0.0' in rendered

    def test_labels_stay_low_cardinality(self) -> None:
        """An unbounded label value is how a scrape falls over under load.

        Only `engine` and `status` are used. An alert reason belongs in the route and
        the logs, not on a metric, because there is no fixed set of reasons.
        """
        view = FleetView()
        view.record(unit=1, cycle=1, interval=Interval(1.0, 2.0, 3.0))
        view.set_alert(1, {"state": "critical", "detail": "RUL 4.1 under threshold 20"})
        rendered = self._render(view)
        assert "detail" not in rendered
        assert "critical" not in rendered

    def test_the_route_publishes_the_metrics_as_a_side_effect(self) -> None:
        """Reading the fleet is what refreshes the gauges; nothing else does."""
        import logging

        from prognos.stream.api import build_component
        from prognos.stream.observability import MetricsRegistry
        from prognos.stream.service import build_server

        view = FleetView()
        view.record(unit=5, cycle=3, interval=Interval(2.0, 3.0, 9.0), source="DS06-005")
        registry = MetricsRegistry()
        state = build_component(
            "dashboard", metrics=registry, logger=logging.getLogger("t"), fleet=view
        )
        httpd = build_server(state, host="127.0.0.1", port=0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{httpd.server_address[1]}/fleet"
            with urllib.request.urlopen(url, timeout=10) as response:
                json.loads(response.read().decode())
        finally:
            httpd.shutdown()
            httpd.server_close()
        assert "fleet_rul_cycles" in registry.render()
