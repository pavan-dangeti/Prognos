"""Serving RUL with a calibrated interval.

The inference service is the only component that holds model state, and it holds two
things rather than one: a fitted model and a conformal calibrator. Shipping the model
without the calibrator is the failure mode worth naming — predictions come back with
a bare point estimate, the dashboard draws a confident line, and nothing records that
the uncertainty was thrown away at the last step. They are therefore serialised
together and `RulPredictor.load` refuses a payload missing either.

LightGBM rather than the CNN is what ships. The CNN is 2.4 cycles better on cross-
validated RMSE, and torch costs roughly 800 MB resident per process; on a 12 GB
Always Free instance that is a quarter of the host spent on a demo that does not need
it. The trade is stated in the README rather than hidden, and the architecture does
not prevent shipping the CNN later — it is a different payload in the same slot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
from opentelemetry.propagate import inject
from opentelemetry.trace import SpanKind, Status, StatusCode

from prognos.gbm import LightGbmModel
from prognos.models import RulModel
from prognos.stream.observability import tracer
from prognos.uncertainty import Interval, SplitConformalCalibrator

MODEL_FILE = "model.pkl"
CALIBRATOR_FILE = "calibrator.json"


@dataclass(frozen=True, slots=True)
class Prediction:
    """One engine's current state: point estimate, interval, and where it came from."""

    unit: int
    cycle: int
    interval: Interval
    model: str
    target_coverage: float

    def to_dict(self) -> dict[str, object]:
        """JSON payload for the API and the dashboard."""
        return {
            "unit": self.unit,
            "cycle": self.cycle,
            "rul": self.interval.point,
            "lower": self.interval.lower,
            "upper": self.interval.upper,
            "model": self.model,
            "target_coverage": self.target_coverage,
        }


class RulPredictor:
    """A fitted model plus its calibrator, ready to serve."""

    __slots__ = ("_calibrator", "_model", "_model_name")

    def __init__(
        self, model: RulModel, calibrator: SplitConformalCalibrator, model_name: str
    ) -> None:
        """Pair a model with the calibrator that was fitted on held-out engines."""
        self._model = model
        self._calibrator = calibrator
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """Which model is serving."""
        return self._model_name

    @property
    def target_coverage(self) -> float:
        """The coverage the intervals were calibrated to."""
        return self._calibrator.target_coverage

    @property
    def calibration_half_width(self) -> float:
        """The conformal half-width, for reporting alongside any coverage figure.

        Width and coverage are the same number seen twice: an interval can be made to
        cover by being made useless, so a coverage claim without its width says nothing
        about whether the interval is fit to use.
        """
        return self._calibrator.width

    def predict(
        self, windows: npt.NDArray[np.float32], cycles: npt.NDArray[np.int64]
    ) -> list[Interval]:
        """Intervals for a batch of windows.

        Vectorised because the ingestion path hands over a whole batch, and a
        per-window call would recompute the feature matrix once per row.
        """
        if windows.shape[0] != cycles.shape[0]:
            raise ValueError(
                f"windows and cycles must be the same length, got "
                f"{windows.shape[0]} and {cycles.shape[0]}"
            )
        if windows.shape[0] == 0:
            return []
        with tracer.start_as_current_span(
            "inference.predict",
            attributes={"model": self._model_name, "windows": int(windows.shape[0])},
        ):
            point = np.asarray(self._model.predict(windows, cycles), dtype=np.float64)
        lower, upper = self._calibrator.bounds(point)
        return [
            Interval(lower=float(lo), point=float(p), upper=float(hi))
            for lo, p, hi in zip(lower, point, upper, strict=True)
        ]

    def save(self, directory: Path) -> None:
        """Write the model and the calibrator side by side.

        The calibrator is stored as the **calibration residuals**, not as a width. The
        conformal quantile depends on the sample size as well as the values — it is
        the ceil((n+1)*coverage)-th order statistic — so a bare width cannot be
        converted back. Persisting it was the first version of this and it produced
        an infinite interval on load, which looks like a working model with no
        uncertainty and is the worst of both.
        """
        import pickle

        directory.mkdir(parents=True, exist_ok=True)
        (directory / MODEL_FILE).write_bytes(pickle.dumps(self._model))
        (directory / CALIBRATOR_FILE).write_text(
            json.dumps(
                {
                    "target_coverage": self._calibrator.target_coverage,
                    "calibration_residuals": [
                        float(r) for r in self._calibrator.calibration_residuals
                    ],
                    "model": self._model_name,
                }
            )
        )

    @classmethod
    def load(cls, directory: Path) -> RulPredictor:
        """Read a predictor back, refusing an incomplete payload.

        A missing calibrator is the dangerous case: it is exactly the "point estimate
        only" degradation, and it would otherwise be discovered by someone noticing the
        dashboard has no bands.
        """
        import pickle

        model_path = directory / MODEL_FILE
        calibrator_path = directory / CALIBRATOR_FILE
        if not model_path.exists():
            raise FileNotFoundError(f"no model at {model_path}")
        if not calibrator_path.exists():
            raise FileNotFoundError(
                f"no calibrator at {calibrator_path}; an interval-free predictor would "
                "ship a point estimate that looks like an answer"
            )
        payload = json.loads(calibrator_path.read_text())
        residuals = np.asarray(payload["calibration_residuals"], dtype=np.float64)
        if residuals.size == 0:
            raise ValueError(
                f"{calibrator_path} carries no calibration residuals; the interval "
                "width cannot be reconstructed from a target rate alone"
            )
        calibrator = SplitConformalCalibrator(payload["target_coverage"]).calibrate(residuals)
        model: RulModel = pickle.loads(model_path.read_bytes())
        return cls(model=model, calibrator=calibrator, model_name=str(payload["model"]))

    @classmethod
    def train(
        cls,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
        targets: npt.NDArray[np.float32],
        calibration_windows: npt.NDArray[np.float32],
        calibration_cycles: npt.NDArray[np.int64],
        calibration_targets: npt.NDArray[np.float32],
        target_coverage: float,
        seed: int,
        model_name: str = "lightgbm",
    ) -> RulPredictor:
        """Fit a model and calibrate it on held-out data, in one step.

        `calibration_*` must come from engines the model was not fitted on. Passing
        the same rows twice produces a confident, wrong interval, and the difference
        is invisible in the served output — so the two sets are required as separate
        arguments rather than derived internally.
        """
        model = LightGbmModel(seed=seed)
        model.fit(windows, cycles, targets)
        residuals = np.asarray(
            model.predict(calibration_windows, calibration_cycles), dtype=np.float64
        ) - np.asarray(calibration_targets, dtype=np.float64)
        calibrator = SplitConformalCalibrator(target_coverage).calibrate(residuals)
        return cls(model=model, calibrator=calibrator, model_name=model_name)


class InferenceUnavailableError(RuntimeError):
    """The inference service could not be reached or refused to answer."""


class Predictor(Protocol):
    """What the pipeline needs from inference, in-process or over HTTP."""

    @property
    def model_name(self) -> str:
        """Which model is serving."""
        ...

    def predict(
        self, windows: npt.NDArray[np.float32], cycles: npt.NDArray[np.int64]
    ) -> list[Interval]:
        """Intervals for a batch of windows."""
        ...


class RemotePredictor:
    """The inference service, called over HTTP.

    Every failure to get an answer — refused connection, timeout, a 5xx, a body that
    is not a prediction — is raised as `InferenceUnavailableError`, one exception the
    pipeline turns into a named degraded state. Retrying here would hide an outage
    behind latency; the pipeline decides what an outage means, and it means no
    prediction rather than a stale one.
    """

    __slots__ = ("_timeout", "_token", "_url")

    def __init__(self, base_url: str, timeout: float = 2.0, token: str | None = None) -> None:
        """Point at a running inference service."""
        self._url = base_url.rstrip("/") + "/predict"
        self._timeout = timeout
        self._token = token

    @property
    def model_name(self) -> str:
        """Named for where it runs; the service reports the model it loaded."""
        return "remote"

    def predict(
        self, windows: npt.NDArray[np.float32], cycles: npt.NDArray[np.int64]
    ) -> list[Interval]:
        """Ask the service for intervals, or raise `InferenceUnavailableError`."""
        body = json.dumps(
            {"windows": np.asarray(windows).tolist(), "cycles": np.asarray(cycles).tolist()}
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        with tracer.start_as_current_span(
            "inference.remote_predict",
            kind=SpanKind.CLIENT,
            attributes={"url.full": self._url, "windows": int(np.asarray(windows).shape[0])},
        ) as current:
            # The caller's trace travels in the traceparent header, so the service's
            # handler span joins it: one trace from window to model across the socket.
            inject(headers)
            try:
                return self._call(body, headers)
            except InferenceUnavailableError as error:
                current.record_exception(error)
                current.set_status(Status(StatusCode.ERROR, "inference unavailable"))
                raise

    def _call(self, body: bytes, headers: dict[str, str]) -> list[Interval]:
        """The HTTP exchange itself, every failure raised as one error."""
        import urllib.request

        request = urllib.request.Request(self._url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read())
            return [
                Interval(lower=float(p["lower"]), point=float(p["rul"]), upper=float(p["upper"]))
                for p in payload["predictions"]
            ]
        except (OSError, ValueError, KeyError, TypeError) as error:
            # URLError and timeouts are OSErrors; a malformed body is the others.
            raise InferenceUnavailableError(f"{self._url}: {error}") from error
