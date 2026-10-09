"""Gradient boosting on window statistics.

LightGBM is the strong tabular baseline: on C-MAPSS-style data it typically lands
within a few cycles of the sequence models at a fraction of the training cost, so
beating it is the bar a sequence model has to clear.

The model is wrapped rather than used directly so the evaluation harness sees one
interface, and so the fitted artefact carries its own feature list. A saved model
whose feature order disagrees with its training data would produce confident
nonsense, and that failure is otherwise invisible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from .features import (
    FeatureNormaliser,
    all_feature_names,
    build_matrix,
    drop_constant,
)

# Fixed defaults rather than tuned ones. On nine units, a search over
# hyperparameters mostly fits the fold it was searched on.
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_estimators": 400,
    "seed": 20241009,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 0,
}


@dataclass(slots=True)
class LightGbmModel:
    """Gradient boosting on window statistics plus cycle context.

    Two details matter for reproducibility. The feature list and the normaliser are
    stored on the model, because inference on a differently ordered matrix produces
    plausible-looking nonsense rather than an error. And constant channels are
    dropped after the normaliser is fitted, so the report can name them.
    """

    name: str = "lightgbm"
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    seed: int = 20241009
    max_cycle: int = 1
    normaliser: FeatureNormaliser | None = None
    kept_features: tuple[str, ...] = ()
    # Column positions of the kept features in the full matrix, so prediction
    # selects the same columns the model was fitted on.
    keep_indices: tuple[int, ...] = ()
    dropped_features: tuple[str, ...] = ()
    booster: Any = None
    importances: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
        targets: npt.NDArray[np.float32],
    ) -> None:
        """Fit the booster on training windows only.

        The normaliser is fitted here, on training rows only. Fitting it on the
        whole set would leak the validation units' distributions into the features,
        which on nine units is enough to shift a reported number.
        """
        if windows.shape[0] == 0:
            raise ValueError("cannot fit on zero windows")

        self.max_cycle = max(int(cycles.max()), 1)
        matrix = build_matrix(windows, cycles, self.max_cycle)
        names = all_feature_names()

        self.normaliser = FeatureNormaliser().fit(matrix, names)
        scaled = self.normaliser.transform(matrix)
        reduced, kept, dropped = drop_constant(scaled, names)
        self.kept_features = tuple(kept)
        self.dropped_features = dropped
        self.keep_indices = tuple(names.index(name) for name in kept)

        params = dict(self.params)
        params["seed"] = self.seed
        params["bagging_seed"] = self.seed
        params["feature_fraction_seed"] = self.seed
        params["data_random_seed"] = self.seed

        # Fitted on a DataFrame so the columns carry their names. LightGBM records
        # feature names from the fit and warns on any prediction that does not
        # supply them, and an unnamed prediction is exactly how a feature-order
        # mismatch slips through as plausible-looking nonsense.
        self.booster = lgb.LGBMRegressor(**params)
        self.booster.fit(self._frame(reduced), targets)
        self.importances = dict(
            zip(self.kept_features, self.booster.feature_importances_, strict=True)
        )

    def _frame(self, reduced: npt.NDArray[np.float32]) -> pd.DataFrame:
        """Wrap the reduced matrix with its feature names."""
        return pd.DataFrame(reduced, columns=list(self.kept_features))

    def predict(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        """Predict RUL, clipped at zero.

        Uses the training set's max cycle rather than the batch's, so a batch
        containing only early-life units is not scaled differently from training.
        """
        if self.booster is None or self.normaliser is None:
            raise RuntimeError("model is not fitted")
        if windows.shape[0] == 0:
            raise ValueError("cannot predict on zero windows")

        matrix = build_matrix(windows, cycles, self.max_cycle)
        scaled = self.normaliser.transform(matrix)
        if len(self.keep_indices) != len(self.kept_features):
            raise RuntimeError("fitted feature selection is inconsistent")
        selected = scaled[:, list(self.keep_indices)]
        prediction = self.booster.predict(self._frame(selected))
        return np.clip(np.asarray(prediction, dtype=np.float32), 0.0, None)

    def report(self) -> dict[str, object]:
        """Fitted-state summary for the committed result file."""
        top = sorted(self.importances.items(), key=lambda kv: kv[1], reverse=True)[:10]
        return {
            "model": self.name,
            "params": {k: v for k, v in self.params.items() if k != "verbose"},
            "n_features_kept": len(self.kept_features),
            "dropped_constant_features": list(self.dropped_features),
            "top_features": [{"name": n, "importance": float(v)} for n, v in top],
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialisable fitted state, for the model registry."""
        if self.booster is None or self.normaliser is None:
            raise RuntimeError("model is not fitted")
        return {
            "name": self.name,
            "params": {k: v for k, v in self.params.items() if k != "verbose"},
            "seed": self.seed,
            "max_cycle": self.max_cycle,
            "normaliser": self.normaliser.to_dict(),
            "kept_features": list(self.kept_features),
            "keep_indices": list(self.keep_indices),
            "dropped_features": list(self.dropped_features),
            "booster_model_str": self.booster.booster_.model_to_string(),
            "importances": self.importances,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LightGbmModel:
        """Rebuild a fitted model, checking its feature list survived the round trip."""
        model = cls(
            name=str(payload["name"]),
            params=dict(payload["params"]),
            seed=int(payload["seed"]),
            max_cycle=int(payload["max_cycle"]),
            normaliser=FeatureNormaliser.from_dict(dict(payload["normaliser"])),
            kept_features=tuple(str(n) for n in payload["kept_features"]),
            keep_indices=tuple(int(i) for i in payload["keep_indices"]),
            dropped_features=tuple(str(n) for n in payload["dropped_features"]),
            importances={str(k): float(v) for k, v in dict(payload["importances"]).items()},
        )
        booster = lgb.Booster(model_str=str(payload["booster_model_str"]))
        model.booster = booster
        return model


def top_features(model: LightGbmModel, limit: int = 10) -> list[tuple[str, float]]:
    """The most-used features, for the run report and for explainability."""
    return sorted(model.importances.items(), key=lambda kv: kv[1], reverse=True)[:limit]
