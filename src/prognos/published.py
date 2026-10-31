"""The closest comparison with published N-CMAPSS DS02 results this project can make.

Published DS02 work uses one split: train on units 2, 5, 10, 16, 18 and 20, test on
units 11, 14 and 15. This project's final test group holds DS02 units 5, 11 and 20
closed, so that split cannot be run here without opening it, and opening it is not
this module's decision to make.

What can be run without touching the closed group is the same split restricted to
the engines cross-validation already uses: train on DS02 units 2, 10, 16 and 18, test
on units 14 and 15. Two of six training engines are missing and one of three test
engines, and every difference is listed in `DIFFERENCES` and in the result file.
Nothing is tuned: the models use the Phase 3 configuration as committed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .dataset import WindowSource
from .gbm import LightGbmModel
from .metrics import nasa_score, rmse
from .models import LinearCycleBaseline

#: The DS02 split used in the literature, and the part of it runnable here.
PUBLISHED_TRAIN, PUBLISHED_TEST = (2, 5, 10, 16, 18, 20), (11, 14, 15)
TRAIN, TEST = (2, 10, 16, 18), (14, 15)
SUBDATASET = "DS02-006"

SOURCE = (
    "Wang et al., Temporal and Heterogeneous Graph Neural Network for Remaining Useful "
    "Life Prediction, arXiv:2405.04336v3, Table 5"
)


#: RMSE in cycles as printed in `SOURCE`, which collects results under the split above
#: (attributed there to Chao et al.). Each row's original paper is the one that table
#: cites. Copied, not recomputed.
def _row(method: str, cited_as: str, *rmse: float) -> dict[str, Any]:
    return {"method": method, "cited_as": cited_as, "rmse": dict(zip(COLUMNS, rmse, strict=True))}


COLUMNS = ("11", "14", "15", "all")
PUBLISHED: tuple[dict[str, Any], ...] = (
    _row("THGNN", "Wang et al.", 4.63, 5.64, 4.06, 5.59),
    _row("Transformer", "Mo et al.", 5.86, 7.69, 5.34, 6.54),
    _row("2D-CNN", "Zhao et al.", 6.38, 7.20, 5.20, 6.99),
    _row("Hybrid", "Chen et al.", 7.86, 8.11, 5.79, 8.06),
    _row("BLSTM", "Huang et al.", 8.76, 8.51, 5.41, 8.26),
)

#: Every way this run differs from the published protocol.
DIFFERENCES = (
    "Training engines: DS02 units 2, 10, 16 and 18, not 2, 5, 10, 16, 18 and 20. Units "
    "5 and 20 are in this project's closed final test group.",
    "Test engines: units 14 and 15 only; unit 11 is in the closed final test group. The "
    "published 'all' column includes unit 11 and is not comparable to anything here.",
    "Inputs: the 4 flight descriptors and 14 measured sensors. The cited table describes "
    "20 channels without listing them.",
    "Windows: 50 readings, every 10th reading within one cycle, RMSE averaged over "
    "windows. The cited table does not state its sampling or what one RMSE sample is.",
    "RUL cap: 125 cycles in training. DS02 engines live at most 89 cycles, so the cap "
    "never binds on them. The cited table does not state one.",
    "Engine age: every model here, LightGBM and the sequence models included, is given "
    "the cycle number, and DS02 engines all live 63 to 89 cycles, so age alone predicts "
    "much of the remaining life. The cited table does not say its models receive it.",
    "Seeds: one seed here. The cited table reports a mean over five subsamples.",
    "Models: the Phase 3 configurations as committed, trained on these four DS02 engines "
    "for this run. They are not the cross-validated models in the README's results "
    "table, which train on all three sub-datasets.",
)


#: What the numbers do and do not show, stated next to them.
CONCLUSION = (
    "Not a like-for-like comparison, and the figures here must not be read as beating the "
    "published ones. The cycle-only baseline, which sees no sensor at all, scores below "
    "every published method on unit 14. A gap that a model without sensors also opens is "
    "made by the protocol, chiefly the engine-age input and two test engines, not by "
    "better modelling. A like-for-like run needs the full published split, which means "
    "opening the closed final test group."
)


def _mask(data: WindowSource, units: tuple[int, ...]) -> npt.NDArray[np.bool_]:
    """Windows belonging to the given DS02 units."""
    wanted = [
        index
        for index, key in enumerate(data.group_labels)
        if key.subdataset == SUBDATASET and key.unit in units
    ]
    return np.isin(np.asarray(data.groups), np.asarray(wanted, dtype=np.int64))


def compare(data: WindowSource, seed: int, epochs: int = 30) -> dict[str, Any]:
    """Train on the runnable training units, score each runnable test unit."""
    from .sequence import HarnessAdapter, TrainConfig, _BaseSequenceModel, build_sequence_models

    train = data.subset(_mask(data, TRAIN))
    sequence = build_sequence_models(
        TrainConfig(epochs=epochs, seed=seed), names=("cnn", "transformer")
    )
    models: dict[str, Any] = {
        "linear_cycle": LinearCycleBaseline(),
        "lightgbm": LightGbmModel(seed=seed),
    }
    for name, model in sequence.items():
        assert isinstance(model, _BaseSequenceModel)
        models[name] = HarnessAdapter(model=model, seed=seed)
    targets = np.asarray(train.targets, dtype=np.float32)
    results: dict[str, Any] = {}
    for name, model in models.items():
        print(f"  {name}", flush=True)
        model.fit(train.windows, train.cycles, targets)
        per_unit: dict[str, float] = {}
        truths, preds = [], []
        for unit in TEST:
            test = data.subset(_mask(data, (unit,)))
            truth = np.asarray(test.targets, dtype=np.float64)
            pred = np.asarray(model.predict(test.windows, test.cycles), dtype=np.float64)
            per_unit[str(unit)] = rmse(truth, pred)
            truths.append(truth)
            preds.append(pred)
        both_truth, both_pred = np.concatenate(truths), np.concatenate(preds)
        results[name] = {
            "rmse": {**per_unit, "14+15": rmse(both_truth, both_pred)},
            "nasa_score": nasa_score(both_truth, both_pred),
            "n_test_windows": int(both_truth.size),
        }
    return {
        "published_split": {"train": list(PUBLISHED_TRAIN), "test": list(PUBLISHED_TEST)},
        "run_split": {"subdataset": SUBDATASET, "train": list(TRAIN), "test": list(TEST)},
        "published": list(PUBLISHED),
        "published_source": SOURCE,
        "differences": list(DIFFERENCES),
        "conclusion": CONCLUSION,
        "ours": results,
        "epochs": epochs,
        "seed": seed,
        "n_train_windows": int(train.windows.shape[0]),
    }


__all__ = ["CONCLUSION", "DIFFERENCES", "PUBLISHED", "SOURCE", "compare"]
