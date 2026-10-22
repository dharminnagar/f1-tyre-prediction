"""Model 4: LSTM/GRU encoder–decoder relative-pace model over lap sequences.

The encoder reads the driver's last ``window`` laps up to the decision lap; its final state
seeds a decoder that reads the known future schedule (compound, tyre age, stint lap, lap
number, race progress, new-stint flag). Each decoder step predicts ``y_s`` as an offset from
a baseline: the mean clean ``y_s - wear`` in the history window plus the wear at the step's
own compound and tyre age. ``wear`` is the tyre prior plus a learned non-negative linear slope
per compound (``LstmPaceNet.extra_wear``), so the recurrent offsets only carry the driver's
level and departures from a monotone wear curve; a shrunk or under-trained network falls
back to that curve's slope instead of a flat pace.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict

from boxbox.config import LstmParams
from boxbox.features.tyre_prior import (
    POOLED_DRY,
    TyrePrior,
    prior_from_summary,
    prior_summary,
    tyre_prior_curve,
)
from boxbox.models.sequence_data import (
    FUTURE_FEATURES,
    HISTORY_CLEAN_INDEX,
    HISTORY_FEATURES,
    HISTORY_Y_INDEX,
    BoolArray,
    SequenceBatch,
    build_sequences,
    future_exists,
)
from boxbox.models.types import FloatArray, TyreAgeClip
from boxbox.schema import MODEL_COMPOUNDS

WEIGHTS_FILE: Final[str] = "lstm.pt"
META_FILE: Final[str] = "lstm.meta.json"
PREDICT_CHUNK: Final[int] = 4096
HUBER_DELTA_S: Final[float] = 1.0
EXTRA_WEAR_INIT: Final[float] = -5.0
"""Raw (pre-softplus) extra wear slope at initialisation: about 0.007 s/lap."""
INNER_VAL_SEASONS: Final[int] = 2

RnnState = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class LstmPaceNet(torch.nn.Module):
    """Encoder–decoder recurrent network mapping (history, future schedule) to ``y_s`` offsets."""

    def __init__(self, n_history: int, n_future: int, params: LstmParams) -> None:
        super().__init__()
        rnn = torch.nn.LSTM if params.cell == "lstm" else torch.nn.GRU
        inner_dropout = params.dropout if params.layers > 1 else 0.0
        self.encoder = rnn(
            n_history + 1, params.hidden, params.layers, batch_first=True, dropout=inner_dropout
        )
        self.decoder = rnn(
            n_future, params.hidden, params.layers, batch_first=True, dropout=inner_dropout
        )
        self.dropout = torch.nn.Dropout(params.dropout)
        self.head = torch.nn.Linear(params.hidden + n_future, 1)
        self.extra_wear = torch.nn.Parameter(torch.full((len(MODEL_COMPOUNDS),), EXTRA_WEAR_INIT))

    def extra_wear_slopes(self) -> torch.Tensor:
        """[C] learned wear slope (s/lap) added to the prior per ``MODEL_COMPOUNDS`` entry."""
        return torch.nn.functional.softplus(self.extra_wear)

    def forward(
        self, history: torch.Tensor, history_mask: torch.Tensor, future: torch.Tensor
    ) -> torch.Tensor:
        """[B, W, F_h] scaled history, [B, W] mask, [B, H, F_f] scaled future -> [B, H]."""
        encoder_in = torch.cat([history, history_mask.unsqueeze(-1)], dim=-1)
        encoded: tuple[torch.Tensor, RnnState] = self.encoder(encoder_in)
        decoded: tuple[torch.Tensor, RnnState] = self.decoder(future, encoded[1])
        features = torch.cat([self.dropout(decoded[0]), future], dim=-1)
        out: torch.Tensor = self.head(features)
        return out.squeeze(-1)


@dataclass(frozen=True)
class Scaler:
    """Per-feature mean/std fit on training samples only."""

    history_mean: FloatArray
    history_std: FloatArray
    future_mean: FloatArray
    future_std: FloatArray


@dataclass(frozen=True)
class TrainedLstm:
    """Fitted sequence model; satisfies ``SequencePaceModel``."""

    network: LstmPaceNet
    params: LstmParams
    scaler: Scaler
    tyre_clip: TyreAgeClip
    prior: TyrePrior
    curve: tuple[tuple[int, float, float], ...]
    """(epoch, train loss, inner-validation loss) per epoch; inner loss NaN without inner split."""
    best_epoch: int
    history_features: tuple[str, ...] = HISTORY_FEATURES
    future_features: tuple[str, ...] = FUTURE_FEATURES
    train_seconds: float = 0.0
    n_train_samples: int = 0
    inner_val_seasons: tuple[int, ...] = field(default_factory=tuple)

    @property
    def name(self) -> Literal["lstm"]:
        return "lstm"

    @property
    def horizon(self) -> int:
        return self.params.horizon

    @property
    def window(self) -> int:
        return self.params.window

    def predict_rollout(
        self, history: FloatArray, history_mask: BoolArray, future: FloatArray
    ) -> FloatArray:
        """Predicted ``y_s`` for each future step of one decision (``future`` is [H', F_fut])."""
        steps = future.shape[0]
        if steps > self.horizon:
            raise ValueError(f"rollout of {steps} laps exceeds the trained horizon {self.horizon}")
        if steps == 0:
            return np.zeros(0, dtype=np.float64)
        predicted = _predict(self, history[None], history_mask[None], future[None])
        return np.asarray(predicted[0], dtype=np.float64)

    def predict_rollouts(
        self, history: FloatArray, history_mask: BoolArray, future: FloatArray
    ) -> FloatArray:
        """[N, H'] predictions for N decisions at once (``future`` is [N, H', F_fut]).

        Trailing all-zero future rows are padding: the decoder is causal, so they never change
        the predictions of earlier steps.
        """
        steps = future.shape[1]
        if steps > self.horizon:
            raise ValueError(f"rollout of {steps} laps exceeds the trained horizon {self.horizon}")
        if steps == 0 or future.shape[0] == 0:
            return np.zeros((future.shape[0], steps), dtype=np.float64)
        return _predict(self, history, history_mask, future)


def clip_tyre_ages(arrays: FloatArray, names: tuple[str, ...], clip: TyreAgeClip) -> FloatArray:
    """Cap ``tyre_age``/``stint_lap`` at the per-compound training cap (like other models)."""
    default_cap = max(clip.max_age.values())
    caps = np.array([clip.max_age.get(c, default_cap) for c in MODEL_COMPOUNDS])
    one_hot = arrays[..., [names.index(f"compound_{c}") for c in MODEL_COMPOUNDS]]
    has_compound = one_hot.sum(axis=-1) > 0
    cap = np.where(has_compound, one_hot @ caps, default_cap)
    out = arrays.copy()
    for column in ("tyre_age", "stint_lap"):
        i = names.index(column)
        out[..., i] = np.minimum(out[..., i], cap)
    return out


def prior_on_steps(prior: TyrePrior, arrays: FloatArray, names: tuple[str, ...]) -> FloatArray:
    """Tyre prior at each step's compound (one-hot columns) and ``tyre_age`` (shape [...]).

    Steps without a model compound (wet laps, padding) use the pooled dry curve.
    """
    age = arrays[..., names.index("tyre_age")]
    one_hot = arrays[..., [names.index(f"compound_{c}") for c in MODEL_COMPOUNDS]]
    out = tyre_prior_curve(prior, POOLED_DRY, age)
    for i, compound in enumerate(MODEL_COMPOUNDS):
        on = one_hot[..., i] > 0
        out = np.where(on, tyre_prior_curve(prior, compound, age), out)
    return np.asarray(out, dtype=np.float64)


def _level_weights(history: FloatArray, history_mask: BoolArray) -> FloatArray:
    """[N, W] averaging weights over clean history laps (all usable laps if none is clean)."""
    clean = history_mask & (history[..., HISTORY_CLEAN_INDEX] > 0)
    use = np.where(clean.any(axis=1, keepdims=True), clean, history_mask)
    counts = use.sum(axis=1, keepdims=True)
    return np.asarray(use / np.maximum(counts, 1), dtype=np.float64)


def _wear_basis(arrays: FloatArray, names: tuple[str, ...]) -> FloatArray:
    """[..., C] tyre age in the step's compound column (zeros without a model compound)."""
    one_hot = arrays[..., [names.index(f"compound_{c}") for c in MODEL_COMPOUNDS]]
    return np.asarray(one_hot * arrays[..., names.index("tyre_age"), None], dtype=np.float64)


def _fit_scaler(batch: SequenceBatch) -> Scaler:
    history_rows = batch.history[batch.history_mask]
    future_rows = batch.future[future_exists(batch.future)]

    def std(rows: FloatArray) -> FloatArray:
        s = rows.std(axis=0)
        return np.asarray(np.where(s > 1e-6, s, 1.0), dtype=np.float64)

    return Scaler(
        history_mean=np.asarray(history_rows.mean(axis=0), dtype=np.float64),
        history_std=std(history_rows),
        future_mean=np.asarray(future_rows.mean(axis=0), dtype=np.float64),
        future_std=std(future_rows),
    )


@dataclass(frozen=True)
class _Tensors:
    history: torch.Tensor
    history_mask: torch.Tensor
    future: torch.Tensor
    baseline: torch.Tensor
    """[N, H] mean clean ``y_s - prior`` of the history plus the prior of each future step."""
    level_weights: torch.Tensor
    history_wear: torch.Tensor
    """[N, W, C] history tyre age per compound column (for the learned extra wear)."""
    future_wear: torch.Tensor
    same_set: torch.Tensor
    """[N, H - 1] 1 where steps h and h + 1 are both real laps on the same tyre set."""


def _to_tensors(
    model_scaler: Scaler,
    clip: TyreAgeClip,
    prior: TyrePrior,
    history: FloatArray,
    history_mask: BoolArray,
    future: FloatArray,
) -> _Tensors:
    """Clip, standardise and zero padding, as float32 tensors.

    The prior is read at the clipped tyre ages: wear past the cap is added by the forecaster
    (``predict_branches``), as for the per-lap models.
    """
    hist = clip_tyre_ages(history, HISTORY_FEATURES, clip)
    fut = clip_tyre_ages(future, FUTURE_FEATURES, clip)
    weights = _level_weights(history, history_mask)
    residual = history[..., HISTORY_Y_INDEX] - prior_on_steps(prior, hist, HISTORY_FEATURES)
    level = (weights * residual).sum(axis=1)
    baseline = level[:, None] + prior_on_steps(prior, fut, FUTURE_FEATURES)
    history_wear = _wear_basis(hist, HISTORY_FEATURES)
    future_wear = np.where(future_exists(future)[..., None], _wear_basis(fut, FUTURE_FEATURES), 0.0)
    exists = future_exists(future)
    new_stint = future[..., FUTURE_FEATURES.index("new_stint")]
    same_set = exists[:, 1:] & exists[:, :-1] & (new_stint[:, 1:] == new_stint[:, :-1])
    hist = (hist - model_scaler.history_mean) / model_scaler.history_std
    hist = np.where(history_mask[..., None], hist, 0.0)
    fut = (fut - model_scaler.future_mean) / model_scaler.future_std
    fut = np.where(future_exists(future)[..., None], fut, 0.0)
    return _Tensors(
        history=torch.from_numpy(hist.astype(np.float32)),
        history_mask=torch.from_numpy(history_mask.astype(np.float32)),
        future=torch.from_numpy(fut.astype(np.float32)),
        baseline=torch.from_numpy(baseline.astype(np.float32)),
        level_weights=torch.from_numpy(weights.astype(np.float32)),
        history_wear=torch.from_numpy(history_wear.astype(np.float32)),
        future_wear=torch.from_numpy(future_wear.astype(np.float32)),
        same_set=torch.from_numpy(same_set.astype(np.float32)),
    )


def _forward(
    net: LstmPaceNet, t: _Tensors, rows: torch.Tensor | slice
) -> tuple[torch.Tensor, torch.Tensor]:
    """Absolute ``y_s`` predictions [B, H] and the recurrent offsets within them."""
    offsets: torch.Tensor = net(t.history[rows], t.history_mask[rows], t.future[rows])
    slopes = net.extra_wear_slopes()
    history_extra = (t.level_weights[rows] * (t.history_wear[rows] @ slopes)).sum(dim=1)
    extra = t.future_wear[rows] @ slopes - history_extra.unsqueeze(-1)
    return offsets + t.baseline[rows] + extra, offsets


def _masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = torch.nn.functional.huber_loss(pred, target, reduction="none", delta=HUBER_DELTA_S)
    return (losses * mask).sum() / mask.sum().clamp(min=1.0)


def _offset_trend(offsets: torch.Tensor, t: _Tensors, rows: torch.Tensor) -> torch.Tensor:
    """Mean squared lap-to-lap change of the recurrent offsets within one tyre set."""
    step = offsets[:, 1:] - offsets[:, :-1]
    same = t.same_set[rows]
    return (step.square() * same).sum() / same.sum().clamp(min=1.0)


def _predict(
    model: TrainedLstm, history: FloatArray, history_mask: BoolArray, future: FloatArray
) -> FloatArray:
    tensors = _to_tensors(model.scaler, model.tyre_clip, model.prior, history, history_mask, future)
    model.network.eval()
    chunks: list[FloatArray] = []
    with torch.no_grad():
        for start in range(0, history.shape[0], PREDICT_CHUNK):
            rows = slice(start, start + PREDICT_CHUNK)
            chunks.append(np.asarray(_forward(model.network, tensors, rows)[0].numpy(), np.float64))
    if not chunks:
        return np.zeros((0, future.shape[1]), dtype=np.float64)
    return np.concatenate(chunks)


def predict_batch(model: TrainedLstm, batch: SequenceBatch) -> FloatArray:
    """[N, H] predicted ``y_s`` for every sample and future step (padding steps included)."""
    return _predict(model, batch.history, batch.history_mask, batch.future)


def training_samples(
    features: pd.DataFrame, params: LstmParams
) -> tuple[SequenceBatch, BoolArray, tuple[int, ...]]:
    """Training-split samples with >= 1 target, the inner-validation mask and its seasons.

    Excluded eras are dropped before sequences are built. Samples of the last
    ``INNER_VAL_SEASONS`` training seasons form the inner early-stopping split (one season when
    there are fewer than ``INNER_VAL_SEASONS + 1``, none with a single training season): one
    race per season, so a one-season inner split stops on a single race's noise.
    """
    train_rows = features[
        (features["split"] == "train") & ~features["era"].astype(str).isin(params.exclude_eras)
    ]
    batch = build_sequences(train_rows, params.window, params.horizon)
    on_stride = batch.keys["lap"].to_numpy(dtype=np.int64) % params.decision_stride == 0
    batch = batch.subset(np.asarray(batch.target_mask.any(axis=1) & on_stride, dtype=np.bool_))
    race_season = train_rows.groupby("race_id")["season"].first()
    seasons = batch.keys["race_id"].map(race_season).to_numpy(dtype=np.int64)
    distinct = np.unique(seasons)
    if len(distinct) < 2:
        return batch, np.zeros(len(batch), dtype=np.bool_), ()
    n_inner = INNER_VAL_SEASONS if len(distinct) > INNER_VAL_SEASONS else 1
    inner_seasons = tuple(int(s) for s in distinct[-n_inner:])
    return batch, np.isin(seasons, inner_seasons), inner_seasons


def train_lstm(
    features: pd.DataFrame, params: LstmParams, tyre_clip: TyreAgeClip, prior: TyrePrior
) -> TrainedLstm:
    """Fit the encoder–decoder on ``split == "train"`` rows (CPU, deterministic for a seed).

    ``prior`` is the baseline wear curve the network's offsets are added to (see the module
    docstring); it should be fitted on the same training rows.

    Early stopping monitors the masked Huber loss on the last training seasons (see
    :func:`training_samples`) so the real validation season stays untouched for model
    selection; the best inner epoch's weights are kept. Without a second training season it
    trains for ``epochs`` and keeps the last weights.
    """
    started = time.perf_counter()
    batch, inner, inner_seasons = training_samples(features, params)
    if len(batch) == 0:
        raise ValueError("no training sequences (check split/era filters)")
    fit_part = batch.subset(~inner)
    scaler = _fit_scaler(fit_part)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(params.seed)
        network = LstmPaceNet(len(HISTORY_FEATURES), len(FUTURE_FEATURES), params)
        curve, best_epoch = _fit(
            network, fit_part, batch.subset(inner), scaler, tyre_clip, prior, params
        )
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
    network.eval()
    return TrainedLstm(
        network=network,
        params=params,
        scaler=scaler,
        tyre_clip=tyre_clip,
        prior=prior,
        curve=curve,
        best_epoch=best_epoch,
        train_seconds=time.perf_counter() - started,
        n_train_samples=len(fit_part),
        inner_val_seasons=inner_seasons,
    )


def _fit(
    network: LstmPaceNet,
    fit_part: SequenceBatch,
    inner_part: SequenceBatch,
    scaler: Scaler,
    clip: TyreAgeClip,
    prior: TyrePrior,
    params: LstmParams,
) -> tuple[tuple[tuple[int, float, float], ...], int]:
    """Adam over shuffled mini-batches; returns the curve and the epoch whose weights are kept."""
    train = _to_tensors(
        scaler, clip, prior, fit_part.history, fit_part.history_mask, fit_part.future
    )
    target = torch.from_numpy(np.nan_to_num(fit_part.target).astype(np.float32))
    mask = torch.from_numpy(fit_part.target_mask.astype(np.float32))
    inner = _to_tensors(
        scaler, clip, prior, inner_part.history, inner_part.history_mask, inner_part.future
    )
    inner_target = torch.from_numpy(np.nan_to_num(inner_part.target).astype(np.float32))
    inner_mask = torch.from_numpy(inner_part.target_mask.astype(np.float32))
    has_inner = len(inner_part) > 0

    # Decay would pull the raw extra-wear parameter towards 0, i.e. a softplus slope of 0.69.
    recurrent = [p for n, p in network.named_parameters() if n != "extra_wear"]
    optimiser = torch.optim.Adam(
        [
            {"params": recurrent, "weight_decay": params.weight_decay},
            {"params": [network.extra_wear], "weight_decay": 0.0},
        ],
        lr=params.lr,
    )
    shuffler = torch.Generator().manual_seed(params.seed)
    curve: list[tuple[int, float, float]] = []
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state = {k: v.clone() for k, v in network.state_dict().items()}
    n = len(fit_part)
    for epoch in range(1, params.epochs + 1):
        network.train()
        order = torch.randperm(n, generator=shuffler)
        total = 0.0
        for start in range(0, n, params.batch):
            rows = order[start : start + params.batch]
            optimiser.zero_grad()
            pred, offsets = _forward(network, train, rows)
            loss = _masked_huber(pred, target[rows], mask[rows])
            if params.offset_trend_penalty > 0:
                loss = loss + params.offset_trend_penalty * _offset_trend(offsets, train, rows)
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimiser.step()
            total += float(loss.item()) * len(rows)
        inner_loss = float("nan")
        if has_inner:
            network.eval()
            with torch.no_grad():
                pred = _forward(network, inner, slice(None))[0]
                inner_loss = float(_masked_huber(pred, inner_target, inner_mask).item())
        curve.append((epoch, total / n, inner_loss))
        if not has_inner:
            best_epoch = epoch
            continue
        if inner_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = inner_loss, epoch, 0
            best_state = {k: v.clone() for k, v in network.state_dict().items()}
        else:
            stale += 1
            if stale >= params.patience:
                break
    if has_inner:
        network.load_state_dict(best_state)
    return tuple(curve), best_epoch


def _scaler_json(scaler: Scaler) -> dict[str, list[float]]:
    return {
        "history_mean": scaler.history_mean.tolist(),
        "history_std": scaler.history_std.tolist(),
        "future_mean": scaler.future_mean.tolist(),
        "future_std": scaler.future_std.tolist(),
    }


def save_lstm(model: TrainedLstm, directory: Path) -> None:
    """Write ``lstm.pt`` (state_dict) and ``lstm.meta.json`` (params, scaler, feature order)."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.network.state_dict(), directory / WEIGHTS_FILE)
    meta = {
        "params": model.params.model_dump(mode="json"),
        "scaler": _scaler_json(model.scaler),
        "tyre_clip": model.tyre_clip.max_age,
        "tyre_min_slope": model.tyre_clip.min_slope,
        "tyre_prior": prior_summary(model.prior),
        "history_features": list(model.history_features),
        "future_features": list(model.future_features),
        "curve": [list(row) for row in model.curve],
        "best_epoch": model.best_epoch,
        "train_seconds": model.train_seconds,
        "n_train_samples": model.n_train_samples,
        "inner_val_seasons": list(model.inner_val_seasons),
    }
    (directory / META_FILE).write_text(json.dumps(meta, indent=2))


class _Meta(BaseModel):
    """Schema of ``lstm.meta.json`` (validated on load)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    params: LstmParams
    scaler: dict[str, list[float]]
    tyre_clip: dict[str, float]
    tyre_min_slope: dict[str, float] = {}
    tyre_prior: dict[str, object]
    history_features: tuple[str, ...]
    future_features: tuple[str, ...]
    curve: tuple[tuple[int, float, float], ...]
    best_epoch: int
    train_seconds: float
    n_train_samples: int
    inner_val_seasons: tuple[int, ...]


def load_lstm(directory: Path) -> TrainedLstm:
    """Rebuild a ``TrainedLstm`` saved by ``save_lstm`` (feature order must match this code)."""
    meta = _Meta.model_validate_json((directory / META_FILE).read_text())
    if meta.history_features != HISTORY_FEATURES or meta.future_features != FUTURE_FEATURES:
        raise ValueError("saved LSTM feature order does not match the current sequence encoding")
    network = LstmPaceNet(len(HISTORY_FEATURES), len(FUTURE_FEATURES), meta.params)
    state: dict[str, torch.Tensor] = torch.load(directory / WEIGHTS_FILE, weights_only=True)
    network.load_state_dict(state)
    network.eval()
    scaler = Scaler(**{k: np.asarray(v, dtype=np.float64) for k, v in meta.scaler.items()})
    return TrainedLstm(
        network=network,
        params=meta.params,
        scaler=scaler,
        tyre_clip=TyreAgeClip(dict(meta.tyre_clip), dict(meta.tyre_min_slope)),
        prior=prior_from_summary(meta.tyre_prior),
        curve=meta.curve,
        best_epoch=meta.best_epoch,
        train_seconds=meta.train_seconds,
        n_train_samples=meta.n_train_samples,
        inner_val_seasons=meta.inner_val_seasons,
    )
