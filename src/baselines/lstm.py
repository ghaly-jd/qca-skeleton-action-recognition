"""LSTM baseline for variable-length skeleton sequences."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.seed import set_global_seed


@dataclass(frozen=True)
class LSTMTrainingHistoryRow:
    """One epoch of LSTM training diagnostics."""

    epoch: int
    train_loss: float
    val_loss: float | None


class LSTMBaseline:
    """Sklearn-style single-layer LSTM classifier for skeleton sequences."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        num_layers: int = 1,
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
        batch_size: int = 32,
        validation_fraction: float = 0.1,
        patience: int = 10,
        seed: int = 0,
        device: str = "cpu",
        standardize: bool = True,
    ) -> None:
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")
        if patience <= 0:
            raise ValueError("patience must be positive.")

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.validation_fraction = float(validation_fraction)
        self.patience = int(patience)
        self.seed = int(seed)
        self.device = device
        self.standardize = bool(standardize)

    def fit(self, X_list: Sequence[np.ndarray], y: Sequence[Any]) -> "LSTMBaseline":
        """Fit the LSTM classifier and return ``self``."""
        torch, nn, optim, _ = _load_torch()
        set_global_seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        sequences, lengths = _validate_sequences(X_list)
        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if len(sequences) != labels.shape[0]:
            raise ValueError("X_list and y must contain the same number of samples.")
        if labels.size == 0:
            raise ValueError("Cannot fit LSTMBaseline on an empty dataset.")

        classes, encoded_labels = np.unique(labels, return_inverse=True)
        if classes.size < 2:
            raise ValueError("LSTMBaseline requires at least two classes.")

        sequences = self._fit_transform_sequences(sequences)
        train_indices, val_indices = _train_val_split(
            encoded_labels,
            validation_fraction=self.validation_fraction,
            seed=self.seed,
        )

        device = torch.device(self.device)
        model = _LSTMSequenceClassifier(
            input_dim=sequences[0].shape[1],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            n_classes=int(classes.size),
            nn=nn,
        ).to(device)

        label_tensor = torch.as_tensor(encoded_labels, dtype=torch.long, device=device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        best_state = _clone_state_dict(model)
        best_loss = np.inf
        epochs_without_improvement = 0
        history: list[LSTMTrainingHistoryRow] = []
        rng = np.random.default_rng(self.seed)

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            shuffled = rng.permutation(train_indices)
            train_losses: list[float] = []
            for batch_indices in _iter_batches(shuffled, self.batch_size):
                batch_values, batch_lengths = _batch_to_tensors(
                    sequences,
                    lengths,
                    batch_indices,
                    torch=torch,
                    device=device,
                )
                batch_labels = label_tensor.index_select(
                    0,
                    torch.as_tensor(batch_indices, dtype=torch.long, device=device),
                )
                optimizer.zero_grad()
                logits = model(batch_values, batch_lengths)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(train_losses))
            val_loss = _loss_on_indices(
                model,
                criterion,
                sequences,
                lengths,
                label_tensor,
                val_indices,
                torch=torch,
                device=device,
            )
            monitor_loss = train_loss if val_loss is None else val_loss
            history.append(
                LSTMTrainingHistoryRow(
                    epoch=int(epoch),
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
            )

            if monitor_loss < best_loss - 1e-8:
                best_loss = monitor_loss
                best_state = _clone_state_dict(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if val_indices.size > 0 and epochs_without_improvement >= self.patience:
                    break

        model.load_state_dict(best_state)
        self.model_ = model
        self.classes_ = classes
        self.n_features_in_ = int(sequences[0].shape[1])
        self.training_history_ = history
        return self

    def predict(self, X_list: Sequence[np.ndarray]) -> np.ndarray:
        """Predict labels for a sequence collection."""
        probabilities = self.predict_proba(X_list)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, X_list: Sequence[np.ndarray]) -> np.ndarray:
        """Return class probabilities for a sequence collection."""
        self._check_is_fitted()
        torch, _, _, _ = _load_torch()
        sequences, lengths = _validate_sequences(X_list)
        if sequences[0].shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected feature dimension {self.n_features_in_}, "
                f"got {sequences[0].shape[1]}."
            )
        sequences = self._transform_sequences(sequences)

        device = next(self.model_.parameters()).device
        indices = np.arange(len(sequences), dtype=np.int64)
        batch_values, batch_lengths = _batch_to_tensors(
            sequences,
            lengths,
            indices,
            torch=torch,
            device=device,
        )
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(batch_values, batch_lengths)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.detach().cpu().numpy()

    def _fit_transform_sequences(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        if not self.standardize:
            self.feature_mean_ = np.zeros(sequences[0].shape[1], dtype=np.float64)
            self.feature_scale_ = np.ones(sequences[0].shape[1], dtype=np.float64)
            return [sequence.astype(np.float32, copy=False) for sequence in sequences]

        stacked = np.concatenate(sequences, axis=0)
        mean = stacked.mean(axis=0)
        scale = stacked.std(axis=0)
        scale[scale == 0.0] = 1.0
        self.feature_mean_ = mean
        self.feature_scale_ = scale
        return self._transform_sequences(sequences)

    def _transform_sequences(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        return [
            ((sequence - self.feature_mean_) / self.feature_scale_).astype(
                np.float32,
                copy=False,
            )
            for sequence in sequences
        ]

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("LSTMBaseline must be fitted before prediction.")


class _LSTMSequenceClassifier:
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        n_classes: int,
        nn: Any,
    ) -> None:
        self._nn = nn
        self.module = nn.Module()
        self.module.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.module.classifier = nn.Linear(hidden_dim, n_classes)

    def __call__(self, padded: Any, lengths: Any) -> Any:
        return self.forward(padded, lengths)

    def forward(self, padded: Any, lengths: Any) -> Any:
        _, _, _, rnn_utils = _load_torch()
        packed = rnn_utils.pack_padded_sequence(
            padded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.module.lstm(packed)
        return self.module.classifier(hidden[-1])

    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()

    def parameters(self) -> Any:
        return self.module.parameters()

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.module.load_state_dict(state_dict)

    def to(self, device: Any) -> "_LSTMSequenceClassifier":
        self.module.to(device)
        return self


def _validate_sequences(X_list: Sequence[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    sequences = [np.asarray(sequence, dtype=np.float64) for sequence in X_list]
    if not sequences:
        raise ValueError("X_list must not be empty.")

    feature_dim = sequences[0].shape[1] if sequences[0].ndim == 2 else None
    lengths: list[int] = []
    for sequence in sequences:
        if sequence.ndim != 2:
            raise ValueError("Every sequence must be a 2D T x D matrix.")
        if sequence.shape[0] == 0:
            raise ValueError("Every sequence must contain at least one frame.")
        if feature_dim is None or sequence.shape[1] != feature_dim:
            raise ValueError("Every sequence must have the same feature dimension.")
        if not np.isfinite(sequence).all():
            raise ValueError("Sequences must contain only finite values.")
        lengths.append(int(sequence.shape[0]))
    return sequences, np.asarray(lengths, dtype=np.int64)


def _train_val_split(
    labels: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(labels.shape[0])
    all_indices = np.arange(n_samples, dtype=np.int64)
    if validation_fraction <= 0.0 or n_samples < 10:
        return all_indices, np.asarray([], dtype=np.int64)

    n_val = int(round(n_samples * validation_fraction))
    n_classes = int(np.unique(labels).size)
    if n_val < n_classes or n_samples - n_val < n_classes:
        return all_indices, np.asarray([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    val_indices: list[int] = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        if class_indices.size < 2:
            return all_indices, np.asarray([], dtype=np.int64)
        take = max(1, int(round(class_indices.size * validation_fraction)))
        val_indices.extend(rng.choice(class_indices, size=take, replace=False).tolist())

    val_array = np.asarray(sorted(val_indices), dtype=np.int64)
    train_array = np.setdiff1d(all_indices, val_array, assume_unique=True)
    if train_array.size == 0:
        return all_indices, np.asarray([], dtype=np.int64)
    return train_array, val_array


def _iter_batches(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [
        indices[start : start + batch_size]
        for start in range(0, indices.size, batch_size)
    ]


def _batch_to_tensors(
    sequences: Sequence[np.ndarray],
    lengths: np.ndarray,
    indices: np.ndarray,
    *,
    torch: Any,
    device: Any,
) -> tuple[Any, Any]:
    selected = [sequences[int(index)] for index in indices]
    selected_lengths = lengths[indices]
    max_length = int(selected_lengths.max())
    feature_dim = int(selected[0].shape[1])
    padded = np.zeros((len(selected), max_length, feature_dim), dtype=np.float32)
    for row, sequence in enumerate(selected):
        padded[row, : sequence.shape[0], :] = sequence
    return (
        torch.as_tensor(padded, dtype=torch.float32, device=device),
        torch.as_tensor(selected_lengths, dtype=torch.long, device=device),
    )


def _loss_on_indices(
    model: _LSTMSequenceClassifier,
    criterion: Any,
    sequences: Sequence[np.ndarray],
    lengths: np.ndarray,
    label_tensor: Any,
    indices: np.ndarray,
    *,
    torch: Any,
    device: Any,
) -> float | None:
    if indices.size == 0:
        return None
    model.eval()
    with torch.no_grad():
        batch_values, batch_lengths = _batch_to_tensors(
            sequences,
            lengths,
            indices,
            torch=torch,
            device=device,
        )
        batch_labels = label_tensor.index_select(
            0,
            torch.as_tensor(indices, dtype=torch.long, device=device),
        )
        loss = criterion(model(batch_values, batch_lengths), batch_labels)
    return float(loss.detach().cpu().item())


def _clone_state_dict(model: _LSTMSequenceClassifier) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _load_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch import nn, optim
        from torch.nn.utils import rnn as rnn_utils
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError(
            "PyTorch is required for LSTMBaseline. Install torch to use this baseline."
        ) from exc
    return torch, nn, optim, rnn_utils
