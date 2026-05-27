"""MLP baseline over pooled skeleton-sequence features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.seed import set_global_seed


@dataclass(frozen=True)
class MLPTrainingHistoryRow:
    """One epoch of MLP training diagnostics."""

    epoch: int
    train_loss: float
    val_loss: float | None


class MLPBaseline:
    """Sklearn-style MLP classifier for variable-length skeleton sequences.

    Each sequence is converted to a fixed vector by concatenating mean, max,
    and standard deviation over time. The pooled vectors are then fed to a
    two-hidden-layer PyTorch MLP.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.3,
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
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
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
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.validation_fraction = float(validation_fraction)
        self.patience = int(patience)
        self.seed = int(seed)
        self.device = device
        self.standardize = bool(standardize)

    def fit(self, X_list: Sequence[np.ndarray], y: Sequence[Any]) -> "MLPBaseline":
        """Fit the MLP classifier and return ``self``."""
        torch, nn, optim = _load_torch()
        set_global_seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        features = _pool_sequences(X_list)
        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("X_list and y must contain the same number of samples.")
        if labels.size == 0:
            raise ValueError("Cannot fit MLPBaseline on an empty dataset.")

        classes, encoded_labels = np.unique(labels, return_inverse=True)
        if classes.size < 2:
            raise ValueError("MLPBaseline requires at least two classes.")

        features = self._fit_transform_features(features)
        train_indices, val_indices = _train_val_split(
            encoded_labels,
            validation_fraction=self.validation_fraction,
            seed=self.seed,
        )

        device = torch.device(self.device)
        model = nn.Sequential(
            nn.Linear(features.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, int(classes.size)),
        ).to(device)

        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        label_tensor = torch.as_tensor(encoded_labels, dtype=torch.long, device=device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        best_state = _clone_state_dict(model)
        best_loss = np.inf
        epochs_without_improvement = 0
        history: list[MLPTrainingHistoryRow] = []
        rng = np.random.default_rng(self.seed)

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            shuffled = rng.permutation(train_indices)
            train_losses: list[float] = []
            for batch_indices in _iter_batches(shuffled, self.batch_size):
                batch_tensor = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
                optimizer.zero_grad()
                logits = model(feature_tensor.index_select(0, batch_tensor))
                loss = criterion(logits, label_tensor.index_select(0, batch_tensor))
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(train_losses))
            val_loss = _loss_on_indices(
                model,
                criterion,
                feature_tensor,
                label_tensor,
                val_indices,
                torch=torch,
                device=device,
            )
            monitor_loss = train_loss if val_loss is None else val_loss
            history.append(
                MLPTrainingHistoryRow(
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
        self.n_features_in_ = int(features.shape[1])
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
        torch, _, _ = _load_torch()
        features = self._transform_features(_pool_sequences(X_list))
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected pooled feature dimension {self.n_features_in_}, "
                f"got {features.shape[1]}."
            )

        device = next(self.model_.parameters()).device
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(feature_tensor)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.detach().cpu().numpy()

    def _fit_transform_features(self, features: np.ndarray) -> np.ndarray:
        if not self.standardize:
            self.feature_mean_ = np.zeros(features.shape[1], dtype=np.float64)
            self.feature_scale_ = np.ones(features.shape[1], dtype=np.float64)
            return features.astype(np.float32, copy=False)

        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale == 0.0] = 1.0
        self.feature_mean_ = mean
        self.feature_scale_ = scale
        return self._transform_features(features)

    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.feature_mean_) / self.feature_scale_).astype(
            np.float32,
            copy=False,
        )

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("MLPBaseline must be fitted before prediction.")


def pooled_sequence_features(X_list: Sequence[np.ndarray]) -> np.ndarray:
    """Return mean/max/std pooled features for a collection of sequences."""
    return _pool_sequences(X_list)


def _pool_sequences(X_list: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(sequence, dtype=np.float64) for sequence in X_list]
    if not arrays:
        raise ValueError("X_list must not be empty.")

    feature_dim = arrays[0].shape[1] if arrays[0].ndim == 2 else None
    pooled: list[np.ndarray] = []
    for sequence in arrays:
        if sequence.ndim != 2:
            raise ValueError("Every sequence must be a 2D T x D matrix.")
        if sequence.shape[0] == 0:
            raise ValueError("Every sequence must contain at least one frame.")
        if feature_dim is None or sequence.shape[1] != feature_dim:
            raise ValueError("Every sequence must have the same feature dimension.")
        if not np.isfinite(sequence).all():
            raise ValueError("Sequences must contain only finite values.")
        pooled.append(
            np.concatenate(
                [
                    sequence.mean(axis=0),
                    sequence.max(axis=0),
                    sequence.std(axis=0),
                ]
            )
        )
    return np.stack(pooled, axis=0)


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


def _iter_batches(indices: np.ndarray, batch_size: int) -> Sequence[np.ndarray]:
    return [
        indices[start : start + batch_size]
        for start in range(0, indices.size, batch_size)
    ]


def _loss_on_indices(
    model: Any,
    criterion: Any,
    feature_tensor: Any,
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
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
        logits = model(feature_tensor.index_select(0, index_tensor))
        loss = criterion(logits, label_tensor.index_select(0, index_tensor))
    return float(loss.detach().cpu().item())


def _clone_state_dict(model: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _load_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn, optim
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError(
            "PyTorch is required for MLPBaseline. Install torch to use this baseline."
        ) from exc
    return torch, nn, optim
