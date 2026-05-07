"""PyTorch MLP baseline for cross-sectional return prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features.preprocessing import drop_model_missing
from src.models.evaluate import evaluate_predictions
from src.models.train_tree_models import split_by_time, time_series_split_dates
from src.utils.io import read_parquet, resolve_path, save_csv, save_parquet


class AlphaMLP(nn.Module):
    """Simple feed-forward neural network for alpha prediction."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted forward return."""
        return self.net(x).squeeze(-1)


def _make_loader(df: pd.DataFrame, factor_cols: list[str], label_col: str, batch_size: int, shuffle: bool) -> DataLoader:
    x = torch.tensor(df[factor_cols].to_numpy(dtype=np.float32))
    y = torch.tensor(df[label_col].to_numpy(dtype=np.float32))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def train_mlp_model(
    df: pd.DataFrame,
    factor_cols: list[str],
    label_col: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train the MLP baseline with validation early stopping."""
    model_df = drop_model_missing(df, factor_cols, label_col)
    boundaries = time_series_split_dates(
        model_df,
        float(config["validation"]["train_ratio"]),
        float(config["validation"]["val_ratio"]),
    )
    train, val, test = split_by_time(model_df, boundaries)
    mlp_cfg = config["models"]["mlp"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config["project"].get("seed", 42)))

    model = AlphaMLP(len(factor_cols), list(mlp_cfg["hidden_dims"]), float(mlp_cfg["dropout"])).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(mlp_cfg["learning_rate"]),
        weight_decay=float(mlp_cfg["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    train_loader = _make_loader(train, factor_cols, label_col, int(mlp_cfg["batch_size"]), shuffle=True)
    val_loader = _make_loader(val, factor_cols, label_col, int(mlp_cfg["batch_size"]), shuffle=False)

    best_loss = float("inf")
    best_state = None
    patience = int(mlp_cfg["patience"])
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(int(mlp_cfg["max_epochs"])):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(float(loss_fn(model(xb), yb).item()))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_dir = resolve_path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "factor_cols": factor_cols, "config": mlp_cfg}, model_dir / "mlp.pt")

    x_test = torch.tensor(test[factor_cols].to_numpy(dtype=np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x_test).detach().cpu().numpy()

    predictions = test[["date", "code", label_col]].copy()
    predictions["score_mlp"] = pred
    metrics = evaluate_predictions(predictions, label_col, ["score_mlp"])
    save_csv(pd.DataFrame(history), Path(config["paths"]["tables_dir"]) / "mlp_training_history.csv")
    save_csv(metrics, Path(config["paths"]["tables_dir"]) / "prediction_metrics_mlp.csv")

    pred_path = resolve_path(config["paths"]["predictions"])
    if pred_path.exists():
        existing = read_parquet(config["paths"]["predictions"])
        merged = existing.merge(predictions[["date", "code", "score_mlp"]], on=["date", "code"], how="outer")
        if label_col not in merged.columns and label_col in predictions.columns:
            merged = merged.merge(predictions[["date", "code", label_col]], on=["date", "code"], how="left")
        save_parquet(merged, config["paths"]["predictions"])
    else:
        save_parquet(predictions, config["paths"]["predictions"])
    return predictions, metrics
