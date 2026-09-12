"""
Training script for ForeSite AI GRU Trajectory Model.

Configuration:
- 100% Real Video Tracking Data (output/trajectories.csv & output/frame_tracks.json)
- 65% Train / 35% Validation split
- 45 Epochs
- Seed: 42 (applied across PyTorch, NumPy, Python random)
- Output checkpoints strictly saved within model/ folder
"""

import os
import json
import random
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

try:
    from .trajectory_model import TrajectoryConfig, TrajectoryGRU
except ImportError:
    from trajectory_model import TrajectoryConfig, TrajectoryGRU


def set_seed(seed: int = 42):
    """Sets random seeds for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RealTrajectoryDataset(Dataset):
    """
    Trajectory Dataset strictly built from real CCTV video tracking data.
    Extracts continuous (obs_len -> pred_len) sliding windows from tracked entities.
    """
    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        csv_path: Optional[str] = None,
        json_path: Optional[str] = None,
        stride: int = 1,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.total_len = obs_len + pred_len
        self.stride = stride

        self.samples_obs: List[torch.Tensor] = []
        self.samples_target: List[torch.Tensor] = []
        self.samples_class: List[int] = []

        # 1. Load real trajectory tracks from CSV
        if csv_path and os.path.exists(csv_path):
            self._load_from_csv(csv_path)

        # 2. Load additional real trajectory tracks from frame_tracks.json if available
        if json_path and os.path.exists(json_path) and len(self.samples_obs) == 0:
            self._load_from_frame_json(json_path)

    def _load_from_csv(self, csv_path: str):
        try:
            df = pd.read_csv(csv_path)
            # Group by track_id
            for track_id, group in df.groupby("track_id"):
                group = group.sort_values("timestamp")
                coords = group[["x", "y"]].values
                class_str = group["class_name"].iloc[0] if "class_name" in group.columns else "worker"
                cls_id = 0 if "worker" in class_str.lower() else (1 if "forklift" in class_str.lower() else (2 if "excavator" in class_str.lower() else 3))

                # Extract sliding windows of length total_len
                if len(coords) >= self.total_len:
                    for start_idx in range(0, len(coords) - self.total_len + 1, self.stride):
                        window = coords[start_idx : start_idx + self.total_len]
                        obs = torch.tensor(window[: self.obs_len], dtype=torch.float32)
                        target = torch.tensor(window[self.obs_len :], dtype=torch.float32)
                        self.samples_obs.append(obs)
                        self.samples_target.append(target)
                        self.samples_class.append(cls_id)
            print(f"[Dataset] Loaded {len(self.samples_obs)} real tracklet samples from {csv_path}")
        except Exception as e:
            print(f"[Warning] Failed to load real trajectories from CSV: {e}")

    def _load_from_frame_json(self, json_path: str):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            tracks_by_id: Dict[int, List[Dict[str, Any]]] = {}
            for frame in data:
                for t in frame.get("tracks", []):
                    tid = t["track_id"]
                    if tid not in tracks_by_id:
                        tracks_by_id[tid] = []
                    tracks_by_id[tid].append(t)

            for tid, t_list in tracks_by_id.items():
                t_list.sort(key=lambda x: x["timestamp"])
                coords = np.array([t["position"] for t in t_list], dtype=np.float32)
                class_str = t_list[0].get("class_name", "worker")
                cls_id = 0 if "worker" in class_str.lower() else 1

                if len(coords) >= self.total_len:
                    for start_idx in range(0, len(coords) - self.total_len + 1, self.stride):
                        window = coords[start_idx : start_idx + self.total_len]
                        obs = torch.tensor(window[: self.obs_len], dtype=torch.float32)
                        target = torch.tensor(window[self.obs_len :], dtype=torch.float32)
                        self.samples_obs.append(obs)
                        self.samples_target.append(target)
                        self.samples_class.append(cls_id)
            print(f"[Dataset] Loaded {len(self.samples_obs)} real tracklet samples from {json_path}")
        except Exception as e:
            print(f"[Warning] Failed to load real trajectories from JSON: {e}")

    def __len__(self) -> int:
        return len(self.samples_obs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.samples_obs[idx],
            self.samples_target[idx],
            torch.tensor(self.samples_class[idx], dtype=torch.long),
        )


def compute_ade_fde(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """Average Displacement Error and Final Displacement Error."""
    distances = torch.norm(preds - targets, p=2, dim=-1)  # (batch, pred_len)
    ade = float(torch.mean(distances).item())
    fde = float(torch.mean(distances[:, -1]).item())
    return ade, fde


def train():
    cfg = TrajectoryConfig()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device} | Seed: {cfg.seed}")

    # 1. Dataset setup strictly with real video tracking data
    model_dir = Path(__file__).parent
    real_csv = model_dir.parent / "output" / "trajectories.csv"
    real_json = model_dir.parent / "output" / "frame_tracks.json"

    dataset = RealTrajectoryDataset(
        obs_len=cfg.obs_len,
        pred_len=cfg.pred_len,
        csv_path=str(real_csv) if real_csv.exists() else None,
        json_path=str(real_json) if real_json.exists() else None,
        stride=1,
    )

    if len(dataset) == 0:
        raise RuntimeError(f"No real trajectory tracks found in {real_csv} or {real_json}.")

    # 2. 65% Train / 35% Validation Split
    total_size = len(dataset)
    train_size = int(total_size * cfg.train_split)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    print(f"[*] Total Real Tracking Samples: {total_size} | Train (65%): {len(train_ds)} | Val (35%): {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=min(cfg.batch_size, len(train_ds)), shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=min(cfg.batch_size, len(val_ds)), shuffle=False)

    # 3. Model, Loss, Optimizer, Scheduler
    model = TrajectoryGRU(cfg).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)

    checkpoints_dir = model_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = checkpoints_dir / "gru_trajectory_best.pt"

    history = {
        "dataset_source": "real_cctv_video_tracks",
        "total_samples": total_size,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_loss": [],
        "val_loss": [],
        "val_ade": [],
        "val_fde": [],
        "epochs": cfg.epochs,
        "seed": cfg.seed,
        "train_split": cfg.train_split,
    }

    best_val_ade = float("inf")
    start_time = time.time()

    print(f"[*] Launching training for {cfg.epochs} epochs on 100% real tracking data...")
    print("-" * 75)

    for epoch in range(1, cfg.epochs + 1):
        # Training loop
        model.train()
        train_loss = 0.0
        for obs, targets, cls_ids in train_loader:
            obs, targets, cls_ids = obs.to(device), targets.to(device), cls_ids.to(device)
            optimizer.zero_grad()
            preds = model(obs, cls_ids)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * obs.size(0)

        train_loss /= len(train_ds)
        scheduler.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_ade_sum = 0.0
        val_fde_sum = 0.0

        with torch.no_grad():
            for obs, targets, cls_ids in val_loader:
                obs, targets, cls_ids = obs.to(device), targets.to(device), cls_ids.to(device)
                preds = model(obs, cls_ids)
                loss = criterion(preds, targets)
                val_loss += loss.item() * obs.size(0)

                ade, fde = compute_ade_fde(preds, targets)
                val_ade_sum += ade * obs.size(0)
                val_fde_sum += fde * obs.size(0)

        val_loss /= len(val_ds)
        val_ade = val_ade_sum / len(val_ds)
        val_fde = val_fde_sum / len(val_ds)

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_ade"].append(round(val_ade, 4))
        history["val_fde"].append(round(val_fde, 4))

        # Save best model
        is_best = ""
        if val_ade < best_val_ade:
            best_val_ade = val_ade
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "best_val_ade": best_val_ade,
            }, best_weights_path)
            is_best = " [BEST SAVED]"

        if epoch % 5 == 0 or epoch == 1 or epoch == cfg.epochs:
            print(
                f"Epoch [{epoch:02d}/{cfg.epochs}] | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val ADE: {val_ade:.4f} | "
                f"Val FDE: {val_fde:.4f}{is_best}"
            )

    elapsed = time.time() - start_time
    print("-" * 75)
    print(f"[*] Training finished in {elapsed:.1f}s. Best Val ADE: {best_val_ade:.4f}")
    print(f"[*] Best weights saved to: {best_weights_path}")

    # Save metrics log inside model/
    metrics_path = model_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[*] Metrics summary saved to: {metrics_path}")


if __name__ == "__main__":
    train()

