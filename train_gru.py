"""
Training Pipeline for Construction Site GRU Trajectory Prediction Model.

This script:
1. Generates/processes construction site trajectory datasets (workers & forklifts).
2. Sets up PyTorch Dataset and DataLoader with dynamic spatial normalization.
3. Trains TrajectoryGRU with AdamW optimizer, Cosine Annealing scheduler, and Smooth L1 Loss.
4. Evaluates Average Displacement Error (ADE) and Final Displacement Error (FDE).
5. Saves model checkpoint to models/gru_trajectory.pt.
"""

import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from prediction.trajectory_model import TrajectoryConfig, TrajectoryGRU


class ConstructionSiteTrajectoryDataset(Dataset):
    """
    Dataset generating dynamic trajectory sequences for construction site safety monitoring.
    Simulates realistic 2D motion dynamics for Workers (0) and Forklifts (1).
    """
    def __init__(
        self,
        num_samples: int = 5000,
        obs_len: int = 8,
        pred_len: int = 12,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.total_len = obs_len + pred_len

        np.random.seed(seed)
        self.obs_data, self.target_data, self.entity_ids = self._generate_trajectories()

    def _generate_trajectories(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_list = []
        target_list = []
        entity_list = []

        for _ in range(self.num_samples):
            # 50% Workers (0), 50% Forklifts (1)
            entity_id = np.random.choice([0, 1])
            
            # Initial position (normalized 0-1000px canvas)
            start_x = np.random.uniform(50.0, 950.0)
            start_y = np.random.uniform(50.0, 950.0)

            # Velocity profile based on entity class
            if entity_id == 0:  # Worker
                speed = np.random.uniform(2.0, 5.0)
                turn_std = 0.15
            else:               # Forklift
                speed = np.random.uniform(8.0, 18.0)
                turn_std = 0.08

            # Initial heading angle
            heading = np.random.uniform(0, 2 * np.pi)

            # Generate trajectory points
            coords = []
            curr_x, curr_y = start_x, start_y
            curr_heading = heading

            for t in range(self.total_len):
                coords.append([curr_x, curr_y])
                # Small turn angle noise over time
                curr_heading += np.random.normal(0, turn_std)
                # Small speed variation
                step_speed = speed * (1.0 + np.random.normal(0, 0.05))
                curr_x += step_speed * np.cos(curr_heading)
                curr_y += step_speed * np.sin(curr_heading)

            coords_tensor = torch.tensor(coords, dtype=torch.float32)
            obs_list.append(coords_tensor[:self.obs_len])
            target_list.append(coords_tensor[self.obs_len:])
            entity_list.append(entity_id)

        obs_tensor = torch.stack(obs_list, dim=0)       # (num_samples, obs_len, 2)
        target_tensor = torch.stack(target_list, dim=0)   # (num_samples, pred_len, 2)
        entity_tensor = torch.tensor(entity_list, dtype=torch.long)  # (num_samples,)

        return obs_tensor, target_tensor, entity_tensor

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.obs_data[idx], self.target_data[idx], self.entity_ids[idx]


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """
    Computes Average Displacement Error (ADE) and Final Displacement Error (FDE).

    Args:
        preds: (batch, pred_len, 2)
        targets: (batch, pred_len, 2)

    Returns:
        (ade, fde)
    """
    # Euclidean distance per step: (batch, pred_len)
    disp = torch.norm(preds - targets, p=2, dim=-1)
    
    # ADE: Mean distance over all time steps and batch
    ade = float(torch.mean(disp).item())

    # FDE: Mean distance at final time step T_pred
    fde = float(torch.mean(disp[:, -1]).item())

    return ade, fde


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0

    for obs, targets, entity_ids in dataloader:
        obs = obs.to(device)
        targets = targets.to(device)
        entity_ids = entity_ids.to(device)

        optimizer.zero_grad()
        preds = model(obs, entity_ids, target_positions=targets)

        loss = criterion(preds, targets)
        loss.backward()

        # Gradient clipping for GRU stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        ade, fde = compute_metrics(preds, targets)

        total_loss += loss.item() * obs.size(0)
        total_ade += ade * obs.size(0)
        total_fde += fde * obs.size(0)

    n = len(dataloader.dataset)
    return total_loss / n, total_ade / n, total_fde / n


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0

    with torch.no_grad():
        for obs, targets, entity_ids in dataloader:
            obs = obs.to(device)
            targets = targets.to(device)
            entity_ids = entity_ids.to(device)

            preds = model(obs, entity_ids)
            loss = criterion(preds, targets)

            ade, fde = compute_metrics(preds, targets)

            total_loss += loss.item() * obs.size(0)
            total_ade += ade * obs.size(0)
            total_fde += fde * obs.size(0)

    n = len(dataloader.dataset)
    return total_loss / n, total_ade / n, total_fde / n


def main():
    parser = argparse.ArgumentParser(description="Train Trajectory GRU for Construction Safety")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--obs-len", type=int, default=8, help="Observed sequence length")
    parser.add_argument("--pred-len", type=int, default=12, help="Predicted sequence length")
    parser.add_argument("--num-samples", type=int, default=6000, help="Dataset trajectory sample count")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build Trajectory Configuration
    config = TrajectoryConfig(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        learning_rate=args.lr,
    )

    # Prepare datasets
    train_dataset = ConstructionSiteTrajectoryDataset(
        num_samples=int(args.num_samples * 0.8),
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        seed=42,
    )
    val_dataset = ConstructionSiteTrajectoryDataset(
        num_samples=int(args.num_samples * 0.2),
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        seed=100,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = TrajectoryGRU(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.SmoothL1Loss()

    print(f"Model architecture initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
    print(f"Training set: {len(train_dataset)} trajectories | Val set: {len(val_dataset)} trajectories")
    print("-" * 75)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_ade = float("inf")
    checkpoint_path = os.path.join(args.save_dir, "gru_trajectory.pt")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_ade, train_fde = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_ade, val_fde = validate(model, val_loader, criterion, device)
        scheduler.step()

        is_best = val_ade < best_val_ade
        if is_best:
            best_val_ade = val_ade
            torch.save({
                "epoch": epoch,
                "config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_ade": val_ade,
                "val_fde": val_fde,
            }, checkpoint_path)

        star_str = "⭐ (Best)" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | Train ADE: {train_ade:.2f}px | "
            f"Val ADE: {val_ade:.2f}px | Val FDE: {val_fde:.2f}px {star_str}"
        )

    elapsed = time.time() - start_time
    print("-" * 75)
    print(f"Training completed in {elapsed:.2f} seconds.")
    print(f"Best Validation ADE: {best_val_ade:.2f} px")
    print(f"Saved model checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
