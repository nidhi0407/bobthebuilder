"""
Trajectory Prediction GRU Model with Dynamic Spatial-Kinematic Embeddings.

This module defines:
1. TrajectoryConfig: Dataclass for dynamic model hyperparameter management.
2. SpatialKinematicEmbedding: Feature embedding network for spatial coordinates, dynamic velocities, accelerations, and entity category classes.
3. TrajectoryGRU: Recurrent encoder-decoder architecture for multi-step trajectory forecasting.
4. predict_trajectory: Standalone inference helper function.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrajectoryConfig:
    """Hyperparameters and architecture configuration for Trajectory GRU."""
    obs_len: int = 8               # Historical frames observed (t-obs_len to t)
    pred_len: int = 12             # Future frames predicted (t+1 to t+pred_len)
    num_entities: int = 2          # Entity categories (e.g. 0: Worker, 1: Forklift)
    entity_embed_dim: int = 16     # Category embedding dimension
    kinematic_dim: int = 6         # [x, y, vx, vy, ax, ay]
    spatial_embed_dim: int = 48    # Kinematic spatial embedding dimension
    hidden_dim: int = 64           # GRU hidden state dimension
    num_layers: int = 2            # Number of GRU layers
    dropout: float = 0.1           # Dropout probability
    learning_rate: float = 1e-3    # Training learning rate
    weight_decay: float = 1e-4     # Optimizer weight decay

    @property
    def embedding_dim(self) -> int:
        return self.spatial_embed_dim + self.entity_embed_dim


def compute_kinematic_features(positions: torch.Tensor) -> torch.Tensor:
    """
    Dynamically computes position (x, y), velocity (vx, vy), and acceleration (ax, ay)
    across the time dimension of a trajectory tensor.

    Args:
        positions: Tensor of shape (batch, seq_len, 2) containing (x, y) coordinates.

    Returns:
        Tensor of shape (batch, seq_len, 6) containing (x, y, vx, vy, ax, ay).
    """
    # Velocity: first derivative wrt time (padded at t=0)
    velocities = torch.zeros_like(positions)
    velocities[:, 1:, :] = positions[:, 1:, :] - positions[:, :-1, :]
    velocities[:, 0, :] = velocities[:, 1, :] if positions.size(1) > 1 else 0.0

    # Acceleration: second derivative wrt time (padded at t=0,1)
    accelerations = torch.zeros_like(velocities)
    accelerations[:, 1:, :] = velocities[:, 1:, :] - velocities[:, :-1, :]
    accelerations[:, 0, :] = accelerations[:, 1, :] if velocities.size(1) > 1 else 0.0

    # Concatenate [x, y, vx, vy, ax, ay]
    kinematic_features = torch.cat([positions, velocities, accelerations], dim=-1)
    return kinematic_features


class SpatialKinematicEmbedding(nn.Module):
    """
    Projects dynamic kinematic state features (positions, velocities, accelerations)
    and categorical entity IDs into a continuous embedding space.
    """
    def __init__(self, config: TrajectoryConfig):
        super().__init__()
        self.config = config

        self.kinematic_mlp = nn.Sequential(
            nn.Linear(config.kinematic_dim, config.spatial_embed_dim),
            nn.LayerNorm(config.spatial_embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.spatial_embed_dim, config.spatial_embed_dim),
            nn.LayerNorm(config.spatial_embed_dim),
        )

        self.entity_embedder = nn.Embedding(config.num_entities, config.entity_embed_dim)

    def forward(self, kinematic_features: torch.Tensor, entity_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kinematic_features: (batch, seq_len, kinematic_dim)
            entity_ids: (batch,) or (batch, seq_len) integer category IDs.

        Returns:
            Tensor of shape (batch, seq_len, embedding_dim)
        """
        spatial_emb = self.kinematic_mlp(kinematic_features)  # (batch, seq_len, spatial_embed_dim)

        if entity_ids.dim() == 1:
            entity_ids = entity_ids.unsqueeze(1).expand(-1, kinematic_features.size(1))

        entity_emb = self.entity_embedder(entity_ids)          # (batch, seq_len, entity_embed_dim)

        # Concatenate spatial and entity embeddings
        combined_emb = torch.cat([spatial_emb, entity_emb], dim=-1)
        return combined_emb


class TrajectoryGRU(nn.Module):
    """
    GRU Encoder-Decoder Network for Trajectory Prediction.
    Encodes observed historical trajectory sequences and autoregressively forecasts
    future waypoints.
    """
    def __init__(self, config: Optional[TrajectoryConfig] = None):
        super().__init__()
        self.config = config or TrajectoryConfig()

        # Spatial Kinematic Embedding Layer
        self.embedding = SpatialKinematicEmbedding(self.config)

        # Encoder GRU
        self.encoder = nn.GRU(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )

        # Decoder GRU Cell (takes previous relative step + entity embedding)
        decoder_input_dim = 2 + self.config.entity_embed_dim
        self.decoder_cell = nn.GRUCell(
            input_size=decoder_input_dim,
            hidden_size=self.config.hidden_dim,
        )

        # Prediction Head for position offset (dx, dy)
        self.predictor = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def forward(
        self,
        obs_positions: torch.Tensor,
        entity_ids: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            obs_positions: (batch, obs_len, 2) - Observed past (x, y) coordinates
            entity_ids: (batch,) - Entity class indices (0=Worker, 1=Forklift)
            target_positions: Optional (batch, pred_len, 2) for teacher forcing during training

        Returns:
            predicted_trajectories: (batch, pred_len, 2) - Predicted future (x, y) waypoints
        """
        batch_size = obs_positions.size(0)

        # Compute dynamic kinematic features: [x, y, vx, vy, ax, ay]
        kinematic_feats = compute_kinematic_features(obs_positions)

        # Embed input sequence
        embedded_obs = self.embedding(kinematic_feats, entity_ids)  # (batch, obs_len, embedding_dim)

        # Encode historical trajectory
        _, hidden = self.encoder(embedded_obs)  # hidden: (num_layers, batch, hidden_dim)

        # Use top-layer hidden state for decoder initialization
        decoder_hidden = hidden[-1]  # (batch, hidden_dim)

        # Entity embedding for decoder step inputs
        entity_emb_single = self.embedding.entity_embedder(entity_ids)  # (batch, entity_embed_dim)

        # Initial decoder input is the last velocity (dx, dy) step from observation
        if obs_positions.size(1) > 1:
            last_step = obs_positions[:, -1, :] - obs_positions[:, -2, :]
        else:
            last_step = torch.zeros(batch_size, 2, device=obs_positions.device)

        curr_pos = obs_positions[:, -1, :]  # (batch, 2)
        predicted_waypoints = []

        for t in range(self.config.pred_len):
            # Concatenate step displacement with entity embedding
            cell_input = torch.cat([last_step, entity_emb_single], dim=-1)

            # Update GRU cell state
            decoder_hidden = self.decoder_cell(cell_input, decoder_hidden)

            # Predict displacement offset (dx, dy)
            delta_pos = self.predictor(decoder_hidden)

            # Update current position forecast
            curr_pos = curr_pos + delta_pos
            predicted_waypoints.append(curr_pos.unsqueeze(1))

            # Prepare step for next autoregressive iteration
            if self.training and target_positions is not None:
                # Teacher forcing option during training
                last_step = delta_pos
            else:
                last_step = delta_pos

        # Stack into (batch, pred_len, 2)
        predicted_trajectories = torch.cat(predicted_waypoints, dim=1)
        return predicted_trajectories


def predict_trajectory(
    history_coords: torch.Tensor,
    entity_id: int,
    model: TrajectoryGRU,
    config: Optional[TrajectoryConfig] = None,
) -> torch.Tensor:
    """
    Inference helper function for single trajectory sequence prediction.

    Args:
        history_coords: (obs_len, 2) or (batch, obs_len, 2) tensor of coordinates
        entity_id: integer class ID (0 for Worker, 1 for Forklift)
        model: Trained TrajectoryGRU instance
        config: TrajectoryConfig instance

    Returns:
        forecast: (pred_len, 2) predicted future coordinates tensor
    """
    model.eval()
    device = next(model.parameters()).device

    if history_coords.dim() == 2:
        history_tensor = history_coords.unsqueeze(0).to(device)
    else:
        history_tensor = history_coords.to(device)

    entity_tensor = torch.tensor([entity_id], dtype=torch.long, device=device)

    with torch.no_grad():
        preds = model(history_tensor, entity_tensor)

    return preds.squeeze(0)
