"""
Trajectory Prediction Model (GRU) for ForeSite AI.

Sequence-to-Sequence recurrent architecture with:
1. Spatial coordinate linear projection
2. Categorical entity class embedding (worker vs machinery)
3. 2-layer GRU encoder with recurrent dropout
4. Autoregressive GRU decoder with linear output head
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn


@dataclass
class TrajectoryConfig:
    input_dim: int = 2          # (x, y) 2D coordinates
    hidden_dim: int = 128       # GRU hidden state dimension
    num_layers: int = 2         # Stacked GRU layers
    num_classes: int = 4        # 0: worker, 1: forklift, 2: excavator, 3: truck
    class_embed_dim: int = 16   # Entity class embedding dimension
    obs_len: int = 8            # Number of observed historical timesteps
    pred_len: int = 12          # Number of forecasted future timesteps
    dropout: float = 0.2        # Dropout probability
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 45            # User requirement: 45 epochs
    batch_size: int = 32
    train_split: float = 0.65   # User requirement: 65% train / 35% validation
    seed: int = 42              # User requirement: deterministic seed


class TrajectoryGRU(nn.Module):
    """
    Encoder-Decoder GRU trajectory forecaster.
    Conditioned on categorical class embeddings so workers and vehicles
    share weights while learning distinct physical momentum dynamics.
    """
    def __init__(self, config: Optional[TrajectoryConfig] = None):
        super().__init__()
        self.config = config or TrajectoryConfig()

        # 1. Feature Embedding Layers
        self.coord_embed = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, self.config.hidden_dim // 2),
        )
        self.class_embed = nn.Embedding(self.config.num_classes, self.config.class_embed_dim)

        # Total encoder input = coordinate features + class embedding
        encoder_input_dim = (self.config.hidden_dim // 2) + self.config.class_embed_dim

        # 2. Recurrent Encoder
        self.encoder = nn.GRU(
            input_size=encoder_input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )

        # 3. Autoregressive Decoder
        decoder_input_dim = self.config.input_dim + self.config.class_embed_dim
        self.decoder_cell = nn.GRUCell(
            input_size=decoder_input_dim,
            hidden_size=self.config.hidden_dim,
        )

        # 4. Output Projection Head -> (delta_x, delta_y)
        self.output_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, self.config.input_dim),
        )

    def forward(
        self,
        obs_traj: torch.Tensor,
        entity_class: torch.Tensor,
        future_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            obs_traj: Tensor of shape (batch, obs_len, 2)
            entity_class: LongTensor of shape (batch,)
            future_steps: Int horizon (defaults to config.pred_len)

        Returns:
            predicted_traj: Tensor of shape (batch, pred_len, 2)
        """
        batch_size = obs_traj.size(0)
        pred_len = future_steps or self.config.pred_len
        device = obs_traj.device

        # Class embedding: (batch, 1, class_embed_dim) repeated along time
        cls_emb = self.class_embed(entity_class)  # (batch, class_embed_dim)
        cls_emb_seq = cls_emb.unsqueeze(1).repeat(1, self.config.obs_len, 1)

        # Coordinate embedding
        coord_emb = self.coord_embed(obs_traj)  # (batch, obs_len, hidden_dim // 2)

        # Concat embeddings
        enc_in = torch.cat([coord_emb, cls_emb_seq], dim=-1)

        # Run Encoder
        _, hidden = self.encoder(enc_in)  # hidden: (num_layers, batch, hidden_dim)
        h_t = hidden[-1]  # Take top layer state: (batch, hidden_dim)

        # Autoregressive decoding
        # Start decoding from the last observed point
        curr_pos = obs_traj[:, -1, :]  # (batch, 2)
        predictions = []

        for _ in range(pred_len):
            # Input to decoder cell is current position + class embedding
            dec_in = torch.cat([curr_pos, cls_emb], dim=-1)  # (batch, 2 + class_embed_dim)
            h_t = self.decoder_cell(dec_in, h_t)
            displacement = self.output_head(h_t)             # (batch, 2)
            curr_pos = curr_pos + displacement
            predictions.append(curr_pos.unsqueeze(1))

        # Shape: (batch, pred_len, 2)
        return torch.cat(predictions, dim=1)
