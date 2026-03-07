import copy
from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn


class Actor(nn.Module, ABC):
    """Actor interface for deterministic policies.""" 
    def __init__(
            self,
            obs_shape: Tuple[int, ...],
            action_dim: int,
            action_scale: float=1.0
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got: {action_dim}") 
        if len(obs_shape) == 0:
            raise ValueError(f"obs_dim must be non-empty, got: {obs_shape}")

        self.obs_shape = tuple(int(element) for element in obs_shape)
        self.action_dim = action_dim
        self.action_scale = action_scale

    @abstractmethod
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Returns actions with shape [B, action_dim].""" 
        raise NotImplementedError

    @torch.inference_mode()
    def predict(self, s: np.ndarray | torch.Tensor) -> np.ndarray:
        device = next(self.parameters()).device 
        if isinstance(s, np.ndarray):
            s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
        else:
            s_t = s.to(device)

        if s_t.dim() == len(self.obs_shape):
            s_t = s_t.unsqueeze(0)
        a_t = self(s_t).detach().cpu().numpy().flatten()
        return self.action_scale * a_t

    def copy(self) -> 'Actor':
        return copy.deepcopy(self)


class ActorMLP(Actor):
        """
        This actor implementation difers from the original paper. 
        
        Reference:
        ---------- 
        Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al., 2018
        https://arxiv.org/abs/1802.09477 
        
        Continuous control with deep reinforcement learning, 
        https://arxiv.org/abs/1509.02971, Lillicrap et al., 2015
        """ 
        def __init__(
                self, 
                state_dim: int, 
                h1_dim: int,
                h2_dim: int,
                action_dim: int,
                action_scale: float=1.0
        ) -> None:
            super().__init__((state_dim,), action_dim, action_scale)
            self.state_dim = state_dim
            self.h1_dim = h1_dim
            self.h2_dim = h2_dim

            self.mlp = nn.Sequential(
                nn.Linear(state_dim, h1_dim),
                nn.ReLU(True),

                nn.Linear(h1_dim, h2_dim),
                nn.ReLU(True),
            
                nn.Linear(h2_dim, action_dim),
                nn.Tanh() 
            )

        def forward(self, s: torch.Tensor) -> torch.Tensor:
            if s.dim() == len(self.obs_shape):
                s.unsqueeze_(0)
            return self.action_scale * self.mlp(s)