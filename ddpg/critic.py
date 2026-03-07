import copy
from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn


class Critic(nn.Module, ABC):
    """Critic interface for an action-value function.""" 
    def __init__(
            self, 
            obs_shape: Tuple[int, ...], 
            action_dim: int
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got: {action_dim}") 
        if len(obs_shape) == 0:
            raise ValueError(f"obs_shape must be non-empty, got: {obs_shape}")

        self.obs_shape = tuple(int(element) for element in obs_shape)
        self.action_dim = action_dim

    @abstractmethod
    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def predict(self, s: np.ndarray | torch.Tensor, a: np.ndarray | torch.Tensor) -> np.ndarray:
        device = next(self.parameters()).device 
        if isinstance(s, np.ndarray):
            s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
        else:
            s_t = s.to(device)
        if isinstance(a, np.ndarray):
            a_t = torch.as_tensor(a, dtype=torch.float32, device=device)
        else:
            a_t = a.to(device)

        if s_t.dim() == len(self.obs_shape):
            s_t.unsqueeze_(0)
        
        if a_t.dim() == 1:
            a_t.unsqueeze_(0)

        q_value = self(s_t, a_t).detach().cpu().numpy()
        return q_value

    def copy(self) -> 'Critic':
        return copy.deepcopy(self)


class CriticMLP(Critic):
        """
        This critic implementation difers from the original paper. 
        
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

        ) -> None:
            super().__init__((state_dim,), action_dim)
            self.state_dim = state_dim
            self.h1_dim = h1_dim
            self.h2_dim = h2_dim

            self.mlp = nn.Sequential(
                nn.Linear(state_dim + action_dim, h1_dim),
                nn.ReLU(True),
            
                nn.Linear(h1_dim, h2_dim),
                nn.ReLU(True),
            
                nn.Linear(h2_dim, 1)
            )

        def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            x = torch.cat([s, a], dim=-1) 
            return self.mlp(x)