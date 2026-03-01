import pytest

import torch
import numpy as np

from ddpg import CriticMLP


@pytest.fixture
def actor() -> CriticMLP:
    return CriticMLP(784, 400, 300, 10)


@pytest.fixture
def state_batch() -> torch.Tensor:
    return torch.randn(64, 784, dtype=torch.float32)


@pytest.fixture
def action_batch() -> torch.Tensor:
    return torch.randn(64, 10, dtype=torch.float32)


@pytest.fixture
def state() -> torch.Tensor:
    return torch.randn(784, dtype=torch.float32)


@pytest.fixture
def action() -> torch.Tensor:
    return torch.randn(10, dtype=torch.float32)


@pytest.fixture
def state_np() -> torch.Tensor:
    return np.random.randn(784).astype(np.float32)


@pytest.fixture
def action_np() -> torch.Tensor:
    return np.random.randn(10).astype(np.float32)


class TestActorMLP:
    def test_dims(self, actor: CriticMLP) -> None:
        assert actor.state_dim == 784
        assert actor.h1_dim == 400
        assert actor.h2_dim == 300
        assert actor.action_dim == 10

    def test_forward_shape(self, actor: CriticMLP, state_batch: torch.Tensor, action_batch: torch.Tensor) -> None:
        out = actor(state_batch, action_batch)
        assert out.shape == (64, 1)

    def test_predict_numpy_1(self, actor: CriticMLP, state: torch.Tensor, action: np.ndarray) -> None:
        out = actor.predict(state, action)
        assert isinstance(out, np.ndarray)
        assert out.shape == (1, 1)

    def test_predict_numpy_2(self, actor: CriticMLP, state_np: np.ndarray, action_np: np.ndarray) -> None:
        out = actor.predict(state_np, action_np)
        assert isinstance(out, np.ndarray)
        assert out.shape == (1, 1)