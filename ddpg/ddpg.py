import os

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import gymnasium as gym

from ddpg.actor import Actor
from ddpg.critic import Critic
from ddpg.replay_buffer import ReplayBuffer


class DDPG:
    """
    Simple pytorch implementation of the DDPG algorithm described by Lillicrap et al., 2016.

    Reference:
    ----------
    Continuous control with deep reinforcement learning, Lillicrap et al., 2016
    https://arxiv.org/abs/1509.02971
    """ 
    def __init__(
            self,
            actor: Actor,
            critic: Critic,
            lr_actor: float,
            lr_critic: float,
            timesteps: int,
            gamma: float,
            tau: float,
            batch_size: int,
            device: str="cpu",
            noise_std: float=1.0,
            weight_decay_critic: float=0.02,
            buffer_capacity: int=100_000,
            buffer_start_size: int=10_000,
            update_target_networks_every: int=10_000,
            eval_every: int=1_000,
            save_every: int=100_000,
    ) -> None:
        self.device = torch.device(device)

        # Networks 
        self.actor = actor
        self.actor_target = actor.copy()
        self.actor.to(self.device)
        self.actor_target.to(self.device)

        self.critic = critic
        self.critic_target = critic.copy()
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        # Optimizers 
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay_critic)

        # Losses 
        self.criterion_critic = nn.MSELoss()

        # Hyperparameters 
        self.timesteps = timesteps
        self.gamma = gamma 
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.noise_std = noise_std

        # Shape stuff
        self.obs_shape = actor.obs_shape
        self.action_dim = actor.action_dim

        # Replay buffer 
        self.replay_buffer = ReplayBuffer(
            self.obs_shape, 
            self.action_dim, 
            buffer_capacity, 
            batch_size,
            self.device
        )

        # More settings
        self.buffer_start_size = buffer_start_size
        self.update_target_networks_every = update_target_networks_every
        self.save_every = save_every
        self.eval_every = eval_every

        # Stats
        self.global_step = 0
        self.stats = {"timestep" : [], "mean_episode_return" : []}

    @torch.no_grad()
    def get_action(self, x: np.ndarray, noise: bool=True) -> np.ndarray:
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        a_t = self.actor(x_t)
        if noise: 
            a_t = a_t + torch.randn_like(a_t) * self.noise_std
        a_t = torch.clamp(a_t, -1.0, 1.0) 
        return a_t.squeeze(0).cpu().numpy()

    def update_networks(self) -> None:
        self.actor.train(); self.critic.train()
        self.actor_target.eval(); self.critic_target.eval() 

        # Sample random minibatch of N transitions
        s, a, r, s_nxt, d = self.replay_buffer.sample()

        # Computing TD target 
        with torch.no_grad():
            a_nxt_tgt = self.actor_target(s_nxt)                        # [B, action_dim]
            q_nxt_tgt = self.critic_target(s_nxt, a_nxt_tgt).view(-1)   # [B]
            td_target = r + self.gamma * (1.0 - d) * q_nxt_tgt          # [B]

        # Update critic
        q = self.critic(s, a)                                       # [B, 1]
        loss_critic = self.criterion_critic(td_target, q.view(-1))  # [1]
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # Update actor
        q = self.critic(s, self.actor(s))
        loss_actor = torch.mean(-q)
        self.optimizer_actor.zero_grad() 
        loss_actor.backward()
        self.optimizer_actor.step()

    def update_target_networks(self) -> None:
        # Update critic (target) 
        for theta, theta_old in zip(self.critic.parameters(), self.critic_target.parameters()):
            theta = theta.data
            theta_old = theta_old.data
            theta_old.copy_(self.tau * theta + (1 - self.tau) * theta_old)
        
        # Update actor (target) 
        for theta, theta_old in zip(self.actor.parameters(), self.actor_target.parameters()):
            theta = theta.data
            theta_old = theta_old.data
            theta_old.copy_(self.tau * theta + (1 - self.tau) * theta_old)

    def train(self, env: gym.Env) -> None:
        self.explore_env(env)
        done = True 
        for _ in range(self.timesteps): 
            if done:
                s, info = env.reset()
            a = self.get_action(s)
            s_nxt, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            self.replay_buffer.push(s, a, r, s_nxt, done)

            self.update_networks()
            self.update_target_networks()

            if self.global_step % self.eval_every == 0:
                mean_ep_reward = np.mean(self.evaluate())
                self.stats["timestep"].append(self.global_step)
                self.stats["mean_episode_return"].append(mean_ep_reward)
                print(f"timestep: {self.global_step}\tmean-episode-return: {mean_ep_reward:.2f}")

            if self.global_step % self.save_every == 0:
                self._checkpoint()

            s = s_nxt
            self.global_step += 1
        env.close()

    def explore_env(self, env: gym.Env) -> None:
        self._cache_env(env)
        done = True 
        for _ in range(self.buffer_start_size): 
            if done:
                s, _ = env.reset()
            a = self.get_action(s)
            s_nxt, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            self.replay_buffer.push(s, a, r, s_nxt, done)
            s = s_nxt
    
    @torch.inference_mode() 
    def evaluate(self) -> list[int]:
        self.actor.eval(); self.critic.eval()
        env = gym.make(self.env_id, render_mode=None)
        rewards = []
        for _ in range(5):
            reward_sum = 0.0 
            done = False
            s, _ = env.reset()
            while not done:
                a = self.actor.predict(s)
                s_nxt, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated 
                s = s_nxt
                reward_sum += r
            rewards.append(reward_sum)
        env.close() 
        return rewards

    def _cache_env(self, env: gym.Env) -> None:
        self.env_id = env.spec.id

    def _checkpoint(self) -> None:
        save_dir = f"{self.env_id}-DDPG-Checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        file_name = f"{self.env_id}-DDPG-Actor-Lr{self.lr_actor}-t{self.global_step}.pt"
        file_name = os.path.join(save_dir, file_name) 
        torch.save(self.actor.state_dict(), file_name)
        
        file_name = f"{self.env_id}-DDPG-Critic-Lr{self.lr_actor}-t{self.global_step}.pt"
        file_name = os.path.join(save_dir, file_name) 
        torch.save(self.critic.state_dict(), file_name)

        file_name = f"{self.env_id}-DDPG.csv"
        pd.DataFrame.from_dict(self.stats).to_csv(file_name, index=False)