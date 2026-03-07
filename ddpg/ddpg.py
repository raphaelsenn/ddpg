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
    Simple pytorch implementation of the DDPG algorithm described by Lillicrap et al., 2015.

    Reference:
    ----------
    Continuous control with deep reinforcement learning, Lillicrap et al., 2015
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
            noise_std: float=0.1,
            weight_decay_actor: float=0.0,
            weight_decay_critic: float=0.0,
            buffer_capacity: int=1_000_000,
            buffer_start_size: int=25_000,
            n_eval_runs: int=10,
            eval_every: int=1_000,
            save_every: int=100_000,
            seed: int=0
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
        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(), lr=lr_actor, weight_decay=weight_decay_actor
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay_critic
        )

        # Loss
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
        self.action_scale = actor.action_scale

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
        self.n_eval_runs = n_eval_runs 
        self.save_every = save_every
        self.eval_every = eval_every
        
        self.env_id = None
        self.seed = seed

        # Stats
        self.global_step = 0
        self.stats = {"t" : [], "average_return" : [], "std_return": []}

    @torch.no_grad()
    def get_action(self, x: np.ndarray, noise: bool=True) -> np.ndarray:
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        a_t = self.actor(x_t)
        if noise: a_t = a_t + torch.randn_like(a_t) * self.noise_std
        a_t = torch.clamp(a_t, -self.action_scale, self.action_scale) 
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
        q = self.critic(s, a)                                           # [B, 1]
        loss_critic = self.criterion_critic(td_target, q.view(-1))      # [1]
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # Update actor
        q = self.critic(s, self.actor(s))                               # [B, 1]
        loss_actor = torch.mean(-q)                                     # [1]
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

    @torch.no_grad()
    def update_target_networks(self) -> None:
        # Update critic (target) 
        for theta, theta_old in zip(self.critic.parameters(), self.critic_target.parameters()):
            theta_old.data.copy_(self.tau * theta.data + (1.0 - self.tau) * theta_old.data) 
        
        # Update actor (target) 
        for theta, theta_old in zip(self.actor.parameters(), self.actor_target.parameters()):
            theta_old.data.copy_(self.tau * theta.data + (1.0 - self.tau) * theta_old.data) 

    def train(self, env: gym.Env) -> None:
        self.explore_env(env)
        
        episode_num = 0
        done = False 
        s, _ = env.reset() 
        for t in range(self.timesteps): 
            a = self.get_action(s, noise=True)
            s_nxt, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            self.replay_buffer.push(s, a, r, s_nxt, terminated)

            self.update_networks()
            self.update_target_networks()

            # Update state 
            s = s_nxt
            self.global_step = t

            if done:
                s, _ = env.reset()
                episode_num += 1

            if self.global_step % self.eval_every == 0:
                self.evaluate()
                average_return = self.stats["average_return"][-1]
                print(
                    f"Total T: {self.global_step:6d}\t"
                    f"Episode: {episode_num:5d}\t"
                    f"Average Return: {average_return:10.3f}"
                )

            if self.global_step % self.save_every == 0:
                self.checkpoint()

        self.checkpoint()
        env.close()

    def explore_env(self, env: gym.Env) -> None:
        if self.env_id is None: self.env_id = env.spec.id

        s, _ = env.reset(seed=self.seed)
        env.action_space.seed(self.seed)

        done = False
        for _ in range(self.buffer_start_size): 
            a = env.action_space.sample()
            s_nxt, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            self.replay_buffer.push(s, a, r, s_nxt, terminated)
            s = s_nxt

            if done:
                s, _ = env.reset()
                done = False

    @torch.inference_mode() 
    def evaluate(self) -> None:
        self.actor.eval(); self.critic.eval()

        done = False
        env = gym.make(self.env_id, render_mode=None)
        s, _ = env.reset(seed=self.seed + 100)
        
        rewards = np.zeros(self.n_eval_runs, dtype=np.float32)
        for n in range(self.n_eval_runs):
            while not done:
                a = self.get_action(s, noise=False)
                s_nxt, reward, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                s = s_nxt
                rewards[n] += reward
            done = False
            s, _ = env.reset()

        env.close()
        self.update_stats(rewards)

    def update_stats(self, rewards: np.ndarray) -> None:
        mean_ep_reward = float(np.mean(rewards))
        std_ep_reward = float(np.std(rewards))
        self.stats["t"].append(self.global_step)
        self.stats["average_return"].append(mean_ep_reward)
        self.stats["std_return"].append(std_ep_reward)

    def checkpoint(self) -> None:
        save_dir = f"{self.env_id}-DDPG-Checkpoints-Seed{self.seed}"
        os.makedirs(save_dir, exist_ok=True)

        file_name = f"{self.env_id}-DDPG-Actor-Lr{self.lr_actor}-t{self.global_step}-Seed{self.seed}.pt"
        file_name = os.path.join(save_dir, file_name) 
        torch.save(self.actor.state_dict(), file_name)
        
        file_name = f"{self.env_id}-DDPG-Critic-Lr{self.lr_actor}-t{self.global_step}-Seed{self.seed}.pt"
        file_name = os.path.join(save_dir, file_name) 
        torch.save(self.critic.state_dict(), file_name)

        file_name = f"{self.env_id}-DDPG-Seed{self.seed}.csv"
        pd.DataFrame.from_dict(self.stats).to_csv(file_name, index=False)