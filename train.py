from argparse import Namespace, ArgumentParser

from ddpg.ddpg import DDPG
from ddpg.actor import ActorMLP
from ddpg.critic import CriticMLP

import gymnasium as gym


def parse_args() -> Namespace:
    parser = ArgumentParser(description="DDPG training")

    parser.add_argument("--env_id", type=str, default="Hopper-v5")
    parser.add_argument("--state_dim", type=int, default=11)
    parser.add_argument("--action_dim", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--h1_dim", type=int, default=400)
    parser.add_argument("--h2_dim", type=int, default=300)

    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    parser.add_argument("--weight_decay_critic", type=float, default=1e-2)

    parser.add_argument("--buffer_capacity", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--expl_noise", type=float, default=0.2)

    parser.add_argument("--num_timesteps", type=int, default=3_000_000)
    parser.add_argument("--update_target_network_every", type=int, default=1)

    parser.add_argument("--save_every", type=int, default=50_000)
    parser.add_argument("--eval_every", type=int, default=10_000)

    parser.add_argument("--verbose", default=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    actor = ActorMLP(args.state_dim, args.h1_dim, args.h2_dim, args.action_dim)
    critic = CriticMLP(args.state_dim, args.h1_dim, args.h2_dim, args.action_dim)

    pa = sum(p.numel() for p in actor.parameters())
    pc = sum(p.numel() for p in critic.parameters())
    print(pa, pc)

    ddpg = DDPG(
        actor=actor,
        critic=critic,
        timesteps=args.num_timesteps,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        weight_decay_critic=args.weight_decay_critic,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        gamma=args.gamma,
        tau=args.tau,
        noise_std=args.expl_noise,
        device=args.device,
        eval_every=args.eval_every,
        save_every=args.save_every

    ) 
    env = gym.make(args.env_id)
    ddpg.train(env)


if __name__ == "__main__":
    main()