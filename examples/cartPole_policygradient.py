import tensorflow as tf
from rl.env import CartPole
from rl.networks import MLBasicLayer
from rl.networks.discrete import Actor, Critic
from rl.data.buffer import ReplayBuffer
from rl.data.collector import Collector
from rl.policy import PolicyGradient
import argparse
import numpy as np
from rl.trainer import Online


def get_args():
    parser = argparse.ArgumentParser(description="ant_pg parameters")
    # 创建环境参数
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--render_mode", type=str, default="human")  # --表示可选参数
    parser.add_argument("--buffer_size", type=int, default=100000)
    # 创建训练参数
    parser.add_argument("--eposide", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--hidden_size", type=list, default=[10])
    parser.add_argument("--activations", type=list, default=["relu"])
    parser.add_argument("--actor_learning_rate", type=float, default=0.01)
    parser.add_argument("--critic_learning_rate", type=float, default=2e-6)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def test_ant_pg(args=get_args()):
    # 设置随即变量种子
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    # 创建环境
    agent = CartPole(xml_file=args.env, render_mode="human", seed=args.seed)
    # 创建内存
    buffer = ReplayBuffer(args.buffer_size)
    # 创建收集器
    collector = Collector(buffer)
    # 创建preprosses_net
    args.state_shape = agent.observation_space.shape
    # print(args.state_shape)
    args.action_shape = (2,)
    # 创建预处理网路
    preprosses_net1 = MLBasicLayer(
        args.state_shape,
        hidden_sizes=args.hidden_size,
        activations=args.activations,
        softmax=False,
    )
    preprosses_net2 = MLBasicLayer(
        args.state_shape,
        hidden_sizes=args.hidden_size,
        activations=args.activations,
        softmax=True,
    )
    # 创建actor
    actor = Actor(
        preprosses_net=preprosses_net1,
        action_shape=args.action_shape,
        hidden_sizes=[],
        activations=[],
    )
    actor.summary()
    # 创建critic
    critic = Critic(
        preprosses_net=preprosses_net2,
        hidden_sizes=[],
        activations=[],
    )
    # 创建优化器
    actor_optimizer = tf.optimizers.Adam(
        learning_rate=args.actor_learning_rate
    )  # 优化器
    critic_optimizer = tf.optimizers.Adam(learning_rate=args.critic_learning_rate)

    policy = PolicyGradient(
        env=agent,
        actor=actor,
        actor_optim=actor_optimizer,
        discount_factor=args.gamma,
        action_scaling=True,
        action_bound_method="clip",
        base_line=True,
        critic=critic,
        critic_optim=critic_optimizer,
    )

    trainer = Online(
        buffer=buffer,
        collector=collector,
        policy=policy,
        eposide=args.eposide,
        rollout_times=1,
    )
    trainer.train(
        method="offpolicy",
        pre_process_funs=[policy.compute_episodic_return, policy.return_normalization],
    )


if __name__ == "__main__":
    test_ant_pg()
