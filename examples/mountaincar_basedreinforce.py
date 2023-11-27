import os
import argparse
import numpy as np
from rl.networks import MLBasicLayer
from rl.policy import Reinforce
from rl.data.collector import Collector
from rl.data.buffer import ReplayBuffer
from rl.networks.continuous import ActorProb, Critic
from rl.env import MountainCar
from rl.trainer import Online
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_args():
    parser = argparse.ArgumentParser()
    # 创建环境参数
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--render_mode', type=str, default='human')  # --表示可选参数
    parser.add_argument('--buffer_size', type=int, default=100000)
    # 创建训练参数
    parser.add_argument('--eposide', type=int, default=10000)
    parser.add_argument('--step', type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--actor_learning_rate", type=float, default=0.0001)
    parser.add_argument("--critic_learning_rate", type=float, default=0.0001)
    parser.add_argument("--hidden_size", type=list, default=[64, 128, 64])
    parser.add_argument("--activations",
                        type=list,
                        default=['relu', 'relu', 'relu'])
    parser.add_argument("--seed", type=int, default=23)

    return parser.parse_args()


def test_ant_pg(args=get_args()):
    # 创建环境
    env0 = MountainCar(xml_file=args.env, id=0, render_mode='human')

    # 创建随机种子
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # 创建内存
    buffer = ReplayBuffer(args.buffer_size)

    # 创建收集器
    collector = Collector(buffer)

    # 创建preprosses_net
    args.state_shape = env0.observation_space.shape

    # print(args.state_shape)
    args.action_shape = env0.action_space.shape

    # 创建预处理网路
    preprosses_net1 = MLBasicLayer(input_shape=args.state_shape,
                                   hidden_sizes=args.hidden_size,
                                   activations=args.activations)
    preprosses_net2 = MLBasicLayer(input_shape=args.state_shape,
                                   hidden_sizes=args.hidden_size,
                                   activations=args.activations)

    # 创建actor
    actor = ActorProb(preprosses_net1,
                      args.action_shape,
                      unbounded=False,
                      conditioned_sigma=False)
    critic = Critic(preprosses_net2)

    # 创建优化器
    actor_optimizer = tf.optimizers.Adam(
        learning_rate=args.actor_learning_rate)  # 优化器
    critic_optimizer = tf.optimizers.Adam(
        learning_rate=args.critic_learning_rate)

    # 创建策略
    policy = Reinforce(env=env0,
                       actor=actor,
                       actor_optim=actor_optimizer,
                       discount_factor=args.gamma,
                       action_scaling=True,
                       action_bound_method='clip',
                       base_line=True,
                       critic=critic,
                       critic_optim=critic_optimizer)

    # 创建训练器
    trainer = Online(buffer=buffer,
                     collector=collector,
                     policy=policy,
                     eposide=args.eposide,
                     rollout_times=1)

    trainer.train()


if __name__ == "__main__":
    test_ant_pg()
