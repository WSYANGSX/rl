import time
import tensorflow as tf
from rl.env import MountainCar
from rl.networks import ActorProb, Critic
from rl.data.buffer import ReplayBuffer
from rl.data.collector import Collector
from rl.policy import ACPolicy
from rl.networks import MLBasicLayer
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='ant_pg parameters')
    # 创建环境参数
    # --表示可选参数
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--render_mode', type=str, default='human')
    parser.add_argument('--buffer_size', type=int, default=100000)
    # 创建训练参数
    parser.add_argument('--eposide', type=int, default=10000)
    parser.add_argument('--step', type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_size", type=list, default=[256, 128, 64, 32])

    return parser.parse_args()


def test_ant_pg(args=get_args()):
    # 创建环境
    ant0 = MountainCar(xml_file=args.env, id=0, render_mode='human')
    # 创建内存
    buffer = ReplayBuffer(args.buffer_size)
    # 创建收集器
    collector = Collector(buffer)
    # 创建preprosses_net
    args.state_shape = ant0.observation_space.shape
    # print(args.state_shape)
    args.action_shape = ant0.action_space.shape
    # 创建预处理网路
    preprosses_net1 = MLBasicLayer(input_shape=args.state_shape,
                                   hidden_sizes=args.hidden_size,
                                   activation='relu',
                                   softmax=True)
    preprosses_net2 = MLBasicLayer(input_shape=args.state_shape,
                                   hidden_sizes=args.hidden_size,
                                   activation='relu',
                                   softmax=True)
    # 创建actor
    actor = ActorProb(preprosses_net1,
                      args.action_shape,
                      unbounded=True,
                      conditioned_sigma=False)
    # 创建critic
    critic = Critic(preprosses_net1, qcritic=False)

    # 创建策略
    optimizer1 = tf.optimizers.Adam(learning_rate=args.learning_rate)  # 优化器
    optimizer2 = tf.optimizers.Adam(learning_rate=args.learning_rate)
    policy = ACPolicy(evn=ant0,
                      actor=actor,
                      critic=critic,
                      optim1=optimizer1,
                      optim2=optimizer2,
                      discount_factor=args.gamma)

    total_rewards = []
    for epo in range(args.eposide):
        ant0.reset(int(time.time()))
        step = 1
        while (1):
            total_rew = 0
            action = policy.forward(ant0._current_obs)
            env_output = ant0.step(action)
            # print(env_output)                           # flow
            env_output['rew'] = env_output[
                'rew'] - 0.1 * step + 0.1 * env_output['obs_next'][1] + 0.2 * (
                    env_output['act']**2)
            total_rew += env_output['rew']
            collector.collect(env_output)
            policy.update(env_output, step)
            step += 1

            if env_output["terminated"] or env_output["truncated"]:
                total_rewards.append(total_rew)
                break

        print(f'eposide: {epo}, rew: {total_rew}')
    print(buffer)
    ant0.close()


if __name__ == "__main__":
    test_ant_pg()
