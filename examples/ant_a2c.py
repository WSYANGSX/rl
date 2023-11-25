import argparse
from src.networks import MLDense, MLBasicLayer
from src.policy import PGPolicy
from src.data.collector import Collector
from src.data.buffer import ReplayBuffer
from src.networks import ActorProb
from src.env import Ant_env
import tensorflow as tf
import time


def get_args():
    parser = argparse.ArgumentParser(description='ant_pg parameters')
    # 创建环境参数
    parser.add_argument('--env', type=str, default='Ant-v4')
    parser.add_argument('--render_mode', type=str, default='human')  # --表示可选参数
    parser.add_argument('--buffer_size', type=int, default=100000)
    # 创建训练参数
    parser.add_argument('--eposide', type=int, default=10000)
    parser.add_argument('--step', type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--hidden_size", type=list, default=[512, 256, 256])

    return parser.parse_args()


def test_ant_pg(args=get_args()):
    # 创建环境
    ant0 = Ant_env(xml_file=args.env,
                   id=0,
                   render_mode='human',
                   terminate_when_unhealthy=True,
                   use_contact_forces=True)
    # 创建内存
    buffer = ReplayBuffer(args.buffer_size)
    # 创建收集器
    collector = Collector(buffer)
    # 创建preprosses_net
    args.state_shape = ant0.observation_space.shape
    # print(args.state_shape)
    args.action_shape = ant0.action_space.shape
    # 创建预处理网路
    preprosses_net1 = MLBasicLayer(args.state_shape,
                                   hidden_sizes=args.hidden_size)
    preprosses_net1.summary()
    preprosses_net2 = MLDense(args.state_shape, hidden_sizes=args.hidden_size)
    # 创建actor
    actor1 = ActorProb(preprosses_net1,
                       args.action_shape,
                       unbounded=True,
                       conditioned_sigma=False)
    actor2 = ActorProb(preprosses_net2,
                       args.action_shape,
                       unbounded=True,
                       conditioned_sigma=False)
    # 创建策略
    optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)  # 优化器
    policy = PGPolicy(model=actor1,
                      optim=optimizer,
                      evn=ant0,
                      discount_factor=args.gamma)

    total_rewards = []
    for epo in range(args.eposide):
        while (1):
            total_rew = 0
            action = policy.forward(ant0._pre_obs)
            action = tf.reshape(action, (-1, ))
            env_output = ant0.step(action)
            total_rew += env_output['rew']
            collector.collect(env_output)

            if env_output["terminated"] or env_output["truncated"]:
                observation, info = ant0.reset(int(time.time()))
                data = policy.process_fn(*buffer.sample(0))
                policy.learn(data)
                print(policy._actor.trainable_variables)
                buffer.reset()
                total_rewards.append(total_rew)
                break
    print(total_rewards)
    ant0.close()


if __name__ == "__main__":
    test_ant_pg()
