import argparse

from agent import HPPO
from env import MECsystem
from utils import *

import matplotlib.pyplot as plt
import numpy as np
import os

def train(args, agent, env, test_env, save_dir, logger, start_episode=0, start_step=0):
    global_episode = start_episode
    global_step = start_step
    max_episode_step = int((args.possion_lambda * 0.15) / args.slot_time)

    # +++++++++++++
    rewards = []
    step_interval = 2048
    steps = []

    while True:
        s_t = env.reset()
        for j in range(max_episode_step):
            actions = agent.select_action(s_t)
            s_t1, r_t, done, _ = env.step(actions)

            agent.buffer.states.append(s_t)
            agent.buffer.rewards.append(r_t)
            agent.buffer.is_terminals.append(done)

            global_step += 1
            s_t = s_t1

            if global_step % args.step == 0:
                agent.update()
                test(args, global_episode, global_step, test_env, agent, logger)

            if done:
                global_episode += 1
                break

            # +++++++++++++
            if global_step % step_interval == 0:
                rewards.append(r_t)
                steps.append(global_step)

        if global_step > args.max_global_step:
            # test(args, global_episode, global_step, test_env, agent, logger)
            break

    agent.save_model(save_dir + 'CTScanckp.pt', args)

    scaled_rewards = np.array(rewards) * 10

    # +++++++++++++
    plt.plot(steps, scaled_rewards, marker=None, linestyle='-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.title('Reward Convergence')
    plt.savefig(os.path.join(save_dir, 'vgg11_test2048_reward_convergence.png'))
    plt.show()

def test(args, episode, step, test_env, agent, logger=None):
    done = False
    s_t = test_env.reset()
    test_reward = 0
    time_used = 0
    energy_used = 0
    finished = 0
    while not done:
        actions = agent.select_action(s_t, test=True)
        #++++++
        #print("s_t：",s_t)
        s_t1, r_t, done, info = test_env.step(actions)
        s_t = s_t1
        test_reward += r_t
        time_used += info['total_time_used']
        energy_used += info['total_energy_used']
        finished += info['total_finished']

    avg_time_used = time_used / finished
    avg_energy_used = energy_used / finished
    if logger is not None:
        logger.info(f'step {step}, reward {test_reward:.4f}, ({avg_time_used:.6f}s {avg_energy_used:.6f}j)/task/device')


def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='default_DRL')

    # system
    parser.add_argument('--net', default='resnet18', type=str)
    parser.add_argument('--possion_lambda', default=200, type=int)
    parser.add_argument('--num_channels', default=2, type=int)
    parser.add_argument('--num_users', default=5, type=int) #5
    parser.add_argument('--num_user_state', default=4, type=int)
    parser.add_argument('--pmax', default=1, type=float, help='max power')
    parser.add_argument('--dmax', default=100, type=float, help='max distance')
    parser.add_argument('--dmin', default=1, type=float, help='min distance')
    parser.add_argument('--beta', default=0.5, type=float) #1
    parser.add_argument('--alpha', default=0.5, type=float)  # args.alpha

    # channel
    parser.add_argument('--path_loss_exponent', default=3, type=float)
    parser.add_argument('--width', default=1e6, type=float)
    parser.add_argument('--noise', default=1e-9, type=float)

    # PPO
    parser.add_argument('--lr_a', default=0.0001, type=float, help='actor net learning rate')
    parser.add_argument('--lr_c', default=0.0001, type=float, help='critic net learning rate')

    #************** rnn
    # parser.add_argument('--lr_a', default=0.001, type=float, help='actor net learning rate')
    # parser.add_argument('--lr_c', default=0.001, type=float, help='critic net learning rate')

    parser.add_argument('--max_global_step', type=int, default=500000)  # 500000(500K)
    parser.add_argument('--gamma', type=float, default=0.95) #0.95
    parser.add_argument('--slot_time', default=0.5, type=float)

    parser.add_argument('--repeat_time', default=20, type=int) #20
    parser.add_argument('--step', default=1024, type=int)

    parser.add_argument('--batch_size', default=256, type=int)  #256
    parser.add_argument('--lam', default=0.95, type=float) #GAE lambda
    parser.add_argument('--eps_clip', default=0.2, type=float)
    parser.add_argument('--w_entropy', default=0.001, type=float)

    return parser.parse_args()


if __name__ == "__main__":
    args = init_parser()
    exp_name = f'{args.net}_MOCI'
    os.makedirs(os.path.join('CTScanresult', exp_name), exist_ok=True)
    logger = setup_logger(__name__, os.path.join('CTScanresult', exp_name))

    d_args = vars(args)
    for k in d_args.keys():
        logger.info(f'{k}: {d_args[k]}')

    user_params = {
        'num_channels': args.num_channels,
        'possion_lambda': args.possion_lambda,
        'pmax': args.pmax,
        'dmin': args.dmin,
        'dmax': args.dmax,
        'net': args.net,
        'test': False
    }
    test_user_params = {
        'num_channels': args.num_channels,
        'possion_lambda': args.possion_lambda,
        'pmax': args.pmax,
        'dmin': args.dmin,
        'dmax': args.dmax,
        'net': args.net,
        'test': True
    }
    channel_params = {
        'path_loss_exponent': args.path_loss_exponent,
        'width': args.width,
        'noise': args.noise
    }
    agent_params = {
        'num_users': args.num_users,
        'num_states': args.num_users * args.num_user_state,
        'num_channels': args.num_channels,
        'lr_a': args.lr_a,
        'lr_c': args.lr_c,
        'pmax': args.pmax,
        'gamma': args.gamma,
        'lam': args.lam,
        'repeat_time': args.repeat_time,
        'batch_size': args.batch_size,
        'eps_clip': args.eps_clip,
        'w_entropy': args.w_entropy,
    }

    #Training Reinforcement Learning Agents
    env = MECsystem(args.slot_time, args.num_users, args.num_channels, user_params, channel_params, args.alpha, args.beta)
    test_env = MECsystem(args.slot_time, args.num_users, args.num_channels, test_user_params, channel_params, args.alpha, args.beta)
    agent = HPPO(**agent_params)

    train(args, agent, env, test_env, os.path.join('CTScanresult', exp_name), logger)

    # CUDA_VISIBLE_DEVICES=3 python train_agent_CTScan.py
    # CUDA_VISIBLE_DEVICES=1 nohup python train_agent_CTScan.py > output.log &

