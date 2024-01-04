# -*- encoding: utf-8 -*-
import pdb
from Maze import Maze

import matplotlib.pyplot as plt
import numpy as np
import time
import pyttsx3
from scipy.io import savemat
import random

start = time.time()

if __name__ == '__main__':
    # import multiagent.scenarios as scenarios
    # scenario = scenarios.load("multiagent-particle-envs/multiagent/scenarios/simple_tag.py").Scenario()
    # world = scenario.make_world()
    # from multiagent.environment import MultiAgentEnv
    # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env = Maze()

    # n_agents = env.n; dim_act = world.dim_p * 2 + 1
    n_agents = 2;
    dim_act = 3
    # obs = env.reset();
    n_states = 34
    n_episode = 200; max_steps = 20
    from maddpg import MADDPG
    maddpg = MADDPG(n_agents, n_states, dim_act )
    # print('n_agents:', n_agents)
    # print('n_states:', n_states)
    # print('dim_act:', dim_act)



    step1 = 0

    average_return1 = []

    for i_episode in range(n_episode):
        print('episode:',i_episode)

        rit = np.random.poisson(10, 10)


        obs = np.array([[0, 0, 1000, 1000, 180, 800, 720, 280, 880, 820, 150, 200, 270, 580, 900, 460, 220, 380,
                         500, 350, 760, 380, 270, 160, rit[0], rit[1], rit[2], rit[3], rit[4],
                         rit[5], rit[6], rit[7], rit[8], rit[9]],
                        [0, 0, 1000, 1000, 180, 800, 720, 280, 880, 820, 150, 200, 270, 580, 900, 460, 220, 380,
                         500, 350, 760, 380, 270, 160, rit[0], rit[1], rit[2], rit[3], rit[4],
                         rit[5], rit[6], rit[7], rit[8], rit[9]]])

        uav_center2 = np.array([[0, 0], [1000, 1000]])


        user_center2 = np.array([[180, 800], [720, 280], [880, 820], [150, 200], [270, 580],
                                [900, 460], [220, 380], [500, 350], [760, 380], [270, 160]])


        total_reward = 0


        cumulative_reward1 = 0

        # print('rit:', rit)

        for t in range(max_steps):
            actions = maddpg.produce_action(obs/100)
            # print('obs:',obs)
            # print('actions:',actions)

            obs_, reward, done, uav_center2, user_center2, rit \
                = env.step(actions.detach(), uav_center2, user_center2, rit, step1, i_episode)



            next_obs = None

            step1 = step1 + 1

            if t < max_steps - 1:
                next_obs = obs_

            for r in reward:
                total_reward += r

            cumulative_reward1 += total_reward

            # print('cumulative_reward1:',cumulative_reward1)

            # adversaries_reward += (reward[0] + reward[1] + reward[2] )
            # goodagent_reward += reward[3]
            reward = reward / 100

            maddpg.memory.push(obs, actions, next_obs, reward)
            obs = next_obs;
            maddpg.train(i_episode);
            # env.render()

            # if done:
            #     average_return1.append(cumulative_reward1)
            #     break

            if t == max_steps - 1:
                average_return1.append(cumulative_reward1)
                break

        print('cumulative_reward1:',cumulative_reward1)
        print('rit:', rit)

        # print('Episode: %u' % (i_episode + 1) )
        # print('总体累积奖赏值 = %f' % (total_reward) )
        # print('adversary得到的累积奖赏值 = %f' % adversaries_reward )
        # print('good agent得到的累积奖赏值 = %f\n' % goodagent_reward )
        maddpg.episode_done += 1

    file_name = 'DQNr1.mat'
    savemat(file_name, {'DQNr1': average_return1})

    plt.figure(1)
    # plt.plot(np.arange(len(average_return)), average_return, 'r', marker='o')
    plt.plot(np.linspace(0, len(average_return1), n_episode), average_return1, label='BatteryLevel1')
    # plt.plot(np.linspace(0, len(average_return2), 1000), average_return2, label='BatteryLevel2')
    # plt.plot(np.linspace(0, len(average_return3), 1000), average_return3, label='BatteryLevel3')
    plt.legend(loc=4)
    plt.ylabel('Average_Return')
    plt.xlabel('training episodes')
    plt.show()
    plt.savefig('res.png')
    
    end = time.time()
    print("game over!")
    print('运行时间:', end - start)
    engine = pyttsx3.init()
    engine.say('程序运行完成')
    engine.runAndWait()