import numpy as np
import tkinter as tk
import time
import random
import math
from scipy.spatial.distance import cdist
import scipy.io


np.random.seed(1)


p = 0.1



d_ite = 100

# use_num = 10
use_num = 10

UAV_fly = 300 #每个时隙最大移动距离
# UAV_fly = 30



class Maze(object):
    def __init__(self):
        super(Maze, self).__init__()

        self.n_actions = 38880
        # self.n_actions = 68040
        self.n_features = 9  # 无人机位置, 电池量，信道接入情况
        # self._build_maze()

    def step(self, action, uav_center2, user_center2, rit, step1, episode):

        # p_j = np.array([p_j1, p_j2])
        # print('p_j: ', p_j)

        # print('action: ', action[0][2])
        # print('action: ', action[1][2])

        if action[0][2] == -1:
            action02 = 0
        else:
            action02 = int((action[0][2] + 1) / 2 * 9)

        if action[1][2] == -1:
            action12 = 0
        else:
            action12 = int((action[1][2] + 1) / 2 * 9)

        rit[action02] = 0
        rit[action12] = 0


        ###第一个无人机

        uav_center2[0][0] = uav_center2[0][0] + UAV_fly * action[0][0] * math.cos(2 * math.pi * action[0][1])
        uav_center2[0][1] = uav_center2[0][1] + UAV_fly * action[0][0] * math.sin(2 * math.pi * action[0][1])
        uav_center2[1][0] = uav_center2[1][0] + UAV_fly * action[1][0] * math.cos(2 * math.pi * action[1][1])
        uav_center2[1][1] = uav_center2[1][1] + UAV_fly * action[1][0] * math.sin(2 * math.pi * action[1][1])


        # print('uav_center2: ', uav_center2)
        # print('uav_center3: ', uav_center3)


        #  限制无人机飞行空间
        if uav_center2[0][0] > 1000:
            uav_center2[0][0] = 1000
        if uav_center2[0][0] < 0:
            uav_center2[0][0] = 0
        if uav_center2[0][1] > 1000:
            uav_center2[0][1] = 1000
        if uav_center2[0][1] < 0:
            uav_center2[0][1] = 0
        if uav_center2[1][0] > 1000:
            uav_center2[1][0] = 1000
        if uav_center2[1][0] < 0:
            uav_center2[1][0] = 0
        if uav_center2[1][1] > 1000:
            uav_center2[1][1] = 1000
        if uav_center2[1][1] < 0:
            uav_center2[1][1] = 0

        # 第一个无人机与所选用户之间的距离
        d_1 = np.sqrt((uav_center2[0][0] - user_center2[action02][0])
                      ** 2 + (uav_center2[0][1] - user_center2[action02][1]) ** 2)


        # 第二个无人机与所选用户之间的距离
        d_2 = np.sqrt((uav_center2[1][0] - user_center2[action12][0])
                      ** 2 + (uav_center2[1][1] - user_center2[action12][1]) ** 2)


        # 找出大于零的数的个数
        count_negative = np.sum(rit > 0)
        reward_1_1 = - 0.1 * count_negative

        if np.all(rit == 0):
            reward_uav1 = 0
            reward_uav2 = 0
        else:
            reward_uav1 = - 0.001*d_1 + reward_1_1
            reward_uav2 = - 0.001*d_2 + reward_1_1

        reward = np.hstack((reward_uav1, reward_uav2))

        if step1 == 20:
            done = True
        else:
            done = False


        # 用户向各方向移动
        user_center2[0][0] = user_center2[0][0] + 1
        user_center2[0][1] = user_center2[0][1] - 1

        rit = np.hstack((rit[0], rit[1], rit[2], rit[3], rit[4],
                        rit[5], rit[6], rit[7], rit[8], rit[9]))

        s_ = np.array([[uav_center2[0][0], uav_center2[0][1], uav_center2[1][0], uav_center2[1][1],
                        user_center2[0][0], user_center2[0][1], user_center2[1][0], user_center2[1][1],
                        user_center2[2][0], user_center2[2][1], user_center2[3][0], user_center2[3][1],
                        user_center2[4][0], user_center2[4][1], user_center2[5][0], user_center2[5][1],
                        user_center2[6][0], user_center2[6][1], user_center2[7][0], user_center2[7][1],
                        user_center2[8][0], user_center2[8][1], user_center2[9][0], user_center2[9][1],
                        rit[0], rit[1], rit[2], rit[3], rit[4],
                        rit[5], rit[6], rit[7], rit[8], rit[9]],
                        [uav_center2[0][0], uav_center2[0][1], uav_center2[1][0], uav_center2[1][1],
                        user_center2[0][0], user_center2[0][1], user_center2[1][0], user_center2[1][1],
                        user_center2[2][0], user_center2[2][1], user_center2[3][0], user_center2[3][1],
                        user_center2[4][0], user_center2[4][1], user_center2[5][0], user_center2[5][1],
                        user_center2[6][0], user_center2[6][1], user_center2[7][0], user_center2[7][1],
                        user_center2[8][0], user_center2[8][1], user_center2[9][0], user_center2[9][1],
                        rit[0], rit[1], rit[2], rit[3], rit[4],
                        rit[5], rit[6], rit[7], rit[8], rit[9]]])

        # return s_, reward, done, uav_center2, user_center2, uav_center3, user_center3, t_com_1sum_1, t_ncom_1sum_1, S_SE
        return s_, reward, done, uav_center2, user_center2, rit




