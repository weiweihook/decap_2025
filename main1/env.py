"""
A new ckt environment based on a new structure of MDP
"""
import gym
from gym import spaces
import numpy as np
import IPython
import os
import math
import config
import torch
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import time
import subprocess
import re
# CONSTANT
SYSTEM_CORES = 50  # max number of cores for multi-processeing

NCOL, NROW = 11, 11

execute_files = ['interposer_ac_novss1.sp', 'interposer_ac_novss2.sp', 'interposer_ac_novss3.sp', 'interposer_ac_novss4.sp']
port_files = ['port1_impeval.txt', 'port2_impeval.txt', 'port3_impeval.txt', 'port4_impeval.txt']

commands = []
for i in range(len(execute_files)):
    commands.append(["ngspice", execute_files[i]])

def run_os(path):
    original_path = os.getcwd()
    os.chdir(path)
    p = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for cmd in commands]
    for pp in p:
        pp.wait()
    os.chdir(original_path)

def readvdi(file):  # 读取csv中的vdi的数据，得到的是array数据，这里作为input
    zvdi = readresult(file)
    z = zvdi[:, 2]
    return z

def readresult(filename):
    a1 = np.genfromtxt(filename)
    return a1

def fill_non_zero(mask, cur_param):
    non_zeros_indices = np.nonzero(mask)
    dis = np.copy(mask)
    for i, j, k in zip(non_zeros_indices[0], non_zeros_indices[1], cur_param):
        dis[i][j] = k

    return dis

def get_target_imped():
    target_imped = []
    freq = readresult('config/case1/port1_impeval.txt')[:, 0]
    for i in range(len(freq)):
        if freq[i] < 3.5e9:
            target_imped.append(0.035)
        else:
            target_imped.append(0.035 * 10 ** (math.log(freq[i], 10) - math.log(3.5e9, 10))) # knee freq  = 3.5Ghz

    return target_imped


class DecapPlaceParallel(gym.Env):

    action_loc = 121
    action_meaning = [2000]  # index: 0,1,2,3,4,5,...,9,10
    action_space = np.array([len(action_meaning) * action_loc])
    action_space_shape = (1,)
    single_action_space = (1,)
    observation_space_shape = (1,  5, NCOL, NROW)
    single_observation_space_shape = (5, NCOL, NROW)

    def __init__(self, idx_list=[1, 2, 3, 4, 5], config_prefix='config/case1.cfg'):
        self.env_count = len(idx_list)
        self.config_names = [config_prefix[0:-5] + str(i) + config_prefix[-4:] for i in idx_list]
        self.vec_path = []
        self.vec_intp_mim = []
        self.vec_chip_mos = []
        self.vec_NCAP = []
        self.vec_intp_n = []
        self.vec_chip_n = []
        self.vec_cur_params_idx = []
        self.vec_dones = []
        self.vec_target_imped = []
        self.vec_sense = []
        self.vec_his_reward = []
        for i, f in enumerate(self.config_names):
            self._get_single_system(i, f)
        self.vec_intp_mask, self.vec_chip_mask, self.vec_mask = self.vec_gen_mask()

    def _get_single_system(self, idx, filename):
        path, intp_mim, chip_mos, NCAP, intp_n, chip_n = config.read_config(filename)
        self.vec_path.append(path)
        self.vec_intp_mim.append(intp_mim)
        self.vec_chip_mos.append(chip_mos)
        self.vec_NCAP.append(NCAP)
        self.vec_intp_n.append(intp_n)
        self.vec_chip_n.append(chip_n)
        self.vec_target_imped.append(get_target_imped())
        self.vec_sense.append(np.loadtxt(path + 'sense_impact1.txt'))

    def gen_mask(self, env_idx):
        intp_mask = np.ones(NCOL * NROW)
        chip_mask = np.zeros(NCOL * NROW)
        for i in range(len(self.vec_intp_n[env_idx])):
            intp_mask[self.vec_intp_n[env_idx][i]] = 0
        for j in range(len(self.vec_chip_n[env_idx])):
            chip_mask[self.vec_chip_n[env_idx][j]] = 1
        intp_mask = intp_mask.reshape(NCOL, NROW)
        chip_mask = chip_mask.reshape(NCOL, NROW)
        mask = list(np.concatenate([intp_mask, chip_mask]).reshape(-1))
        return env_idx, intp_mask, chip_mask, mask

    def vec_gen_mask(self, use_pool=False):
        """
        return the intp/chip mask for given env_idx
        """
        # use pool and map
        arg_list = range(self.env_count)
        chunk_size = 1
        if use_pool:
            with Pool(min(SYSTEM_CORES, self.env_count // chunk_size)) as p:
                mask_idx = p.map(self.gen_mask, arg_list, chunk_size)
                p.close()
            # sorted the reward based on env_idx
            vec_intp_mask = [r[1] for r in sorted(mask_idx, key=lambda x: x[0])]
            vec_chip_mask = [r[2] for r in sorted(mask_idx, key=lambda x: x[0])]
            vec_mask = [r[3] for r in sorted(mask_idx, key=lambda x: x[0])]
        else:
            _, vec_intp_mask, vec_chip_mask, vec_mask = map(list, zip(*[self.gen_mask(i) for i in arg_list]))

        return vec_intp_mask, vec_chip_mask, vec_mask

    def init_cur_params(self, env_idx):
        
        cur_params = np.zeros(self.vec_NCAP[env_idx], dtype=np.int32)
        return env_idx, cur_params

    def vec_init_cur_params(self, use_pool=False):
        """
        return the vec current params for given env_idx
        """
        # use pool and map
        arg_list = range(self.env_count)
        if use_pool:
            with Pool(min(SYSTEM_CORES, self.env_count)) as p:
                cur_idx = p.map(self.init_cur_params, arg_list)
                p.close()
            vec_cur = [r[1] for r in sorted(cur_idx, key=lambda x: x[0])]
        else:
            _, vec_cur = map(list, zip(*[self.init_cur_params(i) for i in arg_list]))
        return vec_cur

    def reset(self):
        self.vec_cur_params_idx = self.vec_init_cur_params()
        self.vec_dones = [False] * self.env_count
        obs = self.vec_get_obs()
        reward, imped = self.vec_cal_reward(use_pool=True)
        self.vec_his_reward = reward
        return np.stack(obs), np.stack(imped)

    def reset_idx(self, env_idx):
        _, self.vec_cur_params_idx[env_idx] = self.init_cur_params(env_idx)
        self.vec_dones[env_idx] = False
        _, obs_idx = self.get_obs(env_idx)
        _, reward_idx, imped_idx = self.cal_reward(env_idx)
        self.vec_his_reward[env_idx] = reward_idx
        return obs_idx, imped_idx

    def step(self, action):
        """
        :param action: is vector with elements between 0 and 2 mapped to the index of the corresponding parameter
        :return:
        """

        # Take action that RL agent returns to change current params
        for idx in range(self.env_count):
            action_idx = action[idx]
            action_loc = int(action_idx)
            self.vec_cur_params_idx[idx][action_loc] = 2000
            self.vec_dones[idx] = False

        state = self.vec_get_obs()
        cost_negative, total_imped = self.vec_cal_reward(use_pool=True)
        reward = [a-b for a,b in zip(cost_negative, self.vec_his_reward)]
        self.vec_his_reward = cost_negative

        info = {"reward_now": cost_negative,}

        return np.stack(state), np.stack(total_imped), reward, self.vec_dones, info

    def cal_reward(self, env_idx):

        str_dc = ''
        esrs = ''
        for i, val in enumerate(self.vec_cur_params_idx[env_idx][:self.vec_intp_mim[env_idx]]):
            str_dc += '.param dcap_int_val%d=%dp\n' % (i + 1, val)

        for j, val in enumerate(self.vec_cur_params_idx[env_idx][self.vec_intp_mim[env_idx]:]):
            str_dc += '.param dcap_int_val%d=%.2fp\n' % (j + 1 + self.vec_intp_mim[env_idx], val * 0.25)
            if val != 0:
                esrs += '.param esr%d=%.3f\n' % (j + 1, 200 / val)
            else:
                esrs += '.param esr%d=0\n' % (j + 1)

        f = open(self.vec_path[env_idx] + str(env_idx) + '/int_param_dcap.txt', 'w')
        f.write(str_dc)
        f.close()

        f1 = open(self.vec_path[env_idx] + str(env_idx) + '/moscap_esr.txt', 'w')
        f1.write(esrs)
        f1.close()

        # run_os(self.vec_path[env_idx])
        run_os(self.vec_path[env_idx] + str(env_idx) + '/')

        port1_arr = readresult(self.vec_path[env_idx] + str(env_idx) + '/port1_impeval.txt')
        freq1_val = port1_arr[:, 1]
        port2_arr = readresult(self.vec_path[env_idx] + str(env_idx) + '/port2_impeval.txt')
        freq2_val = port2_arr[:, 1]
        port3_arr = readresult(self.vec_path[env_idx] + str(env_idx) + '/port3_impeval.txt')
        freq3_val = port3_arr[:, 1]
        port4_arr = readresult(self.vec_path[env_idx] + str(env_idx) + '/port4_impeval.txt')
        freq4_val = port4_arr[:, 1]

        # add pid info
        maxlist1 = []
        for j in range(len(port1_arr)):
            maxlist1.append(max([freq1_val[j], freq2_val[j], freq3_val[j], freq4_val[j]]))

        freq_val = np.array(maxlist1)
        target_val = np.array(self.vec_target_imped[env_idx])
        max_imped = max(max(freq_val-target_val), 0)
        all_val = np.array([target_val-freq1_val, target_val-freq2_val, target_val-freq3_val, target_val-freq4_val]).reshape(-1)

        intp_cap_val = 0  
        chip_cap_val = 0        
        
        for j in self.vec_cur_params_idx[env_idx][:self.vec_intp_mim[env_idx]]:
            if j > 0:
                intp_cap_val += j
        
        for k in self.vec_cur_params_idx[env_idx][self.vec_intp_mim[env_idx]:]:
            if k > 0:
                chip_cap_val += k*0.25

        if max_imped == 0:
            total_cost = 1 * (0.5 * (self.vec_intp_mim[env_idx] * 2000 - intp_cap_val) / (self.vec_intp_mim[env_idx] * 2000)
                          + 0.5 * ((self.vec_chip_mos[env_idx] * 500 - chip_cap_val) / (self.vec_chip_mos[env_idx] * 500)))
        else:
            total_cost = -max_imped

        return env_idx, total_cost, all_val

    def vec_cal_reward(self, use_pool=False):
        """
        return the cost for given env_idx

        """
        # use pool and map
        arg_list = range(self.env_count)
        chunk_size = 1
        if use_pool:
            with Pool(min(SYSTEM_CORES, self.env_count // chunk_size)) as p:
                reward_idx = p.map(self.cal_reward, arg_list, chunk_size)
                p.close()
            # sorted the reward based on env_idx
            vec_reward = [r[1] for r in sorted(reward_idx, key=lambda x: x[0])]
            vec_total_imped = [r[2] for r in sorted(reward_idx, key=lambda x: x[0])]
        else:
            _, vec_reward, vec_total_imped = map(list, zip(*[self.cal_reward(i) for i in arg_list]))

        return vec_reward, vec_total_imped

    def get_obs(self, env_idx):
        intp_cap = self.vec_cur_params_idx[env_idx][:self.vec_intp_mim[env_idx]]
        chip_cap = self.vec_cur_params_idx[env_idx][self.vec_intp_mim[env_idx]:]
        MIM_dis = fill_non_zero(self.vec_intp_mask[env_idx], intp_cap) / 2000
        MOS_dis = fill_non_zero(self.vec_chip_mask[env_idx], chip_cap) / 2000
        sense_cap = self.vec_sense[env_idx].reshape(NCOL, NROW)
        # obs = np.array([self.vec_intp_mask[env_idx] - MIM_dis.astype(bool), MIM_dis,
        #                 self.vec_chip_mask[env_idx] - MOS_dis.astype(bool), MOS_dis])
        obs = np.array([self.vec_intp_mask[env_idx] - MIM_dis.astype(bool), MIM_dis,
                        MOS_dis, MOS_dis, sense_cap])

        return env_idx, obs

    def vec_get_obs(self, use_pool=False):
        """
        return the vec obs
        """
        # use pool and map
        arg_list = range(self.env_count)
        if use_pool:
            with Pool(min(SYSTEM_CORES, self.env_count)) as p:
                obs_idx = p.map(self.get_obs, arg_list)
                p.close()
            vec_obs = [r[1] for r in sorted(obs_idx, key=lambda x:x[0])]
        else:
            _, vec_obs = map(list, zip(*[self.get_obs(i) for i in arg_list]))
        return vec_obs

    def action_mask(self, env_idx):
        mask = np.ones((self.action_loc, len(self.action_meaning)))
        for i in range(self.action_loc):
            if self.vec_cur_params_idx[env_idx][i] != 0:
                mask[i] = np.zeros(len(self.action_meaning))
        action_mask = mask.reshape(-1)

        return env_idx, action_mask

    def vec_action_mask(self, use_pool=False):
        """
        return the action mask for given env_idx

        """
        # use pool and map
        arg_list = range(self.env_count)
        chunk_size = 1
        if use_pool:
            with Pool(min(SYSTEM_CORES, self.env_count // chunk_size)) as p:
                action_mask_idx = p.map(self.action_mask, arg_list, chunk_size)
                p.close()
            # sorted the reward based on env_idx
            vec_action_mask = [r[1] for r in sorted(action_mask_idx, key=lambda x: x[0])]
        else:
            _, vec_action_mask = map(list, zip(*[self.action_mask(i) for i in arg_list]))

        return vec_action_mask


if __name__ == "__main__":
    env = DecapPlaceParallel([5]*1)
    s = time.time()
    o, r = env.reset()
    o, i, r, d, info = env.step(np.array([[0]]))
    e = time.time()
    print(r, e-s)



