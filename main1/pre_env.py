import sys
sys.path.append("..")
import numpy as np
import os
import shutil
from copy import deepcopy
import config
import math
import pyglet
import gym
from pyglet import shapes
import torch
from multiprocessing import Pool
import time
import torch.nn as nn

# CONSTANT
SYSTEM_CORES = 50 # max number of cores for multi-processeing

class PrePDNEnv(gym.Env):

    def __init__(self, idx_list=[1, 2, 3, 4, 5], config_prefix='config/case1.cfg'):
        """initilize the vectorizec env
        Args:
            idx_list: index for config files.
            config_prefix: prefix for pathes of config files.

        Returns:
            Vectorized env for chiplet placement.

        """
        self.env_count = len(idx_list)
        # get config/case + .cfg; this is very customized, have to be changed if the name & path are changed
        self.config_names = [config_prefix[0:-5] + str(i) + config_prefix[-4:] for i in idx_list]
        # Initialize class atributions
        self.vec_path = []

        for i, f in enumerate(self.config_names):
            self._get_single_path(i, f)

    def copy(self):
        """
        Rest the  all vectorized enviroment
        """
        for i in range(self.env_count):
            self.copy_file_idx(i)


    def copy_file_idx(self, env_idx):
        source_path = self.vec_path[env_idx]
        dest_path = self.vec_path[env_idx] + '%d/' % (env_idx)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        file_extensions = ['.subckt', '.txt', '.sp', 'inc']
        files = [f for f in os.listdir(source_path) if any(f.endswith(ext) for ext in file_extensions)]
        for file in files:
            source_file = os.path.join(source_path, file)
            dest_file = os.path.join(dest_path, file)
            shutil.copy(source_file, dest_file)


    def _get_single_path(self, idx, filename):

        path, _, _, _, _, _ = config.read_config(filename)
        self.vec_path.append(path)

if __name__ == '__main__':
    env = PrePDNEnv([6]*20)
    env.copy()

