"""
A new circuit environment based on a new structure of MDP for decap placement optimization.
"""

import os
import math
import time
import subprocess
from multiprocessing import Pool
from typing import List, Tuple, Union

import gym
import numpy as np

import config
# Constants
SYSTEM_CORES = 50  # Maximum number of cores for multi-processing
NCOL, NROW = 11, 11
DEFAULT_CAP_VALUE = 2000
KNEE_FREQUENCY = 3.5e9
TARGET_IMPEDANCE_BASE = 0.035
MOS_CAP_SCALING = 0.25
ESR_SCALING = 200
COST_PENALTY = -1

# File configurations
EXECUTE_FILES = ['interposer_ac_novss1.sp', 'interposer_ac_novss2.sp', 
                 'interposer_ac_novss3.sp', 'interposer_ac_novss4.sp']
PORT_FILES = ['port1_impeval.txt', 'port2_impeval.txt', 
              'port3_impeval.txt', 'port4_impeval.txt']

# Generate ngspice commands
COMMANDS = [["ngspice", file] for file in EXECUTE_FILES]

def run_os(path: str) -> None:
    """Execute ngspice commands in the specified directory."""
    original_path = os.getcwd()
    try:
        os.chdir(path)
        processes = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
                    for cmd in COMMANDS]
        for process in processes:
            process.wait()
    finally:
        os.chdir(original_path)

def readvdi(file: str) -> np.ndarray:
    """Read VDI data from CSV file and return impedance values."""
    zvdi = readresult(file)
    return zvdi[:, 2]

def readresult(filename: str) -> np.ndarray:
    """Read numerical data from file using numpy.genfromtxt."""
    try:
        return np.genfromtxt(filename)
    except (FileNotFoundError, IOError) as e:
        raise FileNotFoundError(f"Cannot read file {filename}: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing data from {filename}: {e}")

def fill_non_zero(mask: np.ndarray, cur_param: np.ndarray) -> np.ndarray:
    """Fill non-zero positions in mask with corresponding parameter values."""
    non_zeros_indices = np.nonzero(mask)
    result = np.copy(mask).astype(float)
    for i, j, k in zip(non_zeros_indices[0], non_zeros_indices[1], cur_param):
        result[i][j] = k
    return result

def get_target_imped(config_path: str = 'config/case1/port1_impeval.txt') -> List[float]:
    """Calculate target impedance based on frequency response."""
    freq = readresult(config_path)[:, 0]
    target_imped = []
    
    for frequency in freq:
        if frequency < KNEE_FREQUENCY:
            target_imped.append(TARGET_IMPEDANCE_BASE)
        else:
            # Calculate impedance above knee frequency
            log_ratio = math.log10(frequency) - math.log10(KNEE_FREQUENCY)
            target_imped.append(TARGET_IMPEDANCE_BASE * (10 ** log_ratio))
    
    return target_imped


class DecapPlaceParallel(gym.Env):
    """Parallel environment for decap placement optimization using reinforcement learning."""
    
    # Class constants
    ACTION_LOCATIONS = 121
    ACTION_MEANINGS = [DEFAULT_CAP_VALUE]
    ACTION_SPACE = np.array([len(ACTION_MEANINGS) * ACTION_LOCATIONS])
    ACTION_SPACE_SHAPE = (1,)
    SINGLE_ACTION_SPACE = (1,)
    OBSERVATION_SPACE_SHAPE = (1, 5, NCOL, NROW)
    SINGLE_OBSERVATION_SPACE_SHAPE = (5, NCOL, NROW)

    def __init__(self, idx_list: List[int] = None, config_prefix: str = 'config/case1.cfg'):
        """Initialize the parallel decap placement environment.
        
        Args:
            idx_list: List of environment indices to create
            config_prefix: Prefix for configuration files
        """
        if idx_list is None:
            idx_list = [1, 2, 3, 4, 5]
            
        self.env_count = len(idx_list)
        self.config_names = self._generate_config_names(idx_list, config_prefix)
        
        # Initialize environment vectors
        self._initialize_vectors()
        
        # Load configurations for each environment
        for i, config_file in enumerate(self.config_names):
            self._get_single_system(i, config_file)
            
        # Generate masks for all environments
        self.vec_intp_mask, self.vec_chip_mask, self.vec_mask = self.vec_gen_mask()
    
    def _generate_config_names(self, idx_list: List[int], config_prefix: str) -> List[str]:
        """Generate configuration file names for each environment."""
        base_name = config_prefix[:-5]  # Remove .cfg extension
        extension = config_prefix[-4:]  # Get .cfg extension
        return [f"{base_name}{i}{extension}" for i in idx_list]
    
    def _initialize_vectors(self) -> None:
        """Initialize all vector attributes."""
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

    def _get_single_system(self, idx: int, filename: str) -> None:
        """Load configuration for a single environment system.
        
        Args:
            idx: Environment index
            filename: Configuration file path
        """
        path, intp_mim, chip_mos, NCAP, intp_n, chip_n = config.read_config(filename)
        
        self.vec_path.append(path)
        self.vec_intp_mim.append(intp_mim)
        self.vec_chip_mos.append(chip_mos)
        self.vec_NCAP.append(NCAP)
        self.vec_intp_n.append(intp_n)
        self.vec_chip_n.append(chip_n)
        self.vec_target_imped.append(get_target_imped())
        
        # Load sense impact data
        sense_file = os.path.join(path, 'sense_impact1.txt')
        self.vec_sense.append(np.loadtxt(sense_file))

    def gen_mask(self, env_idx: int) -> Tuple[int, np.ndarray, np.ndarray, List[float]]:
        """Generate masks for interposer and chip capacitors.
        
        Args:
            env_idx: Environment index
            
        Returns:
            Tuple of (env_idx, intp_mask, chip_mask, combined_mask)
        """
        # Initialize masks
        intp_mask = np.ones(NCOL * NROW, dtype=float)
        chip_mask = np.zeros(NCOL * NROW, dtype=float)
        
        # Set interposer mask (0 where capacitors can be placed)
        intp_mask[self.vec_intp_n[env_idx]] = 0
        
        # Set chip mask (1 where capacitors can be placed)
        chip_mask[self.vec_chip_n[env_idx]] = 1
        
        # Reshape to grid format
        intp_mask = intp_mask.reshape(NCOL, NROW)
        chip_mask = chip_mask.reshape(NCOL, NROW)
        
        # Create combined mask
        combined_mask = np.concatenate([intp_mask, chip_mask]).reshape(-1).tolist()
        
        return env_idx, intp_mask, chip_mask, combined_mask

    def vec_gen_mask(self, use_pool: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[float]]]:
        """Generate masks for all environments.
        
        Args:
            use_pool: Whether to use multiprocessing
            
        Returns:
            Tuple of (interposer_masks, chip_masks, combined_masks)
        """
        arg_list = list(range(self.env_count))
        
        if use_pool and self.env_count > 1:
            chunk_size = max(1, self.env_count // SYSTEM_CORES)
            with Pool(min(SYSTEM_CORES, self.env_count)) as pool:
                mask_results = pool.map(self.gen_mask, arg_list, chunk_size)
            
            # Sort results by environment index
            sorted_results = sorted(mask_results, key=lambda x: x[0])
            vec_intp_mask = [r[1] for r in sorted_results]
            vec_chip_mask = [r[2] for r in sorted_results]
            vec_mask = [r[3] for r in sorted_results]
        else:
            # Sequential processing
            results = [self.gen_mask(i) for i in arg_list]
            _, vec_intp_mask, vec_chip_mask, vec_mask = map(list, zip(*results))

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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[float], List[bool], dict]:
        """Execute one environment step with the given actions.
        
        Args:
            action: Array of actions for each environment
            
        Returns:
            Tuple of (observations, impedances, rewards, dones, info)
        """
        # Apply actions to each environment
        for idx in range(self.env_count):
            action_idx = int(action[idx])
            self.vec_cur_params_idx[idx][action_idx] = DEFAULT_CAP_VALUE
            self.vec_dones[idx] = False

        # Get new observations and rewards
        state = self.vec_get_obs()
        cost_negative, total_imped = self.vec_cal_reward(use_pool=True)
        
        # Calculate reward increment
        reward_increment = [current - previous 
                          for current, previous in zip(cost_negative, self.vec_his_reward)]
        self.vec_his_reward = cost_negative

        info = {
            "reward_now": cost_negative, 
            "reward_increment": reward_increment,
            "his_reward": self.vec_his_reward,
        }
        
        self.vec_his_reward = cost_negative

        return np.stack(state), np.stack(total_imped), cost_negative, self.vec_dones, info

    def _generate_spice_params(self, env_idx: int) -> Tuple[str, str]:
        """Generate SPICE parameter strings for decap and ESR values."""
        str_dc = ''
        esrs = ''
        
        # Generate interposer capacitor parameters
        for i, val in enumerate(self.vec_cur_params_idx[env_idx][:self.vec_intp_mim[env_idx]]):
            str_dc += f'.param dcap_int_val{i + 1}={val}p\n'

        # Generate chip capacitor parameters and ESR values
        chip_params = self.vec_cur_params_idx[env_idx][self.vec_intp_mim[env_idx]:]
        for j, val in enumerate(chip_params):
            param_idx = j + 1 + self.vec_intp_mim[env_idx]
            scaled_val = val * MOS_CAP_SCALING
            str_dc += f'.param dcap_int_val{param_idx}={scaled_val:.2f}p\n'
            
            if val != 0:
                esr_val = ESR_SCALING / val
                esrs += f'.param esr{j + 1}={esr_val:.3f}\n'
            else:
                esrs += f'.param esr{j + 1}=0\n'
        
        return str_dc, esrs
    
    def _write_param_files(self, env_path: str, str_dc: str, esrs: str) -> None:
        """Write parameter files for SPICE simulation."""
        try:
            os.makedirs(env_path, exist_ok=True)
            
            with open(os.path.join(env_path, 'int_param_dcap.txt'), 'w') as f:
                f.write(str_dc)
                
            with open(os.path.join(env_path, 'moscap_esr.txt'), 'w') as f:
                f.write(esrs)
        except (OSError, IOError) as e:
            raise IOError(f"Failed to write parameter files to {env_path}: {e}")

    def cal_reward(self, env_idx: int) -> Tuple[int, float, np.ndarray]:
        """Calculate reward for a single environment.
        
        Args:
            env_idx: Environment index
            
        Returns:
            Tuple of (env_idx, reward, impedance_array)
        """
        # Generate parameter strings for SPICE simulation
        str_dc, esrs = self._generate_spice_params(env_idx)
        
        # Write parameter files
        env_path = os.path.join(self.vec_path[env_idx], str(env_idx))
        self._write_param_files(env_path, str_dc, esrs)

        # Run SPICE simulation
        run_os(env_path + '/')

        # Read simulation results
        port_impedances = self._read_port_results(env_path)
        
        # Calculate maximum impedance across all ports
        max_impedances = np.maximum.reduce(port_impedances)
        target_impedances = np.array(self.vec_target_imped[env_idx])
        
        # Calculate impedance violation
        max_violation = max(np.max(max_impedances - target_impedances), 0)
        
        # Calculate impedance differences for all ports
        impedance_diffs = []
        for port_imped in port_impedances:
            impedance_diffs.extend(target_impedances - port_imped)
        all_impedance_vals = np.array(impedance_diffs)

        # Calculate total cost
        total_cost = self._calculate_cost(env_idx, max_violation)

        return env_idx, total_cost, all_impedance_vals
    
    def _read_port_results(self, env_path: str) -> List[np.ndarray]:
        """Read impedance results from all port files."""
        port_impedances = []
        for i in range(1, 5):  # ports 1-4
            port_file = os.path.join(env_path, f'port{i}_impeval.txt')
            port_data = readresult(port_file)
            port_impedances.append(port_data[:, 1])  # impedance values
        return port_impedances
    
    def _calculate_cost(self, env_idx: int, max_violation: float) -> float:
        """Calculate the cost/reward based on impedance violation and capacitor usage."""
        if max_violation == 0:
            # No impedance violation - reward based on capacitor efficiency
            intp_cap_total = sum(val for val in self.vec_cur_params_idx[env_idx][:self.vec_intp_mim[env_idx]] if val > 0)
            chip_cap_total = sum(val * MOS_CAP_SCALING for val in self.vec_cur_params_idx[env_idx][self.vec_intp_mim[env_idx]:] if val > 0)
            
            # Calculate efficiency metrics (higher is better)
            intp_max_capacity = self.vec_intp_mim[env_idx] * DEFAULT_CAP_VALUE
            chip_max_capacity = self.vec_chip_mos[env_idx] * 500  # Different max capacity for chip caps
            
            intp_efficiency = (intp_max_capacity - intp_cap_total) / intp_max_capacity
            chip_efficiency = (chip_max_capacity - chip_cap_total) / chip_max_capacity
            
            total_cost = 0.5 * intp_efficiency + 0.5 * chip_efficiency
        else:
            # Impedance violation penalty
            total_cost = COST_PENALTY * max_violation

        return total_cost

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

    def get_obs(self, env_idx: int) -> Tuple[int, np.ndarray]:
        """Get observation for a single environment.
        
        Args:
            env_idx: Environment index
            
        Returns:
            Tuple of (env_idx, observation)
        """
        # Split parameters into interposer and chip capacitors
        intp_cap = self.vec_cur_params_idx[env_idx][:self.vec_intp_mim[env_idx]]
        chip_cap = self.vec_cur_params_idx[env_idx][self.vec_intp_mim[env_idx]:]
        
        # Normalize capacitor values
        mim_distribution = fill_non_zero(self.vec_intp_mask[env_idx], intp_cap) / DEFAULT_CAP_VALUE
        mos_distribution = fill_non_zero(self.vec_chip_mask[env_idx], chip_cap) / DEFAULT_CAP_VALUE
        
        # Get sense impact data
        sense_cap = self.vec_sense[env_idx].reshape(NCOL, NROW)
        
        # Create observation with 5 channels:
        # 1. Available interposer locations (mask - current placement)
        # 2. Current interposer capacitor distribution
        # 3. Current chip capacitor distribution (repeated for consistency)
        # 4. Current chip capacitor distribution
        # 5. Sense impact data
        obs = np.array([
            self.vec_intp_mask[env_idx] - mim_distribution.astype(bool),
            mim_distribution,
            mos_distribution,
            mos_distribution,
            sense_cap
        ])

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

    def action_mask(self, env_idx: int) -> Tuple[int, np.ndarray]:
        """Generate action mask for a single environment.
        
        Args:
            env_idx: Environment index
            
        Returns:
            Tuple of (env_idx, action_mask)
        """
        mask = np.ones((self.ACTION_LOCATIONS, len(self.ACTION_MEANINGS)), dtype=float)
        
        # Mask out locations that already have capacitors
        for i in range(self.ACTION_LOCATIONS):
            if self.vec_cur_params_idx[env_idx][i] != 0:
                mask[i] = np.zeros(len(self.ACTION_MEANINGS))
        
        return env_idx, mask.reshape(-1)

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
    # Example usage and testing
    env = DecapPlaceParallel([5])  # Single environment for testing
    
    start_time = time.time()
    observations, impedances = env.reset()
    observations, impedances, rewards, dones, info = env.step(np.array([0]))
    end_time = time.time()
    
    print(f"Rewards: {rewards}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")



