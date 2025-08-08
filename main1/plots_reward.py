import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

def smooth_data(y: np.ndarray, window_size: int) -> np.ndarray:
    """Apply moving average smoothing with valid mode."""
    if window_size <= 0:
        return y
    kernel = np.ones(window_size) / window_size
    return np.convolve(y, kernel, mode='valid')

def load_data(filename: str) -> np.ndarray:
    """Load numerical data from file."""
    try:
        return np.loadtxt(filename, dtype=np.float32)
    except (FileNotFoundError, IOError) as e:
        raise FileNotFoundError(f"Cannot read file {filename}: {e}")

def calculate_reward_stats(reward_data: np.ndarray, smooth_window: int = 4, 
                          start_idx: int = 5, end_idx: int = 505) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate smoothed mean rewards and standard deviations."""
    mean_rewards = smooth_data(np.mean(reward_data, axis=1), smooth_window)
    
    # Extract the specified range
    if end_idx > len(mean_rewards):
        end_idx = len(mean_rewards)
    
    mean_rewards = mean_rewards[start_idx:end_idx]
    
    # Calculate standard deviations with clipping
    std_devs = np.array([])
    for i in range(start_idx, min(end_idx + start_idx, len(reward_data))):
        if i < len(reward_data):
            std_val = 0.5 * np.std(reward_data[i])
            std_devs = np.append(std_devs, min(std_val, 2.0))  # Clip at 2.0
    
    return mean_rewards, std_devs

def create_reward_comparison_plot(runs_path: str, infer_path: str, output_path: str,
                                 case_name: str = 'case7/', 
                                 runs_timestamp: str = '202410302012',
                                 infer_timestamp: str = '202410301330') -> None:
    """Create a comparison plot of reward curves."""
    try:
        # Load reward data
        runs_file = os.path.join('runs', runs_path, case_name, runs_timestamp, 'reward.txt')
        infer_file = os.path.join('infer', infer_path, case_name, infer_timestamp, 'reward.txt')
        
        r1 = load_data(runs_file)[:600]
        r2 = load_data(infer_file)[:600]
        
        # Calculate statistics
        rm1, rs1 = calculate_reward_stats(r1)
        rm2, rs2 = calculate_reward_stats(r2)
        
        # Create plot
        x = np.linspace(0, len(rm1) - 1, len(rm1))
        
        plt.rcParams['font.size'] = 15
        plt.figure(figsize=(10, 6))
        plt.xlim(0, 500)
        plt.ylabel('Mean Reward')
        plt.xlabel('Epoch')
        
        # Plot lines and confidence intervals
        plt.plot(x, rm1, color='orange', linewidth=1.5, label='training from scratch')
        plt.plot(x, rm2, color='#00BFFF', linewidth=1.5, label='pre-trained')
        plt.fill_between(x, rm1 + rs1, rm1 - rs1, facecolor='moccasin', alpha=0.5)
        plt.fill_between(x, rm2 + rs2, rm2 - rs2, facecolor='#87CEEB', alpha=0.5)
        
        plt.legend(loc='lower right', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Reward comparison plot saved to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error creating reward plot: {e}")

def main() -> None:
    """Main function to create reward comparison plot."""
    create_reward_comparison_plot(
        runs_path='',
        infer_path='',
        output_path='infer/transfer-reward.png'
    )

if __name__ == "__main__":
    main()
