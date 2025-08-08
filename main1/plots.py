import os.path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

def load_data(filename: str) -> np.ndarray:
    """Load numerical data from file."""
    try:
        return np.loadtxt(filename, dtype=np.float32)
    except (FileNotFoundError, IOError) as e:
        raise FileNotFoundError(f"Cannot read file {filename}: {e}")

def smooth_data(y: np.ndarray, window_size: int) -> np.ndarray:
    """Apply moving average smoothing to data."""
    if window_size <= 0:
        return y
    kernel = np.ones(window_size) / window_size
    return np.convolve(y, kernel, mode='same')

def create_loss_plot(data: np.ndarray, title: str, output_path: str, 
                    max_epochs: int = 600, smooth_window: int = 5) -> None:
    """Create and save a loss plot."""
    smoothed_data = smooth_data(data, smooth_window)
    x_values = np.linspace(0, len(smoothed_data) - 1, len(smoothed_data))
    
    plt.rcParams['font.size'] = 20
    fig, ax = plt.subplots()
    plt.xlim(0, max_epochs)
    plt.xlabel('Epoch')
    plt.plot(x_values, smoothed_data, color='orange', linewidth=2.5, 
             label='training from scratch')
    plt.title(title, fontsize=30)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
def main() -> None:
    """Main function to generate all loss plots."""
    # Configuration
    base_path = 'runs/'
    case_path = 'case4/'
    run_path = '202508061025/'
    full_path = os.path.join(base_path, case_path, run_path)
    
    # Plot configurations
    plots_config = [
        ('loss.txt', 'Loss', 'loss.png'),
        ('pgloss.txt', 'Policy Loss', 'policyloss.png'),
        ('vloss.txt', 'Value Loss', 'valueloss.png'),
        ('entloss.txt', 'Entropy Loss', 'entloss.png')
    ]
    
    # Generate all plots
    for filename, title, output_name in plots_config:
        try:
            data_path = os.path.join(full_path, filename)
            output_path = os.path.join(full_path, output_name)
            
            data = load_data(data_path)
            create_loss_plot(data, title, output_path)
            print(f"Generated plot: {output_path}")
            
        except FileNotFoundError as e:
            print(f"Warning: {e}")
        except Exception as e:
            print(f"Error generating plot for {filename}: {e}")

if __name__ == "__main__":
    main()


