
import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def plot_results():
    results_dir = "results"
    files = glob.glob(os.path.join(results_dir, "*.json"))
    
    plt.figure(figsize=(10, 6))
    
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
        
        rewards = data["rewards"]
        steps = data["steps"]
        name = os.path.basename(file).replace(".json", "")
        
        # Smooth
        window = 10
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        smoothed_steps = steps[:len(smoothed_rewards)]
        
        plt.plot(smoothed_steps, smoothed_rewards, label=name)
        
    plt.xlabel("Total Steps")
    plt.ylabel("Average Reward")
    plt.title("VMAS Transport Scenario Training Results")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_plot.png")
    print("Plot saved to comparison_plot.png")

if __name__ == "__main__":
    plot_results()
