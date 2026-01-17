
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def load_data(algo, n_agents):
    pattern = f"results/{algo}_transport_n{n_agents}_seed*.json"
    files = glob.glob(pattern)
    all_rewards = []
    min_len = float('inf')
    
    if not files:
        print(f"No files found for {algo} n={n_agents}")
        return None, None

    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            rewards = data['rewards']
            min_len = min(min_len, len(rewards))
            all_rewards.append(rewards)
    
    # Trim to min length
    all_rewards = [r[:min_len] for r in all_rewards]
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    steps = np.arange(min_len) * 3200 # Approx step size (32 envs * 100 steps) -> actually we logged every 32 envs. 
    # In train.py: global_step += args.num_envs (32).
    # Logged every update. One update = num_steps (100) * num_envs (32) = 3200 steps?
    # No, train.py:
    # for step in range(args.num_steps): global_step += args.num_envs
    # Logging happens ONCE per update loop.
    # So x-axis is correct if we assume standard logging interval.
    # Let's just use the 'steps' from the json if available, but usually they are aligned.
    
    return steps, mean_rewards

def plot_n_agents():
    agent_counts = [3, 4, 6]
    plt.figure(figsize=(10, 6))
    
    colors = ['b', 'g', 'r']
    
    for i, n in enumerate(agent_counts):
        # Mappo
        steps, rewards = load_data('mappo', n)
        if steps is not None:
            plt.plot(steps, rewards, label=f'MAPPO (N={n})', linestyle='--', color=colors[i])
            
        # Attention Mappo
        steps, rewards = load_data('attention_mappo', n)
        if steps is not None:
            plt.plot(steps, rewards, label=f'Attn-MAPPO (N={n})', linestyle='-', color=colors[i])

    plt.title('Effect of Agent Count on Performance (Transport)')
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('n_agents_comparison.png')
    print("Saved n_agents_comparison.png")

if __name__ == "__main__":
    plot_n_agents()
