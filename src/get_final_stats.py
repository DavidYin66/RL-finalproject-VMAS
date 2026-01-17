
import json
import glob
import numpy as np
import os

def get_stats(pattern):
    files = glob.glob(pattern)
    final_rewards = []
    
    if not files:
        return "N/A"

    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            # Take average of last 10 updates (approx last 10% of training)
            rewards = data['rewards'][-10:] 
            final_rewards.append(np.mean(rewards))
            
    if not final_rewards:
        return "N/A"
        
    mean = np.mean(final_rewards)
    std = np.std(final_rewards)
    return f"{mean:.2f} \pm {std:.2f}"

print("Algorithm & N=3 & N=4 & N=6 \\\\")
print("\\midrule")

algos = ["ippo", "mappo", "attention_mappo", "cppo"]
algo_names = {"ippo": "IPPO", "mappo": "MAPPO", "attention_mappo": "Attn-MAPPO", "cppo": "CPPO"}

for algo in algos:
    row = f"{algo_names[algo]}"
    for n in [3, 4, 6]:
        # Handle naming inconsistency: N=4 might be saved without _n4
        if n == 4 and algo != "attention_mappo" and not glob.glob(f"results/{algo}_transport_n4_seed*.json"):
             # Fallback to legacy naming for N=4 if specific file not found
             # Actually, we should check if the legacy file exists
             if glob.glob(f"results/{algo}_transport_seed*.json"):
                  stats = get_stats(f"results/{algo}_transport_seed*.json")
             else:
                  stats = get_stats(f"results/{algo}_transport_n{n}_seed*.json")
        else:
             stats = get_stats(f"results/{algo}_transport_n{n}_seed*.json")
             
        row += f" & {stats}"
    print(row + " \\\\")
