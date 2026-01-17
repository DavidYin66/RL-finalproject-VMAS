
import os
import argparse
import time
import json
import torch
import numpy as np
import vmas
from ppo import PPOAgent
from utils import RolloutBuffer

def make_env(scenario, num_envs, device, n_agents):
    return vmas.make_env(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=True,
        n_agents=n_agents
    )

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Create Env
    env = make_env(args.scenario, args.num_envs, device, args.num_agents)
    model_name = f"{args.algo}_{args.scenario}_n{args.num_agents}_seed{args.seed}"
    
    # Dimensions
    # Perform a reset to get obs shape
    dummy_obs = env.reset()
    if isinstance(dummy_obs, list):
        obs_dim = dummy_obs[0].shape[-1]
    else:
        obs_dim = dummy_obs.shape[-1]
        
    # Transport scenario agents use forces (2D)
    # For independent agents, action_dim is 2.
    action_dim = 2 
    
    num_agents = env.n_agents
    print(f"Agents: {num_agents}, Obs Dim: {obs_dim}, Action Dim: {action_dim}")

    # State Dim for MAPPO/CPPO
    # Concatenate all obs
    state_dim = obs_dim * num_agents
    
    # Initialize Agent
    # For CPPO, we treat the system as one big agent
    if args.algo == "cppo":
        agent = PPOAgent(
            obs_dim=state_dim, # Policy takes global state
            action_dim=action_dim * num_agents, # Outputs all actions
            lr=args.lr,
            gamma=args.gamma,
            algo="cppo",
            device=device
        )
    else:
        # IPPO and MAPPO share weights across agents
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim if args.algo == "mappo" else None,
            lr=args.lr,
            gamma=args.gamma,
            algo=args.algo,
            device=device
        )

    # Buffer
    # For CPPO: single buffer for the "central" agent
    # For IPPO/MAPPO: Buffer stores data for ALL agents (obs: [steps, envs, n_agents, obs_dim]?)
    # To keep it simple, we can flatten agents into the batch dimension for IPPO/MAPPO?
    # BUT, MAPPO needs consistent state for all agents. 
    # Let's keep agents separate or use a buffer that handles n_agents.
    # Our RolloutBuffer is simple. Let's create one buffer per agent? Or one large buffer?
    # VMAS is vectorized (num_envs). 
    # Let's say we have effectively num_envs * num_agents experiences per step for IPPO.
    
    # Simplified approach: Treat (num_envs * num_agents) as batch for IPPO.
    # For MAPPO: Same, but Critic uses State.
    # For CPPO: Treat (num_envs) as batch.
    
    effective_num_envs = args.num_envs * num_agents if args.algo != "cppo" else args.num_envs
    effective_obs_dim = obs_dim if args.algo != "cppo" else state_dim
    effective_act_dim = action_dim if args.algo != "cppo" else action_dim * num_agents
    
    buffer = RolloutBuffer(
        args.num_steps, 
        effective_num_envs, 
        effective_obs_dim, 
        effective_act_dim, 
        state_dim=state_dim if (args.algo == "mappo" or args.algo == "attention_mappo") else None
    )

    # Training Loop
    obs_list = env.reset() # List of tensors
    # Convert to appropriate format
    # IPPO/MAPPO: stack to (num_envs, num_agents, obs_dim) -> flatten to (num_envs*num_agents, obs_dim) 
    # CPPO: stack to (num_envs, num_agents*obs_dim)
    
    global_step = 0
    start_time = time.time()
    
    results = {"rewards": [], "steps": []}

    for update in range(1, args.total_timesteps // (args.num_steps * args.num_envs) + 1):
        
        # Anneal LR
        frac = 1.0 - (update - 1.0) / (args.total_timesteps // (args.num_steps * args.num_envs))
        lrnow = frac * args.lr
        agent.optimizer.param_groups[0]["lr"] = lrnow

        # Rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            
            # Prepare Obs
            # obs_list: [ (N, D), (N, D), ... ]
            obs_stack = torch.stack(obs_list, dim=1) # (N, A, D)
            
            if args.algo == "mappo":
                # State is (N, A, N*D) -> Global state
                global_state = obs_stack.reshape(args.num_envs, -1) # (N, A*D)
                curr_state = global_state.unsqueeze(1).repeat(1, num_agents, 1).view(-1, state_dim)
            elif args.algo == "attention_mappo":
                # State is (N, A, D) -> preserve agent dimension for attention
                curr_state = obs_stack # (N, A, D)
            else:
                curr_state = None

            if args.algo == "attention_mappo":
                 # Actor needs flattened obs
                 curr_obs = obs_stack.view(-1, obs_dim)
                 
                 with torch.no_grad():
                     # self.actor(obs) -> (N*A, act)
                     # self.critic(state) -> (N, 1)
                     # We need to broadcast value to (N*A, 1)
                     action, logprob, _, value_global = agent.get_action_and_value(curr_obs, state=curr_state)
                     # value_global is (N, 1).
                     value = value_global.repeat(1, num_agents).view(-1, 1)
                     
                 actions_reshaped = action.view(args.num_envs, num_agents, action_dim)
                 env_actions = [actions_reshaped[:, i, :] for i in range(num_agents)]
                 
                 buffer_obs = curr_obs
                 buffer_act = action
                 buffer_val = value # (N*A, 1) flattened in buffer.add
                 buffer_logprob = logprob
                 
                 # Store state for buffer
                 # Buffer state expects (batch, state_dim).
                 # Here state is (N, A, D). 
                 # We can store it as (N, A*D) flattened? No, attention needs structure.
                 # Buffer handles tensors.
                 # If we modify buffer to accept arbitrary state shape?
                 # RolloutBuffer: self.states = zeros(..., state_dim).
                 # If state_dim is int, it assumes 1D state.
                 # We passed state_dim = obs_dim * num_agents earlier.
                 # Maybe we can flatten obs_stack to (N, A*D) for storage, and unflatten in update?
                 # Yes.
                 buffer_state = obs_stack.reshape(args.num_envs, -1)
            elif args.algo == "cppo":
                 # ... existing cppo block ...
                 pass # handled below but logic flow needs careful merge.
                 
            # Re-organize this block to avoid duplication
            if args.algo == "cppo":
                 # ... (existing CPPO logic) ...
                 curr_obs = obs_stack.reshape(args.num_envs, -1) # (N, A*D)
                 with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(curr_obs)
                 actions_list = []
                 for i in range(num_agents):
                    actions_list.append(action[:, i*action_dim : (i+1)*action_dim])
                
                 env_actions = actions_list
                 buffer_obs = curr_obs
                 buffer_act = action
                 buffer_val = value
                 buffer_logprob = logprob
                 buffer_state = None

            elif args.algo == "attention_mappo":
                 # Actor needs flattened obs
                 curr_obs = obs_stack.view(-1, obs_dim) # (N*A, D)
                 curr_state = obs_stack # (N, A, D)
                 
                 with torch.no_grad():
                     action, logprob, _, value_global = agent.get_action_and_value(curr_obs, state=curr_state)
                     value = value_global.repeat(1, num_agents).view(-1, 1) 
                 
                 actions_reshaped = action.view(args.num_envs, num_agents, action_dim)
                 env_actions = [actions_reshaped[:, i, :] for i in range(num_agents)]
                 buffer_obs = curr_obs
                 buffer_act = action
                 buffer_val = value
                 buffer_logprob = logprob
                 
                 # Store state for buffer
                 # Flatten state to (N*A, A*D)
                 global_state = obs_stack.reshape(args.num_envs, -1)
                 buffer_state = global_state.unsqueeze(1).repeat(1, num_agents, 1).view(-1, state_dim)
                 
            else:
                # IPPO / MAPPO
                curr_obs = obs_stack.view(-1, obs_dim)
                if args.algo == "mappo":
                    global_state = obs_stack.reshape(args.num_envs, -1)
                    curr_state = global_state.unsqueeze(1).repeat(1, num_agents, 1).view(-1, state_dim)
                else:
                    curr_state = None
                
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(curr_obs, state=curr_state)
                
                actions_reshaped = action.view(args.num_envs, num_agents, action_dim)
                env_actions = [actions_reshaped[:, i, :] for i in range(num_agents)]
                
                buffer_obs = curr_obs
                buffer_act = action
                buffer_val = value
                buffer_logprob = logprob
                buffer_state = curr_state

            # Step Env
            # Clamp actions to [-1, 1] as VMAS expects valid range
            env_actions_clipped = [torch.clamp(a, -1.0, 1.0) for a in env_actions]
            next_obs_list, rews_list, dones_list, _ = env.step(env_actions_clipped)
            
            # Process Rewards/Dones
            # rews_list: [ (N,), ... ]
            if isinstance(rews_list, list):
                rews_stack = torch.stack(rews_list, dim=1) # (N, A)
            else:
                rews_stack = rews_list # Assuming it's already stacked or single tensor

            if isinstance(dones_list, list):
                dones_stack = torch.stack(dones_list, dim=1) # (N, A)
            else:
                # It's a single tensor (N,) -> repeat to (N, A)
                dones_stack = dones_list.unsqueeze(1).repeat(1, num_agents)

            if args.algo == "cppo":
                 # Use first agent's done/reward (assuming shared)
                 buffer_done = dones_stack[:, 0].float()
                 buffer_rew = rews_stack[:, 0] 
            else:
                 # Flatten
                 buffer_rew = rews_stack.view(-1)
                 buffer_done = dones_stack.reshape(-1).float()
            
            # Add to buffer
            buffer.add(buffer_obs, buffer_act, buffer_logprob, buffer_rew, buffer_done, buffer_val, state=buffer_state)
            
            obs_list = next_obs_list

        # Compute Returns
        with torch.no_grad():
             obs_stack = torch.stack(obs_list, dim=1)
             if args.algo == "cppo":
                 next_obs = obs_stack.reshape(args.num_envs, -1)
                 next_val = agent.get_value(next_obs)
                 
                 # Prepare next_done
                 if isinstance(dones_list, list):
                     next_done = dones_list[0] # (N,)
                 else:
                     next_done = dones_list # (N,)

             else:
                 next_obs = obs_stack.view(-1, obs_dim)
                 if args.algo == "mappo":
                     global_state = obs_stack.reshape(args.num_envs, -1)
                     next_state = global_state.unsqueeze(1).repeat(1, num_agents, 1).view(-1, state_dim)
                 elif args.algo == "attention_mappo":
                     next_state = obs_stack # (N, A, D)
                 else:
                     next_state = None
                
                 if args.algo == "attention_mappo":
                      next_val_global = agent.get_value(next_obs, state=next_state) # (N, 1)
                      next_val = next_val_global.repeat(1, num_agents).view(-1)
                 else:
                      next_val = agent.get_value(next_obs, state=next_state)
                
                 # Flatten done
                 if isinstance(dones_list, list):
                      next_done = torch.stack(dones_list, dim=1).view(-1)
                 else:
                      next_done = dones_list.unsqueeze(1).repeat(1, num_agents).reshape(-1)

        returns, advantages = buffer.compute_returns_and_advantages(next_val, next_done, gamma=args.gamma)
        
        # Update
        if args.algo == "mappo" or args.algo == "attention_mappo":
             b_obs, b_states, b_actions, b_logprobs, b_advantages, b_returns, b_values = buffer.flatten()
             if args.algo == "attention_mappo":
                 # Reshape b_states back to (Batch, A, D) for Attention Critic
                 b_states = b_states.view(-1, num_agents, obs_dim)
        else:
             b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values = buffer.flatten()
             b_states = None
             
        pg_loss, v_loss, ent_loss, approx_kl = agent.update(b_obs, b_actions, b_logprobs, b_returns, b_advantages, b_values, b_states)
        
        # Logging
        # ... existing logging ...
        avg_rew = buffer.rewards.sum() / args.num_envs # Total reward per env per rollout
        print(f"Update {update}, Step {global_step}: Reward={avg_rew.item():.4f}, PG Loss={pg_loss:.4f}, V Loss={v_loss:.4f}")
        results["rewards"].append(avg_rew.item())
        results["steps"].append(global_step)
        
        # Reset buffer
        buffer.step = 0

    # Save Results
    os.makedirs("results", exist_ok=True)
    with open(f"results/{model_name}.json", "w") as f:
        json.dump(results, f)
    
    torch.save(agent, f"results/{model_name}.pt")
    print("Training Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--algo", type=str, default="ippo", choices=["ippo", "mappo", "cppo", "attention_mappo"])
    parser.add_argument("--scenario", type=str, default="transport")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=100) # Steps per rollout
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    
    args = parser.parse_args()
    train(args)
