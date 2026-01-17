
import vmas
import torch

try:
    env = vmas.make_env(
        scenario="transport",
        num_envs=2,
        device="cpu",
        continuous_actions=True
    )
    print(f"Env created. n_agents: {env.n_agents}")
    obs = env.reset()
    print(f"Reset Obs type: {type(obs)}")
    
    # In VMAS < 1.3 obs might be list of tensors (one per agent). 
    # Let's check.
    if isinstance(obs, list):
        print(f"Obs is list of length: {len(obs)}")
        print(f"Agent 0 obs shape: {obs[0].shape}")
    elif isinstance(obs, torch.Tensor):
        print(f"Obs is tensor of shape: {obs.shape}")
        
    # Actions
    # Agents expect actions. 
    # Transport agents (Sphere) usually have 2D force actions
    actions = []
    for i in range(env.n_agents):
        actions.append(torch.zeros(2, 2)) # (n_envs, action_dim)

    obs, rews, dones, info = env.step(actions)
    print("Step successful.")
    
    if isinstance(rews, list):
        print(f"Rew is list of length: {len(rews)}")
        print(f"Agent 0 rew shape: {rews[0].shape}") # Should be (n_envs,)
    else:
        print(f"Rew is tensor of shape: {rews.shape}")

except Exception as e:
    print(f"Error: {e}")
