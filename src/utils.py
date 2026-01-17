
import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_dim, action_dim, state_dim=None):
        self.obs = torch.zeros((num_steps, num_envs, obs_dim))
        self.actions = torch.zeros((num_steps, num_envs, action_dim))
        self.logprobs = torch.zeros((num_steps, num_envs))
        self.rewards = torch.zeros((num_steps, num_envs))
        self.dones = torch.zeros((num_steps, num_envs))
        self.values = torch.zeros((num_steps, num_envs))
        
        # specific for MAPPO
        if state_dim:
            self.states = torch.zeros((num_steps, num_envs, state_dim))
        else:
            self.states = None

        self.step = 0
        self.num_steps = num_steps

    def add(self, obs, action, logprob, reward, done, value, state=None):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value.flatten()
        
        if self.states is not None and state is not None:
            self.states[self.step] = state
            
        self.step += 1

    def compute_returns_and_advantages(self, last_value, next_done, gamma=0.99, gae_lambda=0.95):
        returns = torch.zeros_like(self.rewards)
        advantages = torch.zeros_like(self.rewards)
        
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done.float()
                nextvalues = last_value.flatten()
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + self.values
        self.advantages = advantages
        self.returns = returns
        return returns, advantages

    def flatten(self):
        b_obs = self.obs.reshape((-1,) + self.obs.shape[2:])
        b_actions = self.actions.reshape((-1,) + self.actions.shape[2:])
        b_logprobs = self.logprobs.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)
        
        if self.states is not None:
            b_states = self.states.reshape((-1,) + self.states.shape[2:])
            return b_obs, b_states, b_actions, b_logprobs, b_advantages, b_returns, b_values
            
        return b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values
