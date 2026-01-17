
import torch
import torch.nn as nn
import torch.optim as optim
from modules import Actor, Critic, AttentionCritic

class PPOAgent:
    def __init__(self, obs_dim, action_dim, state_dim=None, lr=3e-4, gamma=0.99, clip_coef=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, algo="ippo", device="cpu"):
        self.algo = algo
        self.device = device
        self.gamma = gamma
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.actor = Actor(obs_dim, action_dim).to(device)
        if algo == "ippo":
            self.critic = Critic(obs_dim).to(device)
        elif algo == "mappo":
            # State dim usually includes all agents' obs
            self.critic = Critic(state_dim).to(device)
        elif algo == "attention_mappo":
            # Critic takes obs of all agents
            self.critic = AttentionCritic(obs_dim).to(device)
        elif algo == "cppo":
            # Centralized Policy: Actor takes state, Critic takes state
            # Here we assume CPPO means Centralized Actor + Centralized Critic
            # obs_dim passed here should be total_obs_dim
            # action_dim passed here should be total_action_dim
            self.actor = Actor(obs_dim, action_dim).to(device) # Overwrite actor created above
            self.critic = Critic(obs_dim).to(device)

        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

    def get_action_and_value(self, obs, state=None, action=None):
        if self.algo == "ippo":
            return self.actor.get_action_and_value(obs, action) + (self.critic(obs),)
        elif self.algo == "mappo":
            # Actor sees local obs, Critic sees global state
            act, logprob, entropy = self.actor.get_action_and_value(obs, action)
            value = self.critic(state)
            return act, logprob, entropy, value
        elif self.algo == "attention_mappo":
            # Actor sees local obs, Critic sees obs stack (passed as state)
            act, logprob, entropy = self.actor.get_action_and_value(obs, action)
            value = self.critic(state) # state here is (N, A, D)
            return act, logprob, entropy, value
        elif self.algo == "cppo":
            # Actor and Critic see global state (passed as obs)
            return self.actor.get_action_and_value(obs, action) + (self.critic(obs),)

    def get_value(self, obs, state=None):
        if self.algo == "ippo":
            return self.critic(obs)
        elif self.algo == "mappo":
            return self.critic(state)
        elif self.algo == "attention_mappo":
            return self.critic(state)
        elif self.algo == "cppo":
            return self.critic(obs)
    
    def update(self, b_obs, b_actions, b_logprobs, b_returns, b_advantages, b_values, b_states=None):
        # b_obs: (batch_size, obs_dim)
        # b_states: (batch_size, state_dim) - for MAPPO
        # For Attention MAPPO, b_states will be (batch_size, n_agents, obs_dim)
        
        # Get current policy outputs
        if self.algo == "ippo":
             _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs, action=b_actions)
        elif self.algo == "mappo":
             _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs, state=b_states, action=b_actions)
        elif self.algo == "attention_mappo":
             _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs, state=b_states, action=b_actions)
        elif self.algo == "cppo":
             _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs, action=b_actions)

        logratio = newlogprob - b_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            # old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            # clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        v_loss_unclipped = (newvalue - b_returns) ** 2
        v_clipped = b_values + torch.clamp(
            newvalue - b_values,
            -self.clip_coef,
            self.clip_coef,
        )
        v_loss_clipped = (v_clipped - b_returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return pg_loss.item(), v_loss.item(), entropy_loss.item(), approx_kl.item()
