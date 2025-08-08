import torch
from typing import Tuple

class RolloutStorage:
    def __init__(self, num_steps: int, num_envs: int, obs_shape: Tuple, action_shape: Tuple, action_space):
        """Initialize rollout storage for PPO training.
        
        Args:
            num_steps: Number of steps per rollout
            num_envs: Number of parallel environments
            obs_shape: Shape of observations
            action_shape: Shape of actions
            action_space: Action space specification
        """
        self.obs = torch.zeros((num_steps + 1, num_envs) + obs_shape)
        self.imped = torch.zeros((num_steps + 1, num_envs) + (231*4,))  # Impedance data shape
        self.actions = torch.zeros((num_steps, num_envs) + action_shape)
        self.logprobs = torch.zeros((num_steps, num_envs))
        self.rewards = torch.zeros((num_steps, num_envs))
        self.dones = torch.zeros((num_steps, num_envs))
        self.values = torch.zeros((num_steps, num_envs))
        self.advantages = torch.zeros((num_steps, num_envs))
        self.returns = torch.zeros((num_steps, num_envs))
        self.action_masks = torch.zeros((num_steps + 1, num_envs) + (action_space.sum(),))


    def to(self, device: torch.device) -> None:
        """Move all tensors to the specified device."""
        self.obs = self.obs.to(device)
        self.imped = self.imped.to(device)
        self.actions = self.actions.to(device)
        self.logprobs = self.logprobs.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.values = self.values.to(device)
        self.advantages = self.advantages.to(device)
        self.returns = self.returns.to(device)
        self.action_masks = self.action_masks.to(device)

    def insert(self, step: int, obs: torch.Tensor, imped: torch.Tensor, 
               actions: torch.Tensor, logprobs: torch.Tensor, rewards: torch.Tensor, 
               dones: torch.Tensor, values: torch.Tensor, action_masks: torch.Tensor) -> None:
        """Insert data for a single step."""
        self.obs[step + 1].copy_(obs)
        self.imped[step + 1].copy_(imped)
        self.actions[step].copy_(actions)
        self.logprobs[step].copy_(logprobs)
        self.values[step].copy_(values)
        self.rewards[step].copy_(rewards)
        self.dones[step].copy_(dones)
        self.action_masks[step + 1].copy_(action_masks)

    def after_update(self) -> None:
        """Copy final step data to initial step for next rollout."""
        self.obs[0].copy_(self.obs[-1])
        self.imped[0].copy_(self.imped[-1])
        self.action_masks[0].copy_(self.action_masks[-1])

    def compute_returns(self, num_steps: int, use_gae: bool, next_value: torch.Tensor, 
                       next_done: torch.Tensor, gamma: float, gae_lambda: float,
                       values: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and advantages using GAE or standard method.
        
        Args:
            num_steps: Number of steps in rollout
            use_gae: Whether to use Generalized Advantage Estimation
            next_value: Value of next state
            next_done: Whether next state is terminal
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            values: State values
            rewards: Rewards
            dones: Terminal flags
            
        Returns:
            Tuple of (advantages, returns)
        """
        returns = torch.zeros_like(values)
        advantages = torch.zeros_like(values)
        
        if use_gae:
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values
        else:
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                
                returns[t] = rewards[t] + gamma * nextnonterminal * next_return
            
            advantages = returns - values

        return advantages, returns
