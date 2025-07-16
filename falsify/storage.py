# falsify/storage.py
import torch
from common.data_structures import SequenceBatch

class RolloutStorage:
    def __init__(self, num_steps, num_envs, obs_shape, action_space, device):
        self.obs = torch.zeros(num_steps + 1, num_envs, *obs_shape).to(device)
        self.rewards = torch.zeros(num_steps, num_envs, 1).to(device)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1).to(device)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1).to(device)
        self.action_log_probs = torch.zeros(num_steps, num_envs, 1).to(device)
        self.actions = torch.zeros(num_steps, num_envs, 1, dtype=torch.long).to(device)
        self.dones = torch.zeros(num_steps + 1, num_envs, 1).to(device) # Raw done flags
        self.masks = torch.ones(num_steps + 1, num_envs, 1).to(device)
        self.termination_masks = torch.ones(num_steps + 1, num_envs, 1).to(device)

        self.num_steps = num_steps
        self.step = 0
        self.device = device

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.dones = self.dones.to(device)
        self.termination_masks = self.termination_masks.to(device)
        self.device = device

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks, termination_masks):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step + 1].copy_(1.0 - masks) # 'done' is the inverse of the 'not done' mask
        self.masks[self.step + 1].copy_(masks)
        self.termination_masks[self.step + 1].copy_(termination_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        """Prepares the storage for the next rollout."""
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.termination_masks[0].copy_(self.termination_masks[-1])
        self.dones[0].copy_(self.dones[-1])

    def compute_returns(self, next_value, gamma, gae_lambda):
        """Computes returns for the rollout using GAE."""
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.termination_masks[step + 1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def get_sequences(self, sequence_length: int) -> SequenceBatch:
        num_valid_starts = self.num_steps - sequence_length + 1
        if num_valid_starts <= 0:
            return SequenceBatch(torch.empty(0), torch.empty(0), torch.empty(0))
        
        all_obs = [self.obs[i:i + num_valid_starts] for i in range(sequence_length)]
        all_actions = [self.actions[i:i + num_valid_starts] for i in range(sequence_length)]
        all_dones = [self.dones[i:i + num_valid_starts] for i in range(sequence_length)]

        obs_seq = torch.stack(all_obs, dim=0).permute(1, 2, 0, 3, 4, 5).flatten(0, 1)
        actions_seq = torch.stack(all_actions, dim=0).permute(1, 2, 0, 3).flatten(0, 1)
        dones_seq = torch.stack(all_dones, dim=0).permute(1, 2, 0, 3).flatten(0, 1)

        return SequenceBatch(obs_seq=obs_seq, action_seq=actions_seq, done_seq=dones_seq)