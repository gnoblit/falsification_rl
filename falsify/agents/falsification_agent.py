import torch
import torch.optim as optim
import numpy as np

from .base_agent import PPOAgent
from falsify.components import TheoryModule, FalsifierModule
from common.data_structures import FalsifierBatch

class FalsificationAgent(PPOAgent):
    def __init__(self, obs_shape, action_space, config):
        super().__init__(obs_shape, action_space, config)
        
        theory_module = TheoryModule(
            self.feature_extractor.feature_size, action_space, self.device, self.args
        )
        self.falsifier_module = FalsifierModule(
            self.feature_extractor, theory_module, action_space, self.args
        )
        
        self.falsifier_data_cache: FalsifierBatch | None = None
        self.update_counter = 0
        
        self.policy_optimizer = optim.Adam(self.policy_parameters(), lr=self.args.training.lr, eps=self.args.training.adam_eps)
        self.aux_optimizer = optim.Adam(self.aux_parameters(), lr=self.args.training.aux_lr, eps=self.args.training.adam_eps)

    def compute_intrinsic_reward(self, rollouts):
        return self.falsifier_module.compute_intrinsic_reward(rollouts)

    def _prepare_falsifier_targets(self, rollouts):
        H = self.args.agent.falsify_horizon
        num_steps = self.args.training.num_steps
        num_envs = self.args.env.num_envs

        with torch.no_grad():
            # Calculate theory error for every step in the rollout
            all_obs = rollouts.obs[:-1].flatten(0, 1)
            all_next_obs = rollouts.obs[1:].flatten(0, 1)
            all_actions = rollouts.actions.flatten(0, 1).long()
            
            state_feats = self.feature_extractor(all_obs)
            next_feats_target = self.feature_extractor(all_next_obs)
            pred_next_feats = self.falsifier_module.theory_model.model(state_feats, all_actions)
            theory_error_per_step = self.falsifier_module.theory_error_fn(pred_next_feats, next_feats_target).mean(dim=-1)
            
            # Get sequences of observations and actions
            sequence_batch = rollouts.get_sequences(H)
            
            # --- PERFORMANCE FIX: Vectorized Future Error Calculation ---
            # Reshape errors and dones for vectorized processing
            error_map = theory_error_per_step.view(num_steps, num_envs)
            done_map = rollouts.dones[1:].view(num_steps, num_envs)

            # Create sliding windows of errors and dones of size H
            error_windows = error_map.unfold(0, H, 1).permute(2, 0, 1) # (H, num_seqs, num_envs)
            done_windows = done_map.unfold(0, H, 1).permute(2, 0, 1)

            # Create a mask that zeros out errors after the first 'done' in each window
            # cumsum will be > 0 for all elements after the first 'done'
            future_mask = (torch.cumsum(done_windows, dim=0) == 0).float()
            
            # A sequence is only valid for training if it does not contain a 'done' at all
            valid_seq_mask = (future_mask.prod(dim=0) == 1).flatten() # (num_seqs * num_envs)

            # Calculate the sum of future errors, masked by dones
            actual_future_errors = (error_windows * future_mask).sum(dim=0).flatten()
            # --- END PERFORMANCE FIX ---

        if sequence_batch.obs_seq.nelement() > 0:
            # Normalize targets (z-score) for stability
            mean = actual_future_errors.mean()
            std = actual_future_errors.std()
            targets = (actual_future_errors - mean) / (std + 1e-8)
            
            self.falsifier_data_cache = FalsifierBatch(
                obs_seq=sequence_batch.obs_seq,
                action_seq=sequence_batch.action_seq,
                targets=targets.unsqueeze(1),
                valid_mask=valid_seq_mask.unsqueeze(1)
            )
        else:
            self.falsifier_data_cache = None

    def compute_auxiliary_loss(self, mb_obs, rollouts, mb_inds):
        steps = mb_inds // self.args.env.num_envs
        envs = mb_inds % self.args.env.num_envs
        mb_actions = rollouts.actions[steps, envs].long()
        with torch.no_grad():
            next_state_feats_target = self.feature_extractor(rollouts.obs[steps + 1, envs])
        state_feats = self.feature_extractor(mb_obs)
        theory_loss, theory_metrics = self.falsifier_module.theory_model.compute_loss(state_feats, next_state_feats_target, mb_actions)

        falsifier_loss = torch.tensor(0.0, device=self.device)
        falsifier_metrics = {"falsifier_loss": 0.0}
        if self.falsifier_data_cache and self.falsifier_data_cache.valid_mask.sum() > 0:
            cache = self.falsifier_data_cache
            
            # Sample only from valid sequences that don't cross episode boundaries
            valid_indices = cache.valid_mask.squeeze().nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                # Create a minibatch of sequences by sampling from the valid ones
                sample_indices = valid_indices[torch.randint(0, len(valid_indices), (len(mb_inds),))]

                flat_feats = self.feature_extractor(cache.obs_seq[sample_indices].flatten(0, 1))
                feats_seq = flat_feats.view(len(mb_inds), self.args.agent.falsify_horizon, -1)
                predicted_scores = self.falsifier_module.falsifier_model(feats_seq, cache.action_seq[sample_indices])
                
                # The loss is only computed on the valid, sampled sequences.
                falsifier_loss = self.falsifier_module.falsifier_loss_fn(predicted_scores, cache.targets[sample_indices].squeeze())
                falsifier_metrics["falsifier_loss"] = falsifier_loss.item()

        total_aux_loss = (self.args.agent.theory_loss_coef * theory_loss) + \
                         (self.args.agent.falsifier_loss_coef * falsifier_loss)
                         
        return total_aux_loss, {**theory_metrics, **falsifier_metrics}

    def update(self, rollouts):
        self.update_counter += 1
        if self.update_counter % self.args.agent.falsify_update_freq == 0:
            self._prepare_falsifier_targets(rollouts)
        else:
            self.falsifier_data_cache = None
            
        return super().update(rollouts)

    def aux_parameters(self):
        return list(self.falsifier_module.parameters())