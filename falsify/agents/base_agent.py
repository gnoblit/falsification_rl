import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

from falsify.models import MiniGridCNN, PolicyValueNet

class PPOAgent:
    """Base PPO agent, now handling its own update logic."""
    def __init__(self, obs_shape, action_space, config):
        self.args = config
        # Create the torch.device object from the string in the config
        self.device = torch.device(config.device)
        
        self.feature_extractor = MiniGridCNN(obs_shape).to(self.device)
        self.policy_value_net = PolicyValueNet(self.feature_extractor.feature_size, action_space).to(self.device)
        
        # Optimizers are now created in the child classes
        self.policy_optimizer = None
        self.aux_optimizer = None

    def to(self, device):
        self.feature_extractor.to(device)
        self.policy_value_net.to(device)
        self.device = device
        return self

    def get_value(self, x):
        features = self.feature_extractor(x)
        return self.policy_value_net(features)[1]

    def get_action_and_value(self, x, action=None):
        features = self.feature_extractor(x)
        logits, value = self.policy_value_net(features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

    def compute_intrinsic_reward(self, rollouts):
        return torch.zeros_like(rollouts.rewards)

    def compute_auxiliary_loss(self, mb_obs, rollouts, mb_inds):
        return torch.tensor(0.0, device=self.device), {}

    def policy_parameters(self):
        return list(self.feature_extractor.parameters()) + list(self.policy_value_net.parameters())
    
    def aux_parameters(self):
        # Child classes must implement this to return parameters for the auxiliary optimizer.
        return []

    def update(self, rollouts):
        if self.policy_optimizer is None:
            raise NotImplementedError("Optimizers not initialized. Please create them in the agent's __init__ method.")

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        b_obs = rollouts.obs[:-1].view(-1, *rollouts.obs.size()[2:])
        b_actions = rollouts.actions.view(-1, *rollouts.actions.size()[2:])
        b_logprobs = rollouts.action_log_probs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = rollouts.returns[:-1].view(-1)

        pg_losses, v_losses, entropy_losses, approx_kls = [], [], [], []
        all_aux_metrics = {}

        for epoch in range(self.args.training.update_epochs):
            b_inds = torch.randperm(b_obs.size(0))
            for start in range(0, b_obs.size(0), self.args.training.minibatch_size):
                end = start + self.args.training.minibatch_size
                mb_inds = b_inds[start:end]
                
                # --- PPO Policy and Value Update ---
                _, newlogprob, entropy, newvalue = self.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds].squeeze(-1)
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()

                pg_loss1 = -b_advantages[mb_inds] * ratio
                pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - self.args.training.clip_coef, 1 + self.args.training.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                ppo_loss = pg_loss - self.args.training.ent_coef * entropy_loss + v_loss * self.args.training.vf_coef

                self.policy_optimizer.zero_grad()
                ppo_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_parameters(), self.args.training.max_grad_norm)
                self.policy_optimizer.step()

                # --- Auxiliary Model Update ---
                # Only update aux models if an aux optimizer is defined
                if self.aux_optimizer:
                    aux_loss, aux_metrics = self.compute_auxiliary_loss(b_obs[mb_inds], rollouts, mb_inds)
                    if aux_metrics:
                        for k, v in aux_metrics.items():
                            if k not in all_aux_metrics:
                                all_aux_metrics[k] = []
                            all_aux_metrics[k].append(v)

                    self.aux_optimizer.zero_grad()
                    aux_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.aux_parameters(), self.args.training.max_grad_norm)
                    self.aux_optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                approx_kls.append(approx_kl)

        losses = {
            "value_loss": np.mean(v_losses),
            "policy_loss": np.mean(pg_losses),
            "entropy_loss": np.mean(entropy_losses),
            "approx_kl": np.mean(approx_kls),
        }
        
        # Average the auxiliary metrics
        if all_aux_metrics:
            avg_aux_metrics = {k: np.mean(v) for k, v in all_aux_metrics.items()}
            losses.update(avg_aux_metrics)
            
        return losses