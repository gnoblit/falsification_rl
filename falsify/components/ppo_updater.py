# falsify/components/ppo_updater.py
import torch
from torch.distributions.categorical import Categorical
import numpy as np

def update_ppo(agent, rollouts):
    """
    Performs the PPO update for a given agent and rollout buffer.
    This function is now decoupled from any specific agent class.
    """
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    b_obs = rollouts.obs[:-1].view(-1, *rollouts.obs.size()[2:])
    b_actions = rollouts.actions.view(-1, *rollouts.actions.size()[2:])
    b_logprobs = rollouts.action_log_probs.view(-1)
    b_advantages = advantages.view(-1)
    b_returns = rollouts.returns[:-1].view(-1)

    pg_losses, v_losses, entropy_losses, approx_kls = [], [], [], []
    all_aux_metrics = []

    for epoch in range(agent.args.training.update_epochs):
        b_inds = torch.randperm(b_obs.size(0))
        for start in range(0, b_obs.size(0), agent.args.training.minibatch_size):
            end = start + agent.args.training.minibatch_size
            mb_inds = b_inds[start:end]
            
            # Get PPO-specific values
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions.long()[mb_inds].squeeze(-1)
            )

            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()
            approx_kl = ((ratio - 1) - logratio).mean()

            pg_loss1 = -b_advantages[mb_inds] * ratio
            pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - agent.args.training.clip_coef, 1 + agent.args.training.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
            entropy_loss = entropy.mean()

            # The agent itself is responsible for calculating its auxiliary loss
            aux_loss, aux_metrics = agent.compute_auxiliary_loss(b_obs[mb_inds], rollouts, mb_inds)
            if aux_metrics:
                all_aux_metrics.append(aux_metrics)

            loss = pg_loss - agent.args.training.ent_coef * entropy_loss + v_loss * agent.args.training.vf_coef + aux_loss

            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), agent.args.training.max_grad_norm)
            agent.optimizer.step()

            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            entropy_losses.append(entropy_loss.item())
            approx_kls.append(approx_kl.item())

    losses = {
        "value_loss": np.mean(v_losses),
        "policy_loss": np.mean(pg_losses),
        "entropy_loss": np.mean(entropy_losses),
        "approx_kl": np.mean(approx_kls),
    }

    # Merge the PPO losses with any auxiliary losses from the agent
    if all_aux_metrics:
        # Average metrics across all minibatches in the epoch
        avg_aux_metrics = {k: np.mean([d[k] for d in all_aux_metrics]) for k in all_aux_metrics[0]}
        losses.update(avg_aux_metrics)
        
    return losses