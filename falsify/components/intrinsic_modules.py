import torch
import torch.nn as nn
from falsify.models import TheoryModel, FalsifierModel
from falsify.common.data_structures import SequenceBatch

class TheoryModule(nn.Module):
    """A component for the theory model and its loss."""
    def __init__(self, feature_size, action_space, device, args):
        super().__init__()
        self.device = device
        self.model = TheoryModel(
            feature_size, 
            action_space, 
            hidden_size=args.agent.theory_model.hidden_size
        ).to(device)
        self.loss_fn = nn.MSELoss()

    def compute_loss(self, state_feats, next_state_feats_target, actions):
        predicted_next_feats = self.model(state_feats, actions)
        loss = self.loss_fn(predicted_next_feats, next_state_feats_target)
        return loss, {"theory_loss": loss.item()}


class FalsifierModule(nn.Module):
    """A component for the falsifier model, its loss, and intrinsic reward."""
    def __init__(self, feature_extractor, theory_module, action_space, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.feature_extractor = feature_extractor
        self.theory_model = theory_module

        self.falsifier_model = FalsifierModel(
            self.feature_extractor.feature_size, 
            action_space, 
            horizon=args.agent.falsify_horizon,
            hidden_size=args.agent.falsifier_model.hidden_size
        ).to(self.device)
        
        self.theory_error_fn = nn.MSELoss(reduction='none')
        self.falsifier_loss_fn = nn.MSELoss()

    def compute_intrinsic_reward(self, rollouts):
        self.feature_extractor.eval()
        self.falsifier_model.eval()
        with torch.no_grad():
            H = self.args.agent.falsify_horizon
            sequence_batch = rollouts.get_sequences(H)
            if sequence_batch.obs_seq.nelement() == 0:
                return torch.zeros_like(rollouts.rewards)
            
            flat_feats = self.feature_extractor(sequence_batch.obs_seq.flatten(0, 1))
            feats_seq = flat_feats.view(*sequence_batch.obs_seq.shape[:2], -1)
            scores = self.falsifier_model(feats_seq, sequence_batch.action_seq)
            
            padded_scores = torch.zeros(self.args.training.num_steps, self.args.env.num_envs, 1, device=self.device)
            num_valid_seqs = self.args.training.num_steps - H + 1
            padded_scores[:num_valid_seqs] = scores.view(num_valid_seqs, self.args.env.num_envs, 1)
        return padded_scores