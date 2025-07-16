import torch
import torch.optim as optim

from .base_agent import PPOAgent
from falsify.components import TheoryModule

class CuriosityAgent(PPOAgent):
    def __init__(self, obs_shape, action_space, config):
        super().__init__(obs_shape, action_space, config)
        
        self.theory_module = TheoryModule(
            self.feature_extractor.feature_size, action_space, self.device, self.args
        )
        
        self.policy_optimizer = optim.Adam(self.policy_parameters(), lr=self.args.training.lr, eps=self.args.training.adam_eps)
        self.aux_optimizer = optim.Adam(self.aux_parameters(), lr=self.args.training.aux_lr, eps=self.args.training.adam_eps)

    def train(self):
        super().train()
        self.theory_module.train()

    def compute_intrinsic_reward(self, rollouts):
        self.feature_extractor.eval()
        self.theory_module.model.eval()
        with torch.no_grad():
            obs_flat = rollouts.obs[:-1].view(-1, *rollouts.obs.size()[2:])
            next_obs_flat = rollouts.obs[1:].view(-1, *rollouts.obs.size()[2:])
            actions_flat = rollouts.actions.view(-1, *rollouts.actions.size()[2:])
            
            state_feats = self.feature_extractor(obs_flat)
            next_state_feats_target = self.feature_extractor(next_obs_flat)
            
            predicted_next_feats = self.theory_module.model(state_feats, actions_flat.long())
            error = ((predicted_next_feats - next_state_feats_target)**2).mean(dim=-1)
            
            return error.view(self.args.training.num_steps, self.args.env.num_envs, 1)

    def compute_auxiliary_loss(self, mb_obs, rollouts, mb_inds):
        steps = mb_inds // self.args.env.num_envs
        envs = mb_inds % self.args.env.num_envs
        
        mb_actions = rollouts.actions[steps, envs].long()
        mb_next_obs = rollouts.obs[steps + 1, envs]
        
        with torch.no_grad():
            next_state_feats_target = self.feature_extractor(mb_next_obs)
        
        state_feats = self.feature_extractor(mb_obs)
        
        return self.theory_module.compute_loss(state_feats, next_state_feats_target, mb_actions)

    def aux_parameters(self):
        return self.theory_module.parameters()