# falsify/training/trainer.py
import time
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from falsify.storage import RolloutStorage
from falsify.agents import PPOAgent, CuriosityAgent, FalsificationAgent
from falsify.envs import make_env

class Trainer:
    def __init__(self, config: DictConfig):
        self.args = config
        self.run_name = f"{self.args.env.env_id}__{self.args.agent.agent}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        
        # Log hyperparameters by flattening the config
        flat_config = OmegaConf.to_container(self.args, resolve=True, throw_on_missing=True)
        # A simple flattening for logging
        flat_dict = {}
        for k, v in flat_config.items():
            if isinstance(v, dict):
                for ik, iv in v.items():
                    flat_dict[f"{k}.{ik}"] = iv
            else:
                flat_dict[k] = v
        self.writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in flat_dict.items()])))

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.args.device = self.device
        
        self.envs = SyncVectorEnv([make_env(self.args.env.env_id, self.args.seed + i) for i in range(self.args.env.num_envs)])
        
        obs_shape = self.envs.single_observation_space.shape
        action_space = self.envs.single_action_space

        agent_map = {"ppo": PPOAgent, "curiosity": CuriosityAgent, "falsification": FalsificationAgent}
        agent_class = agent_map[self.args.agent.agent]
        self.agent = agent_class(obs_shape, action_space, self.args).to(self.device)
        self.rollouts = RolloutStorage(self.args.training.num_steps, self.args.env.num_envs, obs_shape, action_space, self.device)

    def run(self):
        ep_info_buffer = deque(maxlen=self.args.training.ep_info_buffer_size)
        start_time = time.time()
        num_updates = self.args.total_timesteps // (self.args.training.num_steps * self.args.env.num_envs)

        next_obs, _ = self.envs.reset(seed=self.args.seed)
        self.rollouts.obs[0].copy_(torch.Tensor(next_obs).to(self.device))

        for update in tqdm(range(1, num_updates + 1), desc="Training Updates"):
            global_step_base = (update - 1) * self.args.training.num_steps * self.args.env.num_envs
            
            self.agent.feature_extractor.eval()
            self.agent.policy_value_net.eval()
            
            for step in range(self.args.training.num_steps):
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(self.rollouts.obs[step])
                
                next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                
                self.rollouts.insert(
                    torch.Tensor(next_obs).to(self.device), 
                    action, 
                    logprob, 
                    value, 
                    torch.tensor(reward).float().view(-1, 1).to(self.device), 
                    torch.tensor(1.0 - done).float().view(-1, 1).to(self.device),
                    torch.tensor(1.0 - terminated).float().view(-1, 1).to(self.device)
                )

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            ep_info_buffer.append(info["episode"])
            
            intrinsic_rewards = self.agent.compute_intrinsic_reward(self.rollouts)
            if "intrinsic_coef" in self.args.agent:
                # Normalize the rewards for the batch
                batch_mean = intrinsic_rewards.mean()
                batch_std = intrinsic_rewards.std()
                normalized_rewards = (intrinsic_rewards - batch_mean) / (batch_std + 1e-8)

                clipped_intrinsic = torch.clamp(
                    normalized_rewards,
                    self.args.agent.intrinsic_clip_min, 
                    self.args.agent.intrinsic_clip_max
                ) * self.args.agent.intrinsic_coef
                
                self.rollouts.rewards += clipped_intrinsic
                self.writer.add_scalar("charts/avg_intrinsic_reward", clipped_intrinsic.mean().item(), global_step_base)
            
            self.agent.train()
            with torch.no_grad():
                next_value = self.agent.get_value(self.rollouts.obs[-1])
            
            self.rollouts.compute_returns(next_value, self.args.training.gamma, self.args.training.gae_lambda)

            loss_dict = self.agent.update(self.rollouts)
            self.log_losses(loss_dict, global_step_base)

            self.rollouts.after_update()

            if len(ep_info_buffer) > 0:
                self.log_episode_info(ep_info_buffer, global_step_base)
            
            sps = int((self.args.training.num_steps * self.args.env.num_envs * update) / (time.time() - start_time))
            self.writer.add_scalar("charts/SPS", sps, global_step_base)

        self.envs.close()
        self.writer.close()

    def log_losses(self, losses: dict[str, float], global_step: int):
        for name, value in losses.items():
            self.writer.add_scalar(f"losses/{name}", value, global_step)

    def log_episode_info(self, ep_info_buffer, global_step):
        avg_return = np.mean([ep["r"] for ep in ep_info_buffer])
        avg_length = np.mean([ep["l"] for ep in ep_info_buffer])
        self.writer.add_scalar("charts/episodic_return", avg_return, global_step)
        self.writer.add_scalar("charts/episodic_length", avg_length, global_step)