import torch
import gymnasium as gym
import minigrid
import numpy as np

class MiniGridProcessor(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # The original space is (H, W, C) with dtype uint8
        old_space = env.observation_space["image"]
        # The new space must reflect the permuted shape (C, H, W) and
        # the normalized float data type.
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_space.shape[2], old_space.shape[0], old_space.shape[1]),
            dtype=np.float32,
        )

    def observation(self, obs):
        # Permute and normalize the observation tensor.
        # .contiguous() ensures the tensor has a contiguous memory layout.
        return (
            torch.tensor(obs["image"], dtype=torch.float32)
            .permute(2, 0, 1)
            .contiguous()
            / 255.0
        )

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = MiniGridProcessor(env)
        env.action_space.seed(seed)
        return env

    return thunk