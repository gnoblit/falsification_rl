import torch
import gymnasium as gym

class MiniGridProcessor(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space['image']
    
    def observation(self, obs):
        # Permute and normalize the observation tensor.
        # .contiguous() ensures the tensor has a contiguous memory layout.
        return torch.tensor(obs['image'], dtype=torch.float32).permute(2, 0, 1).contiguous() / 255.0

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = MiniGridProcessor(env)
        env.action_space.seed(seed)
        return env
    return thunk