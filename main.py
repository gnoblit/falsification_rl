import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from falsify.training.trainer import Trainer

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Setup seeds and device
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    cfg.device = device

    # Initialize and run the trainer
    trainer = Trainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()