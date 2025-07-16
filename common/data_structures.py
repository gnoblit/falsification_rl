from dataclasses import dataclass
import torch

@dataclass
class SequenceBatch:
    """Data batch for a sequence of observations, actions, and dones."""
    obs_seq: torch.Tensor
    action_seq: torch.Tensor
    done_seq: torch.Tensor

@dataclass
class FalsifierBatch:
    """Training data for the Falsifier model."""
    obs_seq: torch.Tensor
    action_seq: torch.Tensor
    targets: torch.Tensor
    valid_mask: torch.Tensor