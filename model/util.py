"""Generic utilities useful for the agent code."""
import torch

if torch.cuda.device_count() >= 1:
    DEVICE: torch.device = torch.device('cuda')
else:
    DEVICE: torch.device = torch.device('cpu')
