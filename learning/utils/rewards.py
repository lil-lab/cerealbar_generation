import os
import numpy as np
from typing import Dict, List

import torch

# reward functions
def negative_one(data: Dict):
    return torch.tensor([-1.])

def positive_one(data: Dict):
    return torch.tensor([1.])

# reward scaling functions
def clamping(rewards, configs):
    max_val, min_val = configs["max_val"], configs["min_val"]
    return np.clip(rewards, min_val, max_val)

def linear_scaling(rewards: List[torch.tensor], configs):
    rewards = torch.tensor(rewards)
    max_val, min_val = configs["max_val"], configs["min_val"]
    max_rewards = torch.max(rewards)
    min_rewards = torch.min(rewards)
    orig_scale = max_rewards - min_rewards
    scale = max_val - min_val
    mul =  scale / orig_scale
    mul = 1 if orig_scale == 0. else mul # todo: is this correct?
    scaled_rewards = rewards * mul
    add = min_val - torch.min(scaled_rewards)
    scaled_rewards += add
    scaled_rewards = [scaled_rewards[i].unsqueeze(0) for i in range(len(scaled_rewards))] # converting tensors to the list of tensors
    
    return scaled_rewards 