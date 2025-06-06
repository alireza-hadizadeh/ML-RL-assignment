import numpy as np
import torch.nn as nn

def generate_batch(batch_size, N=20, **kwargs):
    """Return a list of environment instances with new random seeds."""
    seeds = np.random.randint(0, 1e6, size=batch_size)
    from env import DTSPTDS  # local import to avoid circular import
    return [DTSPTDS(N=N, seed=int(s), **kwargs) for s in seeds]

def clones(module, N):
    """Return N deep copies of a module inside a ModuleList."""
    return nn.ModuleList([module for _ in range(N)])
