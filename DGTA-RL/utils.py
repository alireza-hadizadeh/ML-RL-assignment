# dgta_rl/utils.py
import numpy as np

def generate_batch(batch_size, N=20, **kwargs):
    """Return a list of environment instances with new random seeds."""
    seeds = np.random.randint(0, 1e6, size=batch_size)
    from env import DTSPTDS
    return [DTSPTDS(N=N, seed=int(s), **kwargs) for s in seeds]
