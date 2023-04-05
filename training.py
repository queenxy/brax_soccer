import numpy as np


from brax import envs
from brax import jumpy as jp
from brax.io import html
from brax.io import image
import jax
from jax import numpy as jnp

from field import Soccer_field
from brax.training.agents.ppo import train as ppo
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

train_sps = []
env = Soccer_field()

def progress(_, metrics):
  if 'training/sps' in metrics:
    train_sps.append(metrics['training/sps'])

_, params, metrics = ppo.train(
    env, num_timesteps = 100_000_000,
    num_evals = 100, reward_scaling = 1., episode_length = 1000,
    normalize_observations = True, action_repeat = 2, 
    discounting = 0.99, entropy_cost = 1e-4, unroll_length = 5,
    learning_rate = 1e-4, num_envs = 2048, lr_decay=False,
    batch_size = 1024, progress_fn = progress)

print(f'train steps/sec: {np.mean(train_sps[1:])}')
print(metrics)