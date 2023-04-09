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
import wandb
import pickle

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

run = wandb.init(
    # set the wandb project where this run will be logged
    project="1v1 new",
    name="1v1 self-play",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-4,
    "architecture": "CNN",
    "epochs": 100,
    'steps': 1e8,
    'activate': 'tanh',
    'ps': 'score reward * 5, angrew1 * 5, angrew0 * 5',
    'entropy_cost':1e-4,
    }
  )



train_sps = []
env = Soccer_field()

def progress(_, metrics):
  if 'training/sps' in metrics:
    train_sps.append(metrics['training/sps'])


with open('1v1 data/8',mode='rb') as file:
    params = file.read()
decoded_params = pickle.loads(params)

env.opp_params = decoded_params[:2]


_, params, metrics = ppo.train(
      env, num_timesteps = 100_000_000,
      num_evals = 100, reward_scaling = 1., episode_length = 1000,
      normalize_observations = True, action_repeat = 2, 
      discounting = 0.99, entropy_cost = 1e-4, unroll_length = 5,
      learning_rate = 1e-4, num_envs = 2048, lr_decay=False,
      batch_size = 1024, progress_fn = progress)

# env.opp_params = params[:2]

# for i in range(10):
#   _, params, metrics = ppo.train(
#       env, num_timesteps = 80_000_000,
#       num_evals = 10, reward_scaling = 1., episode_length = 1000,
#       normalize_observations = True, action_repeat = 2, pre_params=params, num_i=i+1,
#       discounting = 0.99, entropy_cost = 1e-4, unroll_length = 5,
#       learning_rate = 1e-4, num_envs = 2048, lr_decay=False,
#       batch_size = 1024, progress_fn = progress)
#   env.opp_params = params[:2]

print(f'train steps/sec: {np.mean(train_sps[1:])}')
print(metrics)


wandb.finish()