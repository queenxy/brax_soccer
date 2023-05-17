from brax import envs
from brax.io import html
import jax
from jax import numpy as jp
from fieldv2 import Soccer_field
from brax.training.agents.ppo import train as ppo
import numpy as np
import wandb
import pickle

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

run = wandb.init(
    # set the wandb project where this run will be logged
    project="braxv2",
    name="1v1 self-play",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-4,
    "architecture": "CNN",
    "epochs": "10",
    'steps': "1e8",
    'selfplay iter': 10,
    'activate': 'tanh',
    'ps': '10 score_reward',
    'entropy_cost':1e-4,
    'reward_scaling': 1,
    'discounting':0.99,
    }
  )

# env_name = 'ant'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
# backend = 'positional'  # @param ['generalized', 'positional', 'spring']

# env = envs.get_environment(env_name=env_name,
#                            backend=backend)

env = Soccer_field()

train_sps = []
def progress(_, metrics):
  if 'training/sps' in metrics:
    train_sps.append(metrics['training/sps'])

_, params, metrics = ppo.train(
      env, num_timesteps=50_000_000, num_evals=10, reward_scaling=1., 
      episode_length=1000, normalize_observations=True, action_repeat=2, 
      unroll_length=5, num_minibatches=32, num_updates_per_batch=4, 
      discounting=0.99, learning_rate=1e-4, entropy_cost=1e-2, num_envs=4096, 
      batch_size=2048, seed=1, progress_fn = progress)

pre_params = params
env.opp_params = pre_params[:2]

for i in range(10):
  _, params, metrics = ppo.train(
      env, num_timesteps = 100_000_000,
      num_evals = 10, reward_scaling = 1., episode_length = 1000,
      normalize_observations = True, action_repeat = 2, pre_params=pre_params, num_i=i+1,
      discounting = 0.99, entropy_cost = 1e-2, unroll_length = 5,
      learning_rate = 1e-4, num_envs = 4096, lr_decay=False,
      batch_size = 2048, seed=i, progress_fn = progress)

  pre_params = params
  env.opp_params = pre_params[:2]


print(f'train steps/sec: {np.mean(train_sps[1:])}')
print(metrics)

wandb.finish()