import functools
import time


import numpy as np

import brax

from brax import envs
from brax import jumpy as jp
from brax.io import html
from brax.io import image
import jax
from jax import numpy as jnp

from field import Soccer_field
from brax.training.agents.ppo import train as ppo

import wandb
import random


env = Soccer_field()

# rollout = []
# jit_env_reset = jax.jit(env.reset)
# state = jit_env_reset(rng=jp.random_prngkey(seed=0))
# qp = state.qp
# rollout.append(state.qp)

# jit_env_step = jax.jit(env.step)
# for i in range(100):
#   act = jp.concatenate([jnp.ones(2),jnp.zeros(1),jnp.ones(2),jnp.zeros(1)])
#   # act = jp.concatenate([act,jnp.zeros(6)])
#   state = jit_env_step(state, act)
#   rollout.append(state)


# html = html.render(env.sys, rollout)
# with open("output.html", "w") as f:
#     f.write(html)

train_sps = []

def progress(_, metrics):
  if 'training/sps' in metrics:
    train_sps.append(metrics['training/sps'])

_, params, metrics = ppo.train(
    env, num_timesteps = 300_000_000,
    num_evals = 100, reward_scaling = .1, episode_length = 5000,
    normalize_observations = True, action_repeat = 1, unroll_length = 5,
    num_minibatches = 32, num_updates_per_batch = 4, discounting = 0.97,
    learning_rate = 3e-4, entropy_cost = 1e-2, num_envs = 2048,
    batch_size = 1024, progress_fn = progress)

# _, params, metrics = ppo.train(
#     envs.create('ant'), num_timesteps = 300_0,
#     num_evals = 1, reward_scaling = .1, episode_length = 5000,
#     normalize_observations = True, action_repeat = 1, unroll_length = 5,
#     num_minibatches = 32, num_updates_per_batch = 4, discounting = 0.97,
#     learning_rate = 3e-4, entropy_cost = 1e-2, num_envs = 2048,
#     batch_size = 1024, progress_fn = progress)

print(f'train steps/sec: {np.mean(train_sps[1:])}')
print(metrics)