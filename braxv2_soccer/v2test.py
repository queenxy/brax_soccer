from brax import envs
from brax.io import html
import jax
from jax import numpy as jp
from fieldv2 import Soccer_field
from brax.training.agents.ppo import train as ppo
import numpy as np


import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# env_name = 'ant'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
# backend = 'positional'  # @param ['generalized', 'positional', 'spring']

# env = envs.get_environment(env_name=env_name,
#                            backend=backend)

env = Soccer_field()
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))
rollout = []
rollout.append(state.pipeline_state)

jit_env_step = jax.jit(env.step)
for i in range(500):
  act = jp.array([1,0])
  state = jit_env_step(state, act)
  print(state.metrics['reward'])
  rollout.append(state.pipeline_state)
  
  
html = html.render(env.sys, rollout)
with open("output.html", "w") as f:
    f.write(html)

# train_sps = []
# def progress(_, metrics):
#   if 'training/sps' in metrics:
#     train_sps.append(metrics['training/sps'])


# _, params, metrics = ppo.train(
#       env, num_timesteps=50_000_000, num_evals=10, reward_scaling=10, 
#       episode_length=1000, normalize_observations=True, action_repeat=1, 
#       unroll_length=5, num_minibatches=32, num_updates_per_batch=4, 
#       discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=4096, 
#       batch_size=2048, seed=1, progress_fn = progress)




# print(f'train steps/sec: {np.mean(train_sps[1:])}')
# print(metrics)