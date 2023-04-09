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
import matplotlib.pyplot as plt
import wandb
import random
import pickle
import numpy as np

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

run = wandb.init(project="1v1 new",
    name="visual",)


env = Soccer_field()

with open('1v1 data/init',mode='rb') as file:
    params = file.read()
decoded_params = pickle.loads(params)

env.opp_params = decoded_params[:2]

with open('1v1 data/8',mode='rb') as file:
    params = file.read()
decoded_params = pickle.loads(params)


normalize_fn = lambda x, y: x
normalize_observations = True
if normalize_observations:
      normalize_fn = running_statistics.normalize
ppo_network = ppo_networks.make_ppo_networks(env.observation_size,
                                                 env.action_size(), normalize_fn)
inference = ppo_networks.make_inference_fn(ppo_network)

rollout = []
jit_env_reset = jax.jit(env.reset)
state = jit_env_reset(rng=jax.random.PRNGKey(seed=0))
qp = state.qp
rollout.append(state.qp)

# act_x = []
# act_y = []
# raw_action = []

jit_env_step = jax.jit(env.step)
i = 0
score = 0
while i < 10:
  action, metrics = inference(decoded_params[:2])(state.obs, jax.random.PRNGKey(i))
  state = jit_env_step(state, action)
  rollout.append(state.qp)
  # print(state.metrics['score1'])
  if state.done == 1 or state.metrics['steps'] == 1000:
    i += 1
    score += state.metrics['score1']
    print(score)
    state = jit_env_reset(rng=jax.random.PRNGKey(seed=i))
  

html = html.render(env.sys, rollout)
with open("output.html", "w") as f:
    f.write(html)
html1 = wandb.Html(open('output.html'))
wandb.log({'output':html1})
print(1)
