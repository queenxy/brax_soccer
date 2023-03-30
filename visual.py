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

run = wandb.init(project="shooting_test",
    name="3",)


env = Soccer_field()

before_artifact = run.use_artifact('shooting_test:v1415')
before_dataset = before_artifact.download()
with open(before_dataset + '/98',mode='rb') as file:
    params = file.read()
decoded_params = pickle.loads(params)
# print(decoded_params)

normalize_fn = lambda x, y: x
normalize_observations = True
if normalize_observations:
      normalize_fn = running_statistics.normalize
ppo_network = ppo_networks.make_ppo_networks(env.observation_size,
                                                 env.action_size(), normalize_fn)
inference = ppo_networks.make_inference_fn(ppo_network)

rollout = []
jit_env_reset = jax.jit(env.reset)
state = jit_env_reset(rng=jp.random_prngkey(seed=0))
qp = state.qp
rollout.append(state.qp)

# act_x = []
# act_y = []
# raw_action = []

jit_env_step = jax.jit(env.step)
for i in range(1000):
  action, metrics = inference(decoded_params)(state.obs, jax.random.PRNGKey(0))
  # act_x.append(action[0])
  # act_y.append(action[1])
  
#   raw_action.append(metrics['logits'])
  # act = jp.concatenate([act,jnp.zeros(6)])
  state = jit_env_step(state, action)
  if state.done == 1:
    state = jit_env_reset(rng=jp.random_prngkey(seed=0))
  print(state.metrics['reward'])
  rollout.append(state.qp)

# i = range(1000)
# plt.plot(i,act_x)
# plt.show()
# plt.plot(i,act_y)
# plt.show()



html = html.render(env.sys, rollout)
with open("output.html", "w") as f:
    f.write(html)
