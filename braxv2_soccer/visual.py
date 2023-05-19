from brax import envs
from brax.io import html
import jax
from jax import numpy as jp
from fieldv2 import Soccer_field
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import numpy as np
import wandb
import pickle



import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

run = wandb.init(project="braxv2",
    name="visual",)

# env_name = 'humanoid'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
# backend = 'positional'  # @param ['generalized', 'positional', 'spring']

# env = envs.get_environment(env_name=env_name,
#                            backend=backend)


env = Soccer_field()

with open('/home/qxy/braxv2_soccer/w&b/8-7',mode='rb') as file:
    params = file.read()
decoded_params = pickle.loads(params)

env.opp_params = decoded_params[:2]

with open('/home/qxy/braxv2_soccer/w&b/8-8',mode='rb') as file:
    params = file.read()
decoded_params = pickle.loads(params)


normalize_fn = lambda x, y: x
normalize_observations = True
if normalize_observations:
      normalize_fn = running_statistics.normalize
ppo_network = ppo_networks.make_ppo_networks(env.observation_size,
                                                 env.action_size(), normalize_fn)
inference = ppo_networks.make_inference_fn(ppo_network)

jit_env_reset = jax.jit(env.reset)
state = jit_env_reset(rng=jax.random.PRNGKey(seed=0))
rollout = []
rollout.append(state.pipeline_state)

# act_x = []
# act_y = []
# raw_action = []

jit_env_step = jax.jit(env.step)
i = 0
score = 0
while i < 5:
  action, metrics = inference(decoded_params[:2],True)(state.obs, jax.random.PRNGKey(i))
  state = jit_env_step(state, action)
  # print(state.metrics['reward'])
  rollout.append(state.pipeline_state)
#   print(action)
  if state.done == 1 or state.metrics['steps'] == 5000:
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
