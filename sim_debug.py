from brax import envs
from brax import jumpy as jp
from brax.io import html
from brax.io import image
import jax
from jax import numpy as jnp

from field import Soccer_field


env = Soccer_field()


rollout = []
jit_env_reset = jax.jit(env.reset)
state = jit_env_reset(rng=jp.random_prngkey(seed=0))
qp = state.qp
rollout.append(state.qp)

jit_env_step = jax.jit(env.step)
for i in range(100):
  act = jnp.ones(2)
  state = jit_env_step(state, act)
  if state.done == 1:
    state = jit_env_reset(rng=jp.random_prngkey(seed=0))
  print(state.metrics['reward'])
  rollout.append(state.qp)


html = html.render(env.sys, rollout)
with open("output.html", "w") as f:
    f.write(html)