import brax
from brax import jumpy as jp
from brax.envs import env
from brax import QP
from jax import numpy as jnp
import random




class Soccer_field(env.Env):
    def __init__(self, cutoff= 0,         #cutoff>0 nearneighbor
               **kwargs):
        config = _SYSTEM_CONFIG
        super().__init__(config=config, **kwargs)
        self.cutoff = cutoff
        self._reset_noise_scale = 0.05
        self.act_dim = int(2)

    def reset(self, rng: jp.ndarray) -> env.State:
        """Resets the environment to an initial state."""
        # rng, rng1, rng2, rng3 = jp.random_split(rng, 4)
        qp = self.sys.default_qp()
        qp.pos[14,0] = -0.5 + self._noise(rng)
        qp.pos[14,1] = self._noise(rng)
        qp.pos[13,0] = self._noise(rng)
        qp.pos[13,1] = self._noise(rng)
        # qp.pos[15,0] = 0.5
        self.sys.config.collider_cutoff = self.cutoff
        obs = self._get_obs(qp, self.sys.info(qp))
        reward, done, zero = jp.zeros(3)
        metrics = {
            'reward': zero,
        }
        return env.State(qp, obs, reward, done, metrics)
    
    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        obs = jp.concatenate([qp.pos[13,0:1],qp.pos[14,0:1],qp.vel[13,0:1],qp.vel[14,0:1]])
        return(obs)
        

    def step(self, state: env.State , action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        action = jnp.concatenate([action,jnp.zeros(1)])
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)
        score = jnp.where(qp.pos[13,0] > 0.76,1.0,0.0)
        # score = jnp.where(qp.pos[13,0] < -0.76,-1.0,0.0)
        dis_rew = 0.2 - jnp.linalg.norm(qp.pos[14,0:1]-qp.pos[13,0:1])
        goal = jnp.zeros(2).at[0:2].set([0.75,0])
        kick_rew = 0.5 - jnp.linalg.norm(qp.pos[13,0:1] - goal)
        reward = dis_rew + 3 * kick_rew + 5 * score
        done = jnp.where(qp.pos[13,0] > 0.76,1.0,0.0)

        return state.replace(qp=qp, obs=obs, reward=reward, done=done)


    
    def action_size(self) -> int:
        return self.act_dim


    def _noise(self, rng):
      low, hi = -self._reset_noise_scale, self._reset_noise_scale
      return random.uniform(low,hi)








_SYSTEM_CONFIG = """
  bodies {
    name: "Ground"
    colliders {
      plane {}
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "Wall 1"
    colliders {
      position { y: 0.65 }
      rotation {}
      box{
        halfsize: {x: 0.75 y: 0.01 z: 0.1}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "Wall 2"
    colliders {
      position { y: -0.65 }
      rotation {}
      box{
        halfsize: {x: 0.75 y: 0.01 z: 0.1}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "Wall 3"
    colliders {
      position { x: -0.85 }
      rotation { z: 90}
      box{
        halfsize: {x: 0.2 y: 0.01 z: 0.1}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "Wall 4"
    colliders {
      position { x: 0.85 }
      rotation { z: 90}
      box{
        halfsize: {x: 0.2 y: 0.01 z: 0.1}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "Wall 5"
    colliders {
      position { x: 0.75 y:0.425}
      rotation { z: 90}
      box{
        halfsize: {x: 0.225 y: 0.01 z: 0.1}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "Wall 6"
    colliders {
      position { x: 0.75 y:-0.425}
      rotation { z: 90}
      box{
        halfsize: {x: 0.225 y: 0.01 z: 0.1}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "Wall 7"
    colliders {
      position { x: -0.75 y:0.425}
      rotation { z: 90}
      box{
        halfsize: {x: 0.225 y: 0.01 z: 0.1}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "Wall 8"
    colliders {
      position { x: -0.75 y: -0.425}
      rotation { z: 90}
      box{
        halfsize: {x: 0.225 y: 0.01 z: 0.1}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "Wall 9"
    colliders {
      position { x: 0.8 y: 0.2 }
      rotation {}
      box{
        halfsize: {x: 0.05 y: 0.01 z: 0.1}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "Wall 10"
    colliders {
      position { x: -0.8 y: 0.2 }
      rotation {}
      box{
        halfsize: {x: 0.05 y: 0.01 z: 0.1}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "Wall 11"
    colliders {
      position { x: -0.8 y: -0.2 }
      rotation {}
      box{
        halfsize: {x: 0.05 y: 0.01 z: 0.1}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "Wall 12"
    colliders {
      position { x: 0.8 y: -0.2 }
      rotation {}
      box{
        halfsize: {x: 0.05 y: 0.01 z: 0.1}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "Ball"
    colliders {
      sphere {
        radius: 0.02135
      }
    }
    inertia { x: 2.935e-05 y: 2.935e-05 z: 2.935e-05 }
    mass: 0.046
  }
  bodies {
    name: "Player 0"
    colliders {
      box{
        halfsize: {x: 0.04 y: 0.04 z: 0.04}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  forces {
    name: "forceplayer0"
    body: "Player 0"
    strength: 15.0
    thruster{}
  }
  friction: 1.0
  gravity { z: -9.8 }
  angular_damping: -0.05
  dt: 0.05
  substeps: 10
  dynamics_mode: "pbd"
  """
