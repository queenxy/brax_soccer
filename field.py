import brax
from brax import jumpy as jp
from brax.envs import env
from brax import QP
from jax import numpy as jnp
import random
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import pickle
import jax
from brax.training.acme import specs

N_Robots = 1
init_pos = [-0.5,0,0.5,0]

# training agents : initial position_x < 0, target goal 0.75

def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)

class Soccer_field(env.Env):
    def __init__(self, cutoff= 0,       #cutoff>0 nearneighbor
               **kwargs):
        config = _SYSTEM_CONFIG
        super().__init__(config=config, **kwargs)
        self.cutoff = cutoff
        self._reset_noise_scale = 0.1
        self.kp = 10
        self.act_dim = 2 * N_Robots
        self.obs_dim = 4 * (2*N_Robots+1)
        self.episode_length = 1000
        normalize_fn = lambda x, y: x
        normalize_observations = True
        if normalize_observations:
          normalize_fn = running_statistics.normalize
        ppo_network = ppo_networks.make_ppo_networks(self.obs_dim,
                                                 self.act_dim, normalize_fn)
        self.opp_inference = ppo_networks.make_inference_fn(ppo_network)
        key_policy = jax.random.PRNGKey(seed=0)
        normalizer_params=running_statistics.init_state(
          specs.Array((self.obs_dim,), jnp.float32))
        policy_params =  ppo_network.policy_network.init(key_policy)
        self.opp_params = (normalizer_params,policy_params)
        self.opp_params = jax.device_put_replicated(self.opp_params, jax.local_devices()[:jax.local_device_count()])
        self.opp_params = _unpmap(self.opp_params)
  
    def reset(self, rng: jp.ndarray) -> env.State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        qp = self.sys.default_qp()
        pos = jax.numpy.array(qp.pos)
        vel = jax.numpy.array(qp.vel)

        # initial the state
        pos = pos.at[0,0].set(self._noise(rng))
        pos = pos.at[0,1].set(self._noise(rng1))
        pos = pos.at[0,2].set(0.02135)
        vel = vel.at[0].set([0,0,0])
        for i in range(2 * N_Robots):
          pos = pos.at[i+1,0].set(init_pos[2*i] + self._noise(rng2))
          pos = pos.at[i+1,1].set(init_pos[2*i+1] + self._noise(rng3))
          pos = pos.at[i+1,2].set(0.04)
          vel = vel.at[i+1].set([0,0,0])
        qp = qp.replace(pos = pos,vel = vel)
       
        self.sys.config.collider_cutoff = self.cutoff
        obs = self._get_obs(qp, self.sys.info(qp))
        reward, done, zero = jp.zeros(3)
        goal = jnp.zeros(2).at[0:2].set([0.75,0])
        dis = qp.pos[1,0:2]-qp.pos[0,0:2]
        kick = qp.pos[0,0:2] - goal
        cos = jnp.dot(dis,kick)/jnp.linalg.norm(dis)/jnp.linalg.norm(kick)
        metrics = {
            'reward': zero,
            'score1': zero,
            'score2': zero,
            'pre_dis': jnp.linalg.norm(dis),
            'pre_kick': jnp.linalg.norm(kick),
            'pre_cos': cos,
            'sum_er': zero,
            'vx': zero,
            'vy': zero,
            'steps': zero,
        }
        return env.State(qp, obs, reward, done, metrics)
    
    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        obs = jp.concatenate([qp.pos[0,0:2],qp.vel[0,0:2]])
        for i in range(2 * N_Robots):
          obs = jp.concatenate([obs,qp.pos[i+1,0:2],qp.vel[i+1,0:2]])
        return(obs)

    def _get_opp_obs(self, qp: brax.QP) -> jp.ndarray:
        obs = jp.concatenate([-qp.pos[0,0:2],-qp.vel[0,0:2]])
        obs = jp.concatenate([obs,-qp.pos[2,0:2],-qp.vel[2,0:2]])
        obs = jp.concatenate([obs,-qp.pos[1,0:2],-qp.vel[1,0:2]])
        return(obs)
        
        

    def step(self, state: env.State , action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        
        qp = state.qp
        opp_obs = self._get_opp_obs(qp)
        # action = 2 * (action - 0.5 * jnp.ones(action.shape))
        qp = qp.replace(vel=qp.vel.at[1,0].set(action[0]))
        qp = qp.replace(vel=qp.vel.at[1,1].set(action[1]))

        act, _ = self.opp_inference(self.opp_params)(opp_obs, jax.random.PRNGKey(0))
        qp = qp.replace(vel=qp.vel.at[2,0].set(-act[0]))
        qp = qp.replace(vel=qp.vel.at[2,1].set(-act[1]))

        action = jnp.zeros(6)
        # action = self.pid(action,state.qp)
        qp, info = self.sys.step(qp, action)

        obs = self._get_obs(qp, info)
        score1 = jnp.where(qp.pos[0,0] > 0.76,1.0,0.0)
        score2 = jnp.where(qp.pos[0,0] < -0.76,1.0,0.0)
        flag = self.sys_bug(qp)
        metrics = state.metrics
        metrics['score1'] += score1
        metrics['score2'] += score2
        pre_dis = metrics['pre_dis']
        pre_kick = metrics['pre_kick']
        pre_cos = metrics['pre_cos']
        steps = metrics['steps']
        goal = jnp.zeros(2).at[0:2].set([0.75,0])
        dis = qp.pos[1,0:2]-qp.pos[0,0:2]
        kick = qp.pos[0,0:2] - goal
        dis_rew = 5 * (pre_dis - jnp.linalg.norm(dis))
        kick_rew = 5 * (pre_kick - jnp.linalg.norm(kick))
        ang_rew = jnp.dot(dis,kick)/jnp.linalg.norm(dis)/jnp.linalg.norm(kick) - pre_cos
        vel_rew =  jnp.where(jnp.linalg.norm(qp.vel[1,0:2]) < 0.01,1.0,0.0)
        reward = dis_rew + 3 * kick_rew + 5 * score1 + 10 * ang_rew - 5 * score2
        # reward = score1 * (1 + (self.episode_length - steps)/self.episode_length)

        metrics['pre_dis'] = jnp.linalg.norm(dis)
        metrics['pre_kick'] = jnp.linalg.norm(kick)
        metrics['pre_cos'] = jnp.dot(dis,kick)/jnp.linalg.norm(dis)/jnp.linalg.norm(kick)
        metrics['reward'] = reward
        metrics['steps'] += 1
        
        done = jnp.where(score1 + score2 + flag > 0,1.0,0.0)
        
        return state.replace(qp=qp, obs=obs, reward=reward, done=done, metrics = metrics)


    
    def action_size(self) -> int:
        return self.act_dim


    def _noise(self, rng):
      low, hi = -self._reset_noise_scale, self._reset_noise_scale
      r = jax.random.uniform(key=rng,shape=(1,1),minval=low,maxval=hi)
      return(r[0][0])

    def pid(self, vel: jp.ndarray, qp: brax.QP) -> jp.ndarray:           #transform vel to force
        vel = 2. * (vel - 0.5 * jnp.ones(vel.shape))
        pre_vel = qp.vel[14,0:1]
        er = vel - pre_vel
        action = self.kp * er
        action = jnp.clip(action,jnp.zeros(2).at[0:2].set([-5,-5]),jnp.zeros(2).at[0:2].set([5,5]))
        return(action)


    def sys_bug(self, qp: brax.QP):
      flag = jnp.zeros(8 * N_Robots)
      for i in range(2*N_Robots):
        flag = flag.at[4*i].set(jnp.where(qp.pos[i+1,0] > 0.85,1.0,0.0))
        flag = flag.at[4*i+1].set(jnp.where(qp.pos[i+1,0] < -0.85,1.0,0.0))
        flag = flag.at[4*i+2].set(jnp.where(qp.pos[i+1,1] > 0.65,1.0,0.0))
        flag = flag.at[4*i+3].set(jnp.where(qp.pos[i+1,1] < -0.65,1.0,0.0))
      done = jnp.where(flag.sum() > 0,1.0,0.0)
      # flag = jnp.zeros(4)
      # flag = flag.at[0].set(jnp.where(qp.pos[1,0] > 0.85,1.0,0.0))
      # flag = flag.at[1].set(jnp.where(qp.pos[1,0] < -0.85,1.0,0.0))
      # flag = flag.at[2].set(jnp.where(qp.pos[1,1] > 0.65,1.0,0.0))
      # flag = flag.at[3].set(jnp.where(qp.pos[1,1] < -0.65,1.0,0.0))
      # done = jnp.where(flag.sum() > 0,1.0,0.0)
      return(done)






_SYSTEM_CONFIG = """
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
  bodies {
    name: "Player 1"
    colliders {
      box{
        halfsize: {x: 0.04 y: 0.04 z: 0.04}
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
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
  forces {
    name: "forceplayer0"
    body: "Player 0"
    strength: 1.0
    thruster{}
  }
  forces {
    name: "forceplayer1"
    body: "Player 1"
    strength: 1.0
    thruster{}
  }
  friction: 1.0
  gravity { z: -9.8 }
  angular_damping: -0.05
  dt: 0.05
  substeps: 10
  dynamics_mode: "pbd"
  """
