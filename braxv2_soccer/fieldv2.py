from brax import base
from brax import math
from brax.envs import env
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import specs

N_Robots = 1

def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)

class Soccer_field(env.PipelineEnv):
    def __init__(
      self,
      backend='positional',
      **kwargs,
  ):
      path = '/home/qxy/braxv2_soccer/braxv2_soccer/soccerfield.xml'
      sys = mjcf.load(path)

      n_frames = 5

      if backend in ['spring', 'positional']:
        sys = sys.replace(dt=0.005)
        n_frames = 10

      if backend == 'positional':
        # TODO: does the same actuator strength work as in spring
        sys = sys.replace(
            actuator=sys.actuator.replace(
                gear=0.1 * jp.ones_like(sys.actuator.gear)
            )
        )

      kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
      super().__init__(sys=sys, backend=backend,**kwargs)

      
      self._reset_noise_scale = 0.1
      self.kp = 10
      self.act_dim = 2 * N_Robots
      self.obs_dim = 4 * (2*N_Robots+1)
      normalize_fn = lambda x, y: x
      normalize_observations = True
      if normalize_observations:
        normalize_fn = running_statistics.normalize
      ppo_network = ppo_networks.make_ppo_networks(self.obs_dim,
                                                self.act_dim, normalize_fn)
      self.opp_inference = ppo_networks.make_inference_fn(ppo_network)
      key_policy = jax.random.PRNGKey(seed=0)
      normalizer_params=running_statistics.init_state(
        specs.Array((self.obs_dim,), jp.float32))
      policy_params =  ppo_network.policy_network.init(key_policy)
      self.opp_params = (normalizer_params,policy_params)
      self.opp_params = jax.device_put_replicated(self.opp_params, jax.local_devices()[:jax.local_device_count()])
      self.opp_params = _unpmap(self.opp_params)
      
    def _noise(self, rng):
      low, hi = -self._reset_noise_scale, self._reset_noise_scale
      r = jax.random.uniform(key=rng,shape=(1,1),minval=low,maxval=hi)
      return(r[0][0])


    def reset(self, rng: jp.ndarray) -> env.State:
      """Resets the environment to an initial state."""
      subrng = jax.random.split(rng, 2*N_Robots+2)

      q = self.sys.init_q 
      qd = jp.zeros(self.sys.qd_size())

      pipeline_state = self.pipeline_init(q, qd)
      x = pipeline_state.x
      qpos = pipeline_state.x.pos

      # initial the state
      qpos = qpos.at[1,0].add(self._noise(subrng[0]))
      qpos = qpos.at[1,1].add(self._noise(subrng[1]))
      for i in range(2 * N_Robots):
        qpos = qpos.at[2*i+2,0].add(self._noise(subrng[2*i+2]))
        qpos = qpos.at[2*i+2,1].add(self._noise(subrng[2*i+2]))
        qpos = qpos.at[2*i+3,0].add(self._noise(subrng[2*i+3]))
        qpos = qpos.at[2*i+3,1].add(self._noise(subrng[2*i+3]))

      pipeline_state = pipeline_state.replace(x=x.replace(pos=qpos))

      obs = self._get_obs(pipeline_state)
      qpos = pipeline_state.x.pos

      goal1 = jp.array([0.75,0])
      goal0 = jp.array([-0.75,0])
      dis = qpos[2,0:2]-qpos[1,0:2]
      kick1 = qpos[1,0:2] - goal1
      kick0 = qpos[1,0:2] - goal0
      cos1 = jp.dot(dis,kick1)/jp.linalg.norm(dis + 1e-5)/jp.linalg.norm(kick1 + 1e-5)
      cos0 = jp.dot(dis,kick0)/jp.linalg.norm(dis + 1e-5)/jp.linalg.norm(kick0 + 1e-5)

      reward, done, zero = jp.zeros(3)
      metrics = {
          'reward': zero,
          'score1': zero,
          'score2': zero,
          'pre_dis': jp.linalg.norm(dis),
          'pre_kick': jp.linalg.norm(kick1),
          'pre_cos1': cos1,
          'pre_cos0': cos0,
          'steps': zero,
      }
      return env.State(pipeline_state, obs, reward, done, metrics)


    def _get_obs(self, pipeline_state: base.State) -> jp.ndarray:
      """Observe position and velocities."""
      qpos = pipeline_state.x.pos[1,0:2]
      for i in range(N_Robots):
        qpos = jp.concatenate([qpos,pipeline_state.x.pos[2*i+2,0:2]])
        qpos = jp.concatenate([qpos,pipeline_state.x.pos[2*i+2+N_Robots,0:2]])
      qvel = pipeline_state.xd.vel[1,0:2]
      for i in range(N_Robots):
        qvel = jp.concatenate([qvel,pipeline_state.xd.vel[2*i+2,0:2]])
        qvel = jp.concatenate([qvel,pipeline_state.xd.vel[2*i+2+N_Robots,0:2]])

      return jp.concatenate([qpos] + [qvel])
    
    def _get_opp_obs(self, pipeline_state: base.State) -> jp.ndarray:
      """Observe position and velocities."""
      qpos = pipeline_state.x.pos[1,0:2]
      for i in range(N_Robots):
        qpos = jp.concatenate([qpos,pipeline_state.x.pos[2*i+2+N_Robots,0:2]])
        qpos = jp.concatenate([qpos,pipeline_state.x.pos[2*i+2,0:2]])
      qvel = pipeline_state.xd.vel[1,0:2]
      for i in range(N_Robots):
        qvel = jp.concatenate([qvel,pipeline_state.xd.vel[2*i+2+N_Robots,0:2]])
        qvel = jp.concatenate([qvel,pipeline_state.xd.vel[2*i+2,0:2]])

      return jp.concatenate([-qpos] + [-qvel])


    
    def step(self, state: env.State, action: jp.ndarray) -> env.State:
      """Run one timestep of the environment's dynamics."""
      pipeline_state0 = state.pipeline_state

      opp_obs = self._get_opp_obs(pipeline_state0)
      metrics = state.metrics

      # P-control
      act, _ = self.opp_inference(self.opp_params,True)(opp_obs, jax.random.PRNGKey(0))
      vel = 0.05 * jp.concatenate((action,-act))
      pre_vel = jp.concatenate((pipeline_state0.xd.vel[2,0:2],pipeline_state0.xd.vel[3,0:2]))
      force = self.kp * (vel - pre_vel) 

      pipeline_state = self.pipeline_step(pipeline_state0, force)

      obs = self._get_obs(pipeline_state)
      qpos = pipeline_state.x.pos
      # metrics['obs']=obs

      score1 = jp.where(qpos[1,0] > 0.76,1.0,0.0)
      score2 = jp.where(qpos[1,0] < -0.76,1.0,0.0)

      metrics['score1'] = score1
      metrics['score2'] = score2
      pre_kick = metrics['pre_kick']
      pre_dis = metrics['pre_dis']
      pre_cos1 = metrics['pre_cos1']
      pre_cos0 = metrics['pre_cos0']
      
      goal1 = jp.array([0.75,0])
      goal0 = jp.array([-0.75,0])
      dis = qpos[2,0:2]-qpos[1,0:2]
      kick1 = qpos[1,0:2] - goal1
      kick0 = qpos[1,0:2] - goal0
      dis_rew = 5 * (pre_dis - jp.linalg.norm(dis))
      kick_rew = 5 * (pre_kick - jp.linalg.norm(kick1))
      ang_rew1 = jp.dot(dis,kick1)/(jp.linalg.norm(dis) + 1e-5)/(jp.linalg.norm(kick1) + 1e-5) - pre_cos1
      ang_rew0 = pre_cos0 - jp.dot(dis,kick0)/(jp.linalg.norm(dis) + 1e-5)/(jp.linalg.norm(kick0) + 1e-5)
      reward = dis_rew + 3 * kick_rew + 10 * score1 +  ang_rew1 + ang_rew0 - 10 * score2

      metrics['pre_dis'] = jp.linalg.norm(dis)
      metrics['pre_kick'] = jp.linalg.norm(kick1)
      metrics['pre_cos1'] = jp.dot(dis,kick1)/(jp.linalg.norm(dis) + 1e-5)/(jp.linalg.norm(kick1) + 1e-5)
      metrics['pre_cos0'] = jp.dot(dis,kick0)/(jp.linalg.norm(dis) + 1e-5)/(jp.linalg.norm(kick0) + 1e-5)
      metrics['reward'] += reward
      metrics['steps'] += 1
      done = jp.where(score1 + score2 > 0,1.0,0.0)
      
  
      return state.replace(
          pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
      )

    
    def action_size(self) -> int:
        return self.act_dim




