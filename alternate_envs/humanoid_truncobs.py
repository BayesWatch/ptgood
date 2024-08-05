import numpy as np
# from gym.envs.mujoco import mujoco_env
from alternate_envs.base import MuJocoPyEnv
from d4rl import offline_env
from gym import utils
from d4rl.utils.wrappers import NormalizedBoxEnv
from gym.spaces import Box

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidTruncatedObsEnv(MuJocoPyEnv, utils.EzPickle):
    """
        COM inertia (cinert), COM velocity (cvel), actuator forces (qfrc_actuator),
        and external forces (cfrc_ext) are removed from the observation.
        Otherwise identical to Humanoid-v2 from
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoid.py
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(self, **kwargs):
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(45,), dtype=np.float64
        )

        MuJocoPyEnv.__init__(
            self, 'humanoid.xml', 5,
            observation_space=observation_space,
            **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               # data.cinert.flat,
                               # data.cvel.flat,
                               # data.qfrc_actuator.flat,
                               # data.cfrc_ext.flat
                               ])

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data

        # Bottom reward modifcation is from openai
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.model.opt.timestep
        # lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20


class HumanoidTruncatedObsMBPORewardEnv(MuJocoPyEnv, utils.EzPickle):
    """
        COM inertia (cinert), COM velocity (cvel), actuator forces (qfrc_actuator),
        and external forces (cfrc_ext) are removed from the observation.
        Otherwise identical to Humanoid-v2 from
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoid.py
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(self, **kwargs):
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(45,), dtype=np.float64
        )

        MuJocoPyEnv.__init__(
            self, 'humanoid.xml', 5,
            observation_space=observation_space,
            **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               # data.cinert.flat,
                               # data.cvel.flat,
                               # data.qfrc_actuator.flat,
                               # data.cfrc_ext.flat
                               ])

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20


class HumanoidTruncObs(HumanoidTruncatedObsEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HumanoidTruncatedObsEnv.__init__(self)
        offline_env.OfflineEnv.__init__(self, **kwargs)


class HumanoidTruncObsMBPOReward(HumanoidTruncatedObsMBPORewardEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HumanoidTruncatedObsMBPORewardEnv.__init__(self)
        offline_env.OfflineEnv.__init__(self, **kwargs)


def get_humanoidtruncobs_env():
    return NormalizedBoxEnv(HumanoidTruncObs())


def get_humanoidtruncobsmbporeward_env():
    return NormalizedBoxEnv(HumanoidTruncObsMBPOReward())
