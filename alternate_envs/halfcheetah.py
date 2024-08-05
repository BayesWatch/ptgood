import numpy as np

from gym import utils
from alternate_envs.base import MuJocoPyEnv
# from gym.envs.mujoco import MuJocoPyEnv
from gym.spaces import Box
from d4rl import offline_env
from d4rl.utils.wrappers import NormalizedBoxEnv


class HalfCheetahEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, xml_file, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        # MuJocoPyEnv.__init__(
        #     self, "half_cheetah.xml", 5, observation_space=observation_space, **kwargs
        # )
        MuJocoPyEnv.__init__(
            self, xml_file, 5, observation_space=observation_space, **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        terminated = False

        if self.render_mode == "human":
            self.render()
        return (
            ob,
            reward,
            terminated,
            # False,
            dict(reward_run=reward_run, reward_ctrl=reward_ctrl),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class HeavyHalfCheetahEnv(HalfCheetahEnv, offline_env.OfflineEnv):
    def __init__(self, xml_file, **kwargs):
        HalfCheetahEnv.__init__(self, xml_file)
        offline_env.OfflineEnv.__init__(self, **kwargs)


def get_cheetah_env(xml_file):
    if 'heavy' in xml_file:
        return NormalizedBoxEnv(HeavyHalfCheetahEnv(xml_file))
