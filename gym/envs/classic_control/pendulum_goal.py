import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class PendulumGoalEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self.goal = [np.cos(0.0), np.sin(0.0), 0.0]
        self.n = 0

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state # th := theta

        self.g = 10.
        self.m = 1.
        self.l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering

        newthdot = thdot + (-3*self.g/(2*self.l) * np.sin(th + np.pi) + 3./(self.m*self.l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        self.n += 1

        e0 = self._calc_costs(th, thdot)
        e1 = self._calc_costs(newth, newthdot)
        costs = abs(e1 - e0)

        target_achieved = abs(self.state[0] - self.goal[0]) < 0.02 and abs(self.state[1] - self.goal[1]) < 0.05
        iteration_too_much = self.n > 5000
        if target_achieved:
            costs = -self.min_costs
        elif iteration_too_much:
            costs = 10 * self.min_costs

        done = target_achieved or iteration_too_much

        return self._get_obs(), costs, done, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def setup(self, settings):
        # settings should contain two lists: start state and goal state
        self.state = settings[0] # th, thdot
        self.goal = settings[1]

        e0_start = self._calc_costs(self.state[0], self.state[1])
        e1_start = self._calc_costs(self.goal[0], self.goal[1])
        self.min_costs = abs(e1_start - e0_start)

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def _calc_costs(self, th, thdot):
        return 1 / 3.0 * self.m * self.l ** 2 * thdot ** 2 + self.m * self.g * self.l * (1 + np.cos(th))

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)