import gym
import numpy as np
from gym import spaces

from Env.SMARTS_env import SMARTS_env

class Partitioned_env(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, num_actions, num_lanes,  env: SMARTS_env, id, verbose=1):
        super(Partitioned_env, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.observation_space = spaces.Box(low=np.array([0,0,1]), high=np.array([10, 10, num_lanes-1]), dtype=np.uint8)

        # Example when using discrete actions:
        self.action_space = spaces.Discrete(num_actions)
        self.num_lanes  = num_lanes
        # Example for using image as input:
        self.iter = 0
        self.id = id

        # Main Environment
        self.env = env

        # helper variables
        self.pre_reward = 0
        self.pre_obs = [0,0,num_lanes//2]
        self.up = 0
        self.down = 0
        self.action = 0

        #debug
        self.verbose = verbose

    def step(self, action):
        self.iter += 1
        self.action = action
        self.env.step(action)
        return [1,2,3],1,False,{}

    def get_states(self):
        obs, reward, done, info = self.env.get_states()
        obs, reward, done, info = self.decode_message(obs)

        if self.verbose > 1:
            print("Env partition id: ", self.id, " at ", self.iter)
            print("Action: ", self.action)
            print("States: ", obs)
            print("Reward: ", reward)

        return obs, reward, done, info

    def reset(self):
        #self.sim_client.send_message({"Reset": []})
        return [0,0,3]  # reward, done, info can't be included

    def render(self, mode='human'):
        # Simulation runs separately
        pass

    def close (self):
        pass


    def decode_message(self, obs):

      upstream = obs[0]
      downstream = obs[1]
      lanes = obs[2]


      # reward calcilation using previos time-step data
      up = self.pre_obs[0]
      down = self.pre_obs[1]
      l = self.pre_obs[2]

      if self.verbose > 1:
          print("data: ", upstream, downstream, l)

      if self.action == 0:
            l = max(1,l-1)
      elif self.action == 2:
            l = min(self.num_lanes-1,l+1)
      reward = -abs(up/l - down/(self.num_lanes-l))/max(1,(up+down)*2/self.num_lanes)

      if self.action != 1:
              reward -= 0.05

      observation = [min(10, upstream // 5), min(10, downstream // 5), lanes]
      self.pre_obs = observation
      done = False
      info = {} #TODO: implement in a future version


      return observation, reward, done, info
