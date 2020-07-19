import gym
import numpy as np
from gym import spaces
from External_Interface.zeromq_client import ZeroMqClient
import zmq
import json
import random
from numpy import random

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, num_actions, num_lanes, port):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(num_actions)
    # Example for using image as input:
    self.iter = 0
    self.sim_client = ZeroMqClient()
    self.observation_space = spaces.Box(low=np.array([0,0,1]), high=np.array([10, 10, num_lanes-1]), dtype=np.uint8)
    self.num_lanes  = num_lanes
    self.pre_reward = 0
    self.pre_obs = [0,0,num_lanes//2]
    self.up = 0
    self.down = 0
    self.is_episodic = False
    self.is_simulator_used = False

    self.randForUp = 0
    self.randForDown = 0

  def step(self, action):
    self.iter += 1

    if action == 1:
        road = {'edges': [{"index": 4, "laneChange": True, "speed": 0}]}
    elif action == 2:
        road = {'edges': [{"index": 11, "laneChange": True, "speed": 0}]}
    else:
        road = {'edges': []}

    print(self.iter)
    print("Action: ", action)
    if self.is_simulator_used:
        message = self.sim_client.send_message(road)
    else:
        message = 0
    observation, reward, done, info = self.decode_message(message, action)
    print("States: ", observation)
    print("Reward: ", reward)
    return observation, reward, done, info

  def reset(self):
      #self.sim_client.send_message({"Reset": []})
      return [0,0,3]  # reward, done, info can't be included

  def render(self, mode='human'):
    # Simulation runs separately
    pass

  def close (self):
    pass

  def decode_message(self, message, action):

      if True:
          reward = self.pre_reward
          upstream = 0
          downstream = 0
          lanes  = 3

          if self.is_simulator_used:
            for edge in message["trafficData"]:
                if edge["index"] == 11:
                    upstream = int(round(edge["numVehiclesMvg"]))
                    lanes = edge["numLanes"]

                if edge["index"] == 4:
                    downstream = int(round(edge["numVehiclesMvg"]))
          else:
            if self.iter % 1 == 0:
                 self.randForUp = random.randint(0,250)
                 self.randForDown = random.randint(0, 250)

            if self.iter%1 == 0:
                  alpha = 0
                  self.up = alpha*self.up + (1-alpha) * random.poisson(self.randForUp, 1)[0]
                  self.down = alpha*self.down + (1-alpha) * random.poisson(self.randForDown, 1)[0]

            downstream = int(self.down)
            upstream = int(self.up)
            lanes = self.pre_obs[2]

          up = self.pre_obs[0]+1
          down = self.pre_obs[1]+1
          #lanes = self.pre_obs[2]
          #
          #print(upstream, downstream)
          if action == 0:
                l = max(2,lanes-1)
          elif action == 2:
                l = min(self.num_lanes-2,lanes+1)
          else:
                l = lanes
          print("data: ", upstream, downstream)
          reward = -abs(up/l - down/(self.num_lanes-l))/max(1,(up+down)*2/self.num_lanes)

          if action != 1:
              reward -= 0.05

          if not self.is_simulator_used:
            lanes = l # change in case of Simulator

          observation = [upstream//25, downstream//25, lanes]
          self.pre_obs = observation
          done = False
          info = {} #TODO: implement in a future version
      else:
          observation = []
          reward = 0
          done  = False
          info = {}

      return observation, reward, done, info
