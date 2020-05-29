import gym
import numpy as np
from gym import spaces
from External_Interface.zeromq_client import ZeroMqClient
import zmq
import json
import random

class VSLEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, num_actions, num_lanes, port):
    super(VSLEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(num_actions)
    # Example for using image as input:
    self.iter = 0
    self.sim_client = ZeroMqClient(port="tcp://localhost:5556")
    self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([10, 10, 30, 30]), dtype=np.uint8)
    self.num_lanes  = num_lanes
    self.pre_reward = 0
    self.pre_obs = [0, 0]
    self.up = 0
    self.down = 0
    self.is_episodic = True
    self.time_step_size = 10

  def step(self, action):
    self.iter += 1

    road = {"edges":[{"index": 2, "laneChange": False, "speed": int(action)}]}


    message = self.sim_client.send_message(road)
    observation, reward, done, info = self.decode_message(message, action)
    print(self.iter)
    print("Action: ", action)
    print("States: ", observation)
    print("Reward: ", reward)
    return observation, reward, done, info

  def reset(self):
      #self.sim_client.send_message({"Reset": []})
      return np.zeros( shape=self.observation_space.shape ) # reward, done, info can't be included

  def render(self, mode='human'):
        # Simulation runs separately
        pass

  def close (self):
        pass

  def decode_message(self, message, action):

      if True:

          for edge in message["trafficData"]:
              if edge["index"] == 2:
                  onMove =  int(round(edge["numVehiclesOnMove"]))
                  stopped = int(round(edge["numVehiclesStopped"]))
                  color_g = min(30, int(round(edge["timeToGreen"])))
                  color_r = min(30, int(round(edge["timeToRed"])))
                  exit_Flow = edge["exitFlow"]

          if "settings" in  message:
            self.time_step_size = message["settings"]["extListenerUpdateInterval"] / message["settings"]["numStepsPerSecond"]
            print("Time step size: ", self.time_step_size, "(s)")

          #if self.iter%1 == 0:
          #      self.up = random.randint(0,250)
          #      self.down = random.randint(0,250)
          #
          #downstream = self.down
          #upstream = self.up

          observation = [onMove//5, stopped//5, color_g, color_r]

          if onMove + stopped == 0:
              reward = 0
          else:
              reward = -stopped

          if self.iter % int((1200//self.time_step_size)) == 0:
              print("############ End episode #############")
              if self.is_episodic:
                  done = True
              else:
                  done = False
          else:
              done = False

          info = {} #TODO: implement in a future version
      else: #TODO: Currently not used
          observation = []
          reward = 0
          done  = False
          info = {}

      return observation, reward, done, info
