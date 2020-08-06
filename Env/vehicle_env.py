import gym
import numpy as np
from gym import spaces
from External_Interface.zeromq_client import ZeroMqClient
from Env.Vehicle_obj import  Vehicle
import zmq
import json
import random
from numpy import random

class Vehicle_env(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    # num_actions : 3 accelerate, no change, de-acceleration
    # max_speed: max speed of vehicles in ms-1
    # time_to_reach: Time remaining to reach destination
    # distance: Distance to destination

    def __init__(self, id, num_actions, max_speed=10, time_to_reach=30, distance=100):
        super(Vehicle_env, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(num_actions)
        # Example for using image as input:
        self.iter = 0
        self.sim_client = ZeroMqClient()
        self.observation_space = spaces.Box(low=np.array([0,0,0]), high=np.array([max_speed, time_to_reach, distance]), dtype=np.uint8)

        self.is_episodic = True
        self.is_simulator_used = False

        self.time_to_reach = time_to_reach
        self.step_size = 0.2

        self.id = 1
        self.episode_num = 0

        self.correctly_ended = []

        # if simulator not used
        self.vehicle = Vehicle()

    def set_id(self, id):
        self.id = id


    def step(self, action):
        self.iter += 1

        if action == 0:
            paddleCommand = -1
        elif action == 1:
            paddleCommand = 0
        else:
            paddleCommand = 1

        #print("Action", paddleCommand)
        message_send = {'edges': [], 'vehicles':[{"index":self.id, "paddleCommand": paddleCommand}]}

        if self.iter % 500 == 0:
            print("Step ", self.iter)

        if self.is_simulator_used:
            message = self.sim_client.send_message(message_send)
        else:
            message = 0

        observation, reward, done, info = self.decode_message(message, paddleCommand)

        return observation, reward, done, info

    def reset(self):
        #self.sim_client.send_message({"Reset": []})
        if self.is_simulator_used:
            self.time_to_reach = np.random.randint(8,16)
            return [0, 50, self.time_to_reach]  # reward, done, info can't be included
        else:
            return self.vehicle.reset()

    def render(self, mode='human'):
      # Simulation runs separately
      pass

    def close (self):
      pass

    def decode_message(self, message, action):

        speed = 0
        time = 0
        distance = 0
        obs = [speed,time,distance]
        done = False
        reward = 0
        info = {'is_success':False}

        if self.is_simulator_used:
            print(message["vehicles"])
            for vehicle in message["vehicles"]:
                if vehicle["vid"] == self.id:
                    speed   = int(round(vehicle["speed"]))
                    time    = vehicle["timeRemain"]
                    distance = int(round(vehicle["headPositionFromEnd"]))//50
                    done = vehicle["done"]

                    obs = [speed, time, distance]

                    if done:
                        self.episode_num += 1
                        if vehicle["headPositionFromEnd"] < 5 and time < 1:
                            reward = 500
                            self.correctly_ended.append(self.episode_num)
                        else:
                            reward = -500
                    else:
                        reward = -distance

                    print("Reward  given", reward)

        else:
            obs, reward, done, info = self.vehicle.step(action)
            #print("Obs: ", obs, "reward: ", reward, "done: ", done)

            if done:
                self.episode_num += 1
                if info['is_success']:
                    self.correctly_ended.append(self.episode_num)



        return obs, reward, done, info
