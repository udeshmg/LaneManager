
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines import ACKTR, DQN
from Env.Wrapper_env import Wrapper

import threading
import copy

class RL_Agent_Handler():

    def __init__(self, wrapper : Wrapper):
        self.env = wrapper
        self.agents = []
        self.threads = []
        self.partitions = []

    def create_agents(self, pre_trained, path="backups/RL_agent6iter200000.zip"):

        dummy_agent_4 = DQN.load("backups/Lane_iter200000_lane4.zip")
        dummy_agent_6 = DQN.load("backups/RL_agent6iter200000.zip")

        for index, partition in enumerate(self.env.get_partitions()):
            self.partitions.append(partition)

            if pre_trained:
                if partition.num_lanes == 4:
                    agent = copy.copy(dummy_agent_4)
                else:
                    agent = copy.copy(dummy_agent_6)
                agent.env = partition
                agent.exploration_initial_eps = 0
                agent.exploration_final_eps = 0
            else:
                agent = DQN(MlpPolicy, partition, verbose=2, gamma=0.9, exploration_fraction=0.6, exploration_final_eps=0)

            print("Agent added: ", index)
            self.agents.append(agent)

    def learn(self, time_steps):
        for index, agent in enumerate(self.agents):
            self.threads.append(threading.Thread(target=agent.learn, args=(time_steps,), name=str(index)))
            print("Thread added: ", index)

        for thread in self.threads:
            thread.setDaemon(True)
            thread.start()

        for index, thread in enumerate(self.threads):
            print("Joined: ", thread.getName())
            thread.join()


    def load_weights(self, id):
        print("Agent Loaded: ", id)
        self.agents[id] = DQN.load("backups/Lane_iter200000_lane4.zip", env=self.partitions[id])
        print("Agent Loaded: ", id, "Competed")

    def predict(self, time_steps):
        self.threads = []
        for index, agent in enumerate(self.agents):
            self.threads.append(threading.Thread(target=self.predict_for_num_steps,
                                                 args=(agent, self.partitions[index], time_steps),
                                                 name=str(index)))
            print("Thread added: ", index)
        for thread in self.threads:
            thread.setDaemon(True)
            thread.start()

        for index, thread in enumerate(self.threads):
            print("Joined: ", thread.getName())
            thread.join()

    def predict_for_num_steps(self, agent, env, time_steps):
        obs = [0, 0, 3]

        for i in range(time_steps):
            """num_lanes = env.num_lanes

            if agent.num_lanes != num_lanes:
                print("OBS:", obs)
                path = "backups/RL_agent"+str(num_lanes)+"iter200000.zip"
                agent = DQN.load(path)
                agent.num_lanes = env.num_lanes
                agent.env = env
                agent.exploration_initial_eps = 0
                agent.exploration_final_eps = 0
                self.agents[env.id] = agent   
             """


            action = agent.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action[0])
            obs, rewards, dones, info = env.get_states()


