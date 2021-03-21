import gym
import numpy as np
from gym import spaces
from External_Interface.zeromq_client import ZeroMqClient
from Env.Road_Network.RoadNetwork import RoadNetwork
from Dependency_Graph.DependencyGraph import DependencyGraph

import zmq
import json
import random
import threading


class SMARTS_env():
    """Custom Environment that follows gym interface"""

    def __init__(self):
        # Zero Mq Client
        self.sim_client = ZeroMqClient()

        # Handle Threads
        self.thread_count_step = []
        self.thread_count_get_states = []
        self.g_step = threading.RLock()
        self.g_get_states = threading.RLock()
        self.step_complete = False
        self.get_state_complete = True
        self.data = []

        # Statistics
        self.iter = 0
        self.num_agents = 0

        #Graphs
        self.road_network = RoadNetwork()
        self.dependency_graph = DependencyGraph()
        self.agent_actions = []

        #settings
        self.is_CLLA = False
        self.secPerStep = 60


    def init(self):
        print("Calling Init SMARTS...")

        road_graph = self.sim_client.send_message({'Init':'start'})

        self.get_settings(road_graph)
        self.road_network.buildGraphFromDict(road_graph["trafficData"])
        self.road_network.osmGraph.build_edge_map()

        self.num_agents = self.road_network.osmGraph.controllable_roads()

        self.thread_count_step = [False for i in range(self.num_agents)]
        self.thread_count_get_states = [False for i in range(self.num_agents)]
        self.agent_actions = [1 for i in range(self.num_agents)]

    def get_settings(self, road_graph):
        if "settings" in road_graph:
           settings = road_graph["settings"]
           if settings["externalListener"] == "CLLA":
               self.is_CLLA = True
           else:
               self.is_CLLA = False


           self.secPerStep = settings["extListenerUpdateInterval"] / settings["numStepsPerSecond"]
           self.dependency_graph.vehicle_value = settings["extListenerUpdateInterval"]/settings["mvgVehicleCount"]

    def step(self, action):
        #print("Step : Thread: ", threading.current_thread().getName(), self.iter)

        self.g_step.acquire()

        while not self.get_state_complete:
            pass

        self.thread_count_step[int(threading.current_thread().getName())] = True
        self.agent_actions[int(threading.current_thread().getName())] = action
        #for index, i in enumerate(self.thread_count_step):
        #    if i == False:
        #        print(index,end =" ")
        #print("step done")

        all_steps = all(i == True for i in self.thread_count_step)

        if all_steps:
            self.thread_count_step = [False for i in range(self.num_agents)]
            self.simulate_one_step()
            self.step_complete = True
            self.iter += 1
            print("########## ", self.iter, "################")

        self.g_step.release()
        return [1, 2, 3], 1, False, {}

    def simulate_one_step(self):
        roads, commonIndexes, actions = self.road_network.osmGraph.edge_index_from_nodes(self.agent_actions)
        if self.is_CLLA:
            roads = self.dependency_graph.generateCoordinateActions(self.road_network, commonIndexes, actions)
        #print("Controls sent." ,roads)

        if self.iter <= 1:  # Do not execute action with-in first few minutes, in order to traffic propagate
            roads = []

        # build message
        message_road = []
        for road in roads:
            message_road.append({"index": road, "laneChange": True, "speed": -1})
        message = {'edges':message_road}



        dictionary =  self.sim_client.send_message(message)

        self.road_network.updateTrafficData(dictionary["trafficData"])
        self.dependency_graph.createVariableDAG(self.road_network.osmGraph.nxGraph, dictionary["paths"])

    def get_road_details(self, id):
        obs = self.road_network.osmGraph.get_road_data(id)
        return obs, 1, False, {}


    def get_states(self):

        self.g_get_states.acquire()
        partition = int(threading.current_thread().getName())
        self.thread_count_get_states[partition] = True

        while not self.step_complete:
            pass
        self.get_state_complete = False

        state, reward, done, info = self.get_road_details(partition)

        all_steps = all(i == True for i in self.thread_count_get_states)

        if all_steps:
            self.thread_count_get_states = [False for i in range(self.num_agents)]
            self.step_complete = False
            self.get_state_complete = True


        self.g_get_states.release()
        return state, reward, done, info

    def reset(self):
        # self.sim_client.send_message({"Reset": []})
        return [0, 0, 3]  # reward, done, info can't be included

    def render(self, mode='human'):
        # Simulation runs separately
        pass

    def close(self):
        pass

    def get_env_partitions(self):
        return self.agent_actions

    def decode_message(self, action):

        lanes = 3

        if self.iter % 1 == 0:
            self.up = random.randint(0, 25)
            self.down = random.randint(0, 25)

        downstream = self.down
        upstream = self.up

        up = self.pre_obs[0]
        down = self.pre_obs[1]

        if action == 0:
            l = max(1, lanes - 1)
        elif action == 2:
            l = min(self.num_lanes - 1, lanes + 1)
        else:
            l = lanes
        print("data: ", upstream, downstream)
        reward = -abs(up / l - down / (self.num_lanes - l)) / max(1, (up + down) * 2 / self.num_lanes)

        observation = [upstream // 25, downstream // 25, lanes]
        self.pre_obs = observation
        done = False
        info = {}  # TODO: implement in a future version

        return observation, reward, done, info
