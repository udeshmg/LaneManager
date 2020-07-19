from Env.partitioned_env import Partitioned_env
from Env.SMARTS_env import SMARTS_env

class Wrapper():

    def __init__(self, env : SMARTS_env):
        self.env = env
        self.partitions = []

    def partition_env(self):
        self.env.init()
        self.partitions = []
        for index, partition in enumerate(self.env.get_env_partitions()):
            debug = [] #[self.env.road_network.osmGraph.nxGraph[37][44]['index']]
            if index in debug:
                verbose = 2
            else:
                verbose = 0
            self.partitions.append(Partitioned_env(3, 6,  self.env, index, verbose))

    def get_partitions(self):
        return self.partitions
