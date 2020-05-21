import gym
import os
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines.bench import Monitor

from stable_baselines.common.policies import MlpPolicy

from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import ACKTR
from Env.custom_env import CustomEnv
from Monitor.callback import SaveOnBestTrainingRewardCallback
from stable_baselines import results_plotter

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

if __name__ == '__main__':
    env_id = "CartPole-v1"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    #env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    #env = gym.make(env_id)
    env = CustomEnv(3, 6, "tcp://*:5556")
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    # Create log dir
    log_dir = "Logs/env_id/"
    os.makedirs(log_dir, exist_ok=True)
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

   # env = Monitor(env, log_dir)

    model = ACKTR(MlpPolicy, env, verbose=2)
    model.load("RL_agent")

    while True:
        user_in = input("Enter States: ").split(',')
        obs = [int(i) for i in user_in]
        print(model.action_probability(obs))
        action = model.predict(obs, deterministic = True)
        print(action)