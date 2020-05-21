import gym
import os
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines.bench import Monitor

from stable_baselines.common.policies import MlpPolicy

from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import DQN, ACKTR
#from stable_baselines.deepq.policies import MlpPolicy
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
    env = gym.make(env_id)
    #env = CustomEnv(3, 6, "tcp://*:5556")
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    # Create log dir
    log_dir = "Logs/Custom_env/"
    os.makedirs(log_dir, exist_ok=True)
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=500, log_dir=log_dir)

    #env = Monitor(env, log_dir)

    model = ACKTR(MlpPolicy, env, verbose=2)
    #model.load("DQN_agent")
    model.learn(total_timesteps=20000, callback=callback)
    model.save("temp_agent")

    a = input("Training completed")

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        probs = model.action_probability(obs)
        obs, rewards, dones, info = env.step(action)
        print("Observation:", obs, rewards, probs)

    results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "Lane Manager")
    plt.show()