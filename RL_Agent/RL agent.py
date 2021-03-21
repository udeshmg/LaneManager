import gym
import numpy as np
import os
import  seaborn as sns
import  time

from Monitor.callback import SaveOnBestTrainingRewardCallback
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import ACKTR, DQN
from Env.custom_env import CustomEnv
from Env.VSL_env import VSLEnv
from stable_baselines import results_plotter
import matplotlib.pyplot as plt
from stable_baselines.bench import Monitor
#from Plot.plotter import plot_results

import threading

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[12, 12],
                                           layer_norm=False,
                                           feature_extraction="mlp")

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
    env_id = "Lane"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    #env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    log_dir = "Logs/"+env_id
    os.makedirs(log_dir, exist_ok=True)

    num_lanes = 4
    action_space = 3
    time_steps = 200000


    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)

    print("Save to :", "RL_agent_vsl_"+str(num_lanes)+"iter"+str(time_steps))
    #env = gym.make(env_id)
    env = CustomEnv(action_space, num_lanes, "tcp://*:5556")
    env.is_simulator_used = False
    env = Monitor(env, log_dir)

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    model = DQN(CustomDQNPolicy, env, gamma=0, exploration_fraction=0.6, exploration_final_eps=0, learning_rate=5e-4)
    #model = DQN.load("VSL_iter9600ver2.zip", env=env)
    start = time.time()
    model.learn(total_timesteps=time_steps, callback=callback)
    end  = time.time()
    model.save(env_id+"_iter"+str(time_steps)+"_lane"+str(num_lanes))
    print("Training time: ", end-start)

    #Results Plot
    #results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "Speed Limit Manager")
    #plt.show()

    #Additional Logs
    for k in range(3):
        array = np.zeros(shape=(10, 10))
        for i in range(10):
            for j in range(10):
                obs = [i, j, k+1]
                array[i][j] = model.predict(obs, deterministic=True)[0]
        ax = sns.heatmap(array, linewidth=0.5)
        plt.show()

    #Run Simulation after training
    #obs = env.reset()
    #for _ in range(1000):
    #    action, _states = model.predict(obs)
    #    obs, rewards, dones, info = env.step(action)
    #    env.render()