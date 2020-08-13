__all__ = ['Monitor_save_step_data', 'get_monitor_files', 'load_results']

import csv
import json
import os
import time
from glob import glob
from typing import Tuple, Dict, Any, List, Optional

import gym
import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pandas as pd

class Monitor_save_step_data(gym.Wrapper):
    EXT = "monitor.csv"
    file_handler = None

    def __init__(self,
                 env: gym.Env,
                 filename: Optional[str],
                 allow_early_resets: bool = True,
                 reset_keywords=(),
                 info_keywords=()):
        """
        A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

        :param env: (gym.Env) The environment
        :param filename: (Optional[str]) the location to save a log file, can be None for no log
        :param allow_early_resets: (bool) allows the reset of the environment before it is done
        :param reset_keywords: (tuple) extra keywords for the reset call, if extra parameters are needed at reset
        :param info_keywords: (tuple) extra information to log, from the information return of environment.step
        """
        super(Monitor_save_step_data, self).__init__(env=env)
        self.t_start = time.time()
        if filename is None:
            self.file_handler = None
            self.logger = None
        else:
            if not filename.endswith(Monitor_save_step_data.EXT):
                if os.path.isdir(filename):
                    filename = os.path.join(filename, Monitor_save_step_data.EXT)
                else:
                    filename = filename + "." + Monitor_save_step_data.EXT
            self.file_handler = open(filename, "wt")
            self.file_handler.write('#%s\n' % json.dumps({"t_start": self.t_start, 'env_id': env.spec and env.spec.id}))
            self.logger = csv.DictWriter(self.file_handler,
                                         fieldnames=('r', 'l', 't') + reset_keywords + info_keywords)
            self.logger.writeheader()
            self.file_handler.flush()

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()

        self.episode_actions = []
        self.episode_speed = []
        self.episode_remain_time = []
        self.episode_distance = []
        self.episode_success = []
        self.mask = []

        self.actions = []
        self.speeds = []
        self.times = []
        self.distance = []

        self.iter = 0

    def reset(self, **kwargs) -> np.ndarray:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: (np.ndarray) the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, "
                               "wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError('Expected you to pass kwarg {} into reset'.format(key))
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def move_step_to_episode(self, outcome, fill_value):
        if outcome:
            fill_value = 3
        else:
            fill_value = -1

        self.episode_actions.append(self.fill_array(self.actions, fill_value).copy())
        self.actions.clear()
        self.episode_speed.append(self.fill_array(self.speeds, fill_value).copy())
        self.speeds.clear()
        self.episode_remain_time.append(self.fill_array(self.times, fill_value).copy())
        self.times.clear()
        self.episode_distance.append(self.fill_array(self.distance, fill_value).copy())
        self.distance.clear()

        self.mask.append([False if i < len(self.actions) else True for i in range(47)])

    def fill_array(self, array, fill_value):
        fill = [fill_value for i in range(47 - len(array))]
        concat = array + fill
        return concat

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """
        Step the environment with the given action

        :param action: (np.ndarray) the action
        :return: (Tuple[np.ndarray, float, bool, Dict[Any, Any]]) observation, reward, done, information
        """
        self.iter += 1
        if ( self.iter % 50000 == 0 and self.iter > 1):
            self.prepare_dataFrame()

        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)

        self.rewards.append(reward)
        self.actions.append(action)
        self.speeds.append(observation[0])
        self.times.append(observation[1])
        self.distance.append(observation[2])

        # if True:
        #    self.needs_reset = False
        if done or self.env.is_episodic == False:
            if self.env.is_episodic == False:
                self.needs_reset = False
            else:
                self.needs_reset = True

            if info['is_success']:
                self.episode_success.append(True)
            else:
                self.episode_success.append(False)
            self.move_step_to_episode(info['is_success'], sum(self.rewards))

            ep_rew = sum(self.rewards)
            eplen = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": eplen, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(ep_info)
                self.file_handler.flush()
            info['episode'] = ep_info
            self.rewards = []  # TODO: Added by Udesh
        self.total_steps += 1
        return observation, reward, done, info

    def prepare_dataFrame(self):
        new_list = []
        for index, (action, speed, time, distance, mask) in enumerate(zip(self.episode_actions,
                                          self.episode_speed,
                                          self.episode_remain_time,
                                          self.episode_distance,
                                        self.mask)):
            for step, (a,s,t,d, m) in enumerate(zip(action, speed, time, distance, mask)):
                new_list.append([step, index, a, s, t, d, self.episode_success[index], m])

        df = pd.DataFrame(new_list, columns=['step', 'episode number', 'action', 'speed', 'time', 'distance', 'success', 'm'], index=None)
        df.to_csv("Episode_data.csv")


    def close(self):
        """
        Closes the environment
        """
        self.prepare_dataFrame()
        with open('episode_actions.csv', 'w+', newline='') as myFile:
            wr = csv.writer(myFile)
            wr.writerows(self.episode_actions)

        super(Monitor_save_step_data, self).close()
        if self.file_handler is not None:
            self.file_handler.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return: (int)
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes

        :return: ([float])
        """
        return self.episode_rewards

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes

        :return: ([int])
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return: ([float])
        """
        return self.episode_times


class LoadMonitorResultsError(Exception):
    """
    Raised when loading the monitor log fails.
    """
    pass


def get_monitor_files(path: str) -> List[str]:
    """
    get all the monitor files in the given path

    :param path: (str) the logging folder
    :return: ([str]) the log files
    """
    return glob(os.path.join(path, "*" + Monitor_save_step_data.EXT))


def load_results(path: str) -> pandas.DataFrame:
    """
    Load all Monitor logs from a given directory path matching ``*monitor.csv`` and ``*monitor.json``

    :param path: (str) the directory path containing the log file(s)
    :return: (pandas.DataFrame) the logged data
    """
    # get both csv and (old) json files
    monitor_files = (glob(os.path.join(path, "*monitor.json")) + get_monitor_files(path))
    if not monitor_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (Monitor_save_step_data.EXT, path))
    data_frames = []
    headers = []
    for file_name in monitor_files:
        with open(file_name, 'rt') as file_handler:
            if file_name.endswith('csv'):
                first_line = file_handler.readline()
                assert first_line[0] == '#'
                header = json.loads(first_line[1:])
                data_frame = pandas.read_csv(file_handler, index_col=None)
                headers.append(header)
            elif file_name.endswith('json'):  # Deprecated json format
                episodes = []
                lines = file_handler.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                data_frame = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            data_frame['t'] += header['t_start']
        data_frames.append(data_frame)
    data_frame = pandas.concat(data_frames)
    data_frame.sort_values('t', inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame['t'] -= min(header['t_start'] for header in headers)
    # data_frame.headers = headers  # HACK to preserve backwards compatibility
    return data_frame

