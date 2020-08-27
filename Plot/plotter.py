import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#from stable_baselines.results_plotter import ts2xy, load_results

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve', window_size=360):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=window_size)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

def load_result_from_csv(lower, upper, file_name='../RL_Agent/Vehicle/Episode_data.csv'):
    df = pd.read_csv(file_name)

    def func(df, lower, upper):
        data = df[(df['episode number'] > lower) & (df['episode number'] < upper)]
        return data


    df = func(df, lower ,upper)

    annot = df.pivot('episode number', 'step', 'speed')
    df = df.pivot('episode number', 'step', 'action')

    print(df)

    sns.heatmap(df, annot=annot,
                vmin=-1, vmax=3
                )
    plt.show()

file_name='C:/Users/pgunarathna/PycharmProjects/LaneManager/backups/Vehicle/Trained_400000_64x3_lr_5_SMARTS/Episode_data.csv'
if __name__ == '__main__':
    load_result_from_csv(11650,11700,
                         #file_name=file_name
                         )