import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_csv(filename):
    fields = ['Step', 'Value']
    data = pd.read_csv(filename, skipinitialspace=True, usecols=fields)
    return data

def get_rolling_data(filename, window_size):
    csv_data = load_csv(filename)
    rewards_series = pd.Series(csv_data['Value'])
    timesteps = pd.Series(csv_data['Step'])
    data = {'mean': rewards_series.rolling(window_size).mean(),
            'std' : rewards_series.rolling(window_size).std(),
            'var' : rewards_series.rolling(window_size).var(),
            'plain': rewards_series,
            'steps': timesteps}
    return data

def plot_mean_and_variance(data, savename):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the rolling mean on the left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Rolling Mean Reward', color=color)
    ax1.plot(data['steps'], data['mean'], color=color, label='Mean')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot the rolling variance on the right y-axis
    color = 'tab:red'
    ax2.set_ylabel('Rolling Variance', color=color)
    ax2.plot(data['steps'],data['var'], color=color, label='Variance')
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    plt.title('Rolling Mean and Variance Over Time')
    plt.grid()
    # Show and save the plot
    plt.savefig(savename + "MeanAndVariance.png")
    plt.show()

def plot_mean_and_deviation(data, savename):
    fig, ax = plt.subplots(figsize=(10, 6))
    #plt.style.use('ggplot')  # Change/Remove This If you Want

    #plot lines:
    ax.plot(data['steps'], data['mean'], alpha=1, color='red', label='Mean', linewidth=1.0)
    ax.fill_between(data['steps'], data['mean'] - 2 * data['std'],
                    data['mean'] + 2 * data['std'], color='#fa0000', alpha=0.2)

    #plot setting:
    plt.grid()
    ax.legend(loc='lower right')
    ax.set_ylabel("Mean Reward")
    ax.set_xlabel("Recorded Timestep")
    plt.savefig(savename + "MeanAndDeviation.png")
    plt.show()

def plot_mean(data, savename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['steps'], data['mean'], alpha=1, color='red', label='Mean', linewidth=1.0)
    plt.grid()
    ax.legend(loc='lower right')
    ax.set_ylabel("Mean Reward")
    ax.set_xlabel("Window Timestep")
    plt.savefig(savename + "Mean.png")
    plt.show()

def plot_plain_data(data, savename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['steps'], data['plain'], alpha=1, color='red', label='Completions', linewidth=1.0)
    plt.grid()
    ax.legend(loc='lower right')
    ax.set_ylabel("Completions")
    ax.set_xlabel("Timestep")
    plt.savefig(savename + "Plain.png")
    plt.show()

def main():
    filename = input("Name of file: ")
    savename = input("Name of save: ")
    window_size = int(input("Window size: "))
    savename = savename + "(" + str(window_size) + ")"
    data = get_rolling_data(filename, window_size)

    plot_mean_and_deviation(data, savename)
    plot_mean_and_variance(data, savename)
    plot_mean(data, savename)
    plot_plain_data(data, savename)


if __name__ == "__main__":
    main()
