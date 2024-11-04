import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotGen

font = {#'weight' : 'bold',
        'size'   : 16}
matplotlib.rc('font', **font)
def plot_mean_comparison(data1, data2,  savename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data1['steps'], data1['mean'], alpha=1, color='red', label='12 agents', linewidth=1.0)
    ax.plot(data2['steps'], data2['mean'], alpha=1, color='blue', label='6 agents', linewidth=1.0)
    plt.grid()
    ax.legend(loc='lower right')
    ax.set_ylabel("Mean Reward")
    ax.set_xlabel("Timestep")
    plt.savefig(savename + "Comp.png")
    plt.show()

def plot_completions_comparison(data1, data2,data3,  savename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data1['steps'], data1['plain'], alpha=1, color='red', label='12 agents', linewidth=1.0, linestyle='dashed')
    ax.plot(data3['steps'], data3['plain'], alpha=1, color='green', label='6 agents', linewidth=1.0, linestyle='dashdot' )
    ax.plot(data2['steps'], data2['plain'], alpha=1, color='blue', label='4 agents', linewidth=1.0)
    plt.grid()
    ax.legend(loc='upper left')
    ax.set_ylabel("Completions")
    ax.set_xlabel("Timestep")
    plt.savefig(savename + "Plain.png")
    plt.show()


def main():
    filename1 = "Mario12runCompletions.csv" #input("Name of first file: ")
    filename2 = "Mario4runCompletions.csv" #input("Name of second file: ")
    filename3 = "MarioStrange.csv"  # input("Name of second file: ")
    mario12C = "Mario12runRewardCut.csv"
    mario6C = "Mario6newRewardCut.csv"
    data_mario12 = plotGen.get_rolling_data(mario12C,20)
    data_mario6 = plotGen.get_rolling_data(mario6C,20)
    minecraft6C = "Minecraft4run.csv"
    minecraft4C = "Minecraft6runR.csv"
    data_minecraft6 = plotGen.get_rolling_data(minecraft4C, 50)
    data_minecraft4 = plotGen.get_rolling_data(minecraft6C, 50)
    savename = "MarioCutnew"#input("Name of save: ")
    savename2 = "MinecraftComp"
    data1 = plotGen.get_rolling_data(filename1, 1)
    data2 = plotGen.get_rolling_data(filename2, 1)
    data3 = plotGen.get_rolling_data(filename3, 1)
    plot_completions_comparison(data1, data2,data3, savename)
    plot_mean_comparison(data_mario12, data_mario6, savename)
    plot_mean_comparison(data_minecraft6, data_minecraft4, savename2)

if __name__ == "__main__":
    main()
