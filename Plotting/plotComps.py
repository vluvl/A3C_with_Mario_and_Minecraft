import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotGen


def plot_completions_comparison(data1, data2, savename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data1['steps'], data1['plain'], alpha=1, color='red', label='12 agents', linewidth=1.0)
    ax.plot(data2['steps'], data2['plain'], alpha=1, color='blue', label='4 agents', linewidth=1.0)
    plt.grid()
    ax.legend(loc='lower right')
    ax.set_ylabel("Completions")
    ax.set_xlabel("Timestep")
    plt.savefig(savename + "Plain.png")
    plt.show()


def main():
    filename1 = "Mario12runCompletions.csv" #input("Name of first file: ")
    filename2 = "Mario4runCompletions.csv" #input("Name of second file: ")
    savename = "MarioCompletionsComparison"#input("Name of save: ")
    data1 = plotGen.get_rolling_data(filename1, 1)
    data2 = plotGen.get_rolling_data(filename2, 1)
    plot_completions_comparison(data1, data2, savename)


if __name__ == "__main__":
    main()
