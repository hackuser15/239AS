import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def scatterPlot(x, y, xlabel, ylabel, title, color):
    plt.scatter(x, y, color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
