import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def scatterPlot(x, y, xlabel, ylabel, title, c, name):
    plt.scatter(x, y, color=c)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.show()
    plt.savefig(name+'.png')
    plt.clf()

def residualPlot(x, y, xlabel, ylabel, title, c, name):
    plt.scatter(x, y, color=c)
    plt.hlines(y=0,xmin=0,xmax=50)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.show()
    plt.savefig(name+'.png')
    plt.clf()

def linePlot(x, y, xlabel, ylabel, title, c, name):
    plt.plot(x, y, color=c)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.savefig(name+'.png')
    plt.clf()
