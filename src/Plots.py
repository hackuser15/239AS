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
    plt.hlines(y=0,xmin=0,xmax=max(x))
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
    #plt.show()
    plt.savefig(name+'.png')
    plt.clf()

def plotWorkFlow(data, num, label):
    workflow = data[data['Work-Flow-ID=work_flow_' + str(num)] == 1]
    num_operation = workflow['Size of Backup (GB)'].size
    plt.scatter(np.linspace(1, num_operation, num_operation), workflow['Size of Backup (GB)'])
    plt.xlabel('Operations')
    plt.ylabel('Size of Backup (GB)')
    plt.title('Workflow '+ str(num)+ ' ' + label)
    # plt.show()
    plt.savefig('Workflow_' + str(num) + '_' + label + '.png')
    plt.clf()