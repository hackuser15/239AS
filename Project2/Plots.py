import matplotlib.pyplot as plt
import plotly.plotly as py

def linePlot(x, y, xlabel, ylabel, title, c, name,num_mapping,labels):
    plt.plot(num_mapping, y, color=c)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(num_mapping,labels)
    plt.title(title)
    plt.show()
    #plt.savefig(name+'.png')
    #plt.clf()


def histogram(y, xlabel, ylabel, title, c, name,num_mapping,labels):

    print(y)
    plt.hist(y,color=c)
    plt.xticks(y, labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #plt.xticks(num_mapping,labels)
    plt.title(title)
    plt.show()
    #plt.savefig(name+'.png')
    #plt.clf()


def barPlot(D):
    plt.bar(range(len(D)), D.values(), align='center',color = "red")
    plt.xticks(range(len(D)), D.keys())
    plt.show()
