import matplotlib.pyplot as plt

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


#Plot of a ROC curve for a specific class
def plotROC(fpr,tpr,name):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(name+'.png')
    plt.clf()