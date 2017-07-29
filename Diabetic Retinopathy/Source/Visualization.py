import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import proj3d

# visualizing confidence score as 5D scatter plot using 3D plot
def visualize5DData (X,scale,cmap,Labels):
    # Size of the figure
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Getting the values for each dimension
    im = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 3], s=X[:, 4] * scale, cmap=cmap,alpha = 1, picker = True)
    # Set label names for each dimensions
    ax.set_xlabel(Labels[0])
    ax.set_ylabel(Labels[1])
    ax.set_zlabel(Labels[2])
    # Color bar as fourth dimension
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel(Labels[3])
    objs = X[:, 4]
    # Circle size as fifth dimension
    max_size = np.amax(objs) * scale
    min_size = np.amin(objs) * scale
    handles, labels = ax.get_legend_handles_labels()
    display = (0, 1, 2)
    size_max = plt.Line2D((0, 1), (0, 0), color='k', marker='o', markersize=max_size, linestyle='')
    size_min = plt.Line2D((0, 1), (0, 0), color='k', marker='o', markersize=min_size, linestyle='')
    legend1 = ax.legend([handle for i, handle in enumerate(handles) if i in display] + [size_max, size_min],
                        [label for i, label in enumerate(labels) if i in display] + ["%.2f" % (np.amax(objs)),
                                                                                     "%.2f" % (np.amin(objs))],
                        labelspacing=1.5, title=Labels[4], loc=1, frameon=True, numpoints=1, markerscale=1)
    plt.show()



# Cinfusion matrix for predicted outputs
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    # Showing the values
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # Title and color
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # check for normalization
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# pie chart for accuracy
def PlotAccuracy(PredY,LabelY):
    correct_Pred=0
    for i in range(0,len(PredY)):
        if(PredY[i]==LabelY[i]):
            correct_Pred+=1
    # Accuracy caluculations
    Accuracy=correct_Pred/len(PredY)
    # Labels around pie chart
    Labels=["Accurate","Worng"]
    sizes=[Accuracy*100,(1-Accuracy)*100]
    explode=(0.1,0)
    plt.pie(sizes, explode=explode, labels=Labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')
    plt.show()



if __name__ == '__main__':

    X=np.loadtxt("visualization/Confidence_score.txt")
    scale = 1000
    cmap = plt.cm.spectral
    labels = [line.rstrip() for line
              in tf.gfile.GFile("Output/model/output_labels.txt")]
    # confidence score graph in 5 Dimensional
    visualize5DData(X, scale, cmap,labels)


    Y_pred=np.loadtxt("visualization/Predicted.txt",dtype=np.int16)
    Y_Labels=np.loadtxt("visualization/TrueLabels.txt",dtype=np.int16)
    cnf_matrix=confusion_matrix(Y_Labels,Y_pred)
    # confusion matrix without normalization
    plot_confusion_matrix(cnf_matrix, classes=labels,title='Confusion matrix, without normalization')
    # confusion matrix with normalization
    plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,title='Normalized confusion matrix')


    # Accuracy Pie Plot
    PlotAccuracy(Y_pred,Y_Labels)

