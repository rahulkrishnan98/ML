import numpy as np
from matplotlib import pyplot as plt 

X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
])
# where -1 is bias unit
y = np.array([-1,-1,1,1,1])
#plotting examples
for d, sample in enumerate(X):
    # Plot the negative samples (the first 2)
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples (the last 3)
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

#svm function
def svm_sgd_plot(X, Y):
    #set up all parameters
    # set weight to number of parameters
    w= np.zeros(len(X[0]))
    #setting learning rate
    eta =1
    #set iterations
    iterations = 1000
    #error count
    error = [] 
    #training
    for current_iteration in range(1, iterations):
        error = 0
        for i, x in enumerate(X):
        #misclassification | numpy.dot returns the dot product(explained in class about it's significance)
            if (Y[i]*np.dot(X[i], w)) < 1:
                w =w + eta * (X[i] * Y[i] +(-2 *(1/current_iteration) * w))
                error =1
            else:
                #correct classification
                w = w + eta * (-2  *(1/current_iteration)* w) 
        errors.append(error)
    plt.plot(errors, '|')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()

    return w         
 