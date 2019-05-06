import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import matplotlib
# Plotting style
plt.style.use('fivethirtyeight')
font = {'family': 'normal',
        'weight': 'bold',
        'size': 23}
matplotlib.rc('font',**font)


# Number of points
N = 100
# Dimensions that we want to test
dim_set = [5,10,20,50,100]

def gen_data(dims):
    r"""
    Generates the training data where the first 50 points
    have 0 in the first dimension and 1 for the next 50 
    points
    """
    training_data = np.random.randint(2,size=(N,dims))
    training_data[:50,0] = np.ones(len(training_data[:50]))
    training_data[51:,0] = np.zeros(len(training_data[51:]))
    return training_data

def get_nn(train,validation,lp=1):
    r"""
    Get the nearest neighbors
    """
    dist = [np.linalg.norm(validation[0]- train[i],lp) for i in range(len(train)      )]
    sorted_dist = np.argsort(dist)
    for i in range(1,len(validation)):
        d = [np.linalg.norm(validation[i]-train[j],lp) for j in range(len(train))      ]
        dist = np.vstack((dist,d))
        sorted_dist = np.vstack((sorted_dist,np.argsort(d)))
    return dist[:,0], sorted_dist[:,0]

def average_nn(dims,times=10):
    r"""
    Average the results of the nearest neighbor classification
    """
    acc = np.zeros(N)
    for t in range(times):
        train = gen_data(dims)
        test  = gen_data(dims)
        _, s_dist = get_nn(train,test)
        for i in range(N):
            if test[i][0] == train[s_dist[i]][0]:
                acc[i] += 1
    acc = [acc[i]/times for i in range(N)]
    acc = np.average(acc)
    return acc

def main():
    y = [average_nn(dims) for dims in dim_set]
    plt.title("Accuracy of 1-NN With Varying Dimensions")
    plt.xlabel("Number of dimensions")
    plt.ylabel("Accuracy")
    plt.plot(dim_set,y)
    plt.show()

if __name__ == '__main__':
    main()
