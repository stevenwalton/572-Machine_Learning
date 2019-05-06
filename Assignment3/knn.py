import numpy as np
import statistics as stat

def read_data(datafile,d=','):
    r"""
    Reads the datafile and returns features and classification
    """
    data = np.genfromtxt(datafile,delimiter=d)
    data = data[1:]
    #print(data[:][3]/ 1000)
    #data[:,4] /= 1000
    data[:,1] /= stat.stdev(data[:,1])
    data[:,4] /= stat.stdev(data[:,4])
    data[:,5] /= stat.stdev(data[:,5])
    features = data[:,1:-1]
    classification = data[:,-1]
    return np.asarray(features),np.asarray(classification)

def read_toTrain(datafile,d=','):
    r"""
    Reads datafile of data TO BE classified
    """
    data = np.genfromtxt(datafile,delimiter=d)
    data = data[1:]
    data[:,3] /= 1000
    return np.asarray(data)

#def knn(x0, train_data, train_clas,v_data, v_class,lp=2):
#    r"""
#    """
#    dist = [np.linalg.norm(x0[0] - train_data[j],lp) for j in range(len(train_data))]
#    for i in range(1,len(x0)):
#        dist = np.vstack((dist,[np.linalg.norm(x0[i] - train_data[j],lp) for j in range(len(train_data))]))
#    sorted_dist = [sorted(dist[i]) for i in range(len(dist))]
#    #for i,k in zip(range(len(dist)),range(1,100)):
#    classification = np.zeros(len(dist))
#    #k = train(dist,train_clas)
#    for k in range(2,100):
#        for i in range(len(dist)):
#            index = [np.where(dist==sorted_dist[i][p])[0][0] for p in range(k)]
#            print(index)
#            if np.sum([train_clas[index[p]] for p in range(k)]) >= k/2.:
#                print("Classified")
#                classification[i] = 1
#        if validate(k,v_data,v_class):
#            return k,classification
#        
#    print("FAILED")
#    return -1,-1

def neighbors(validation, t_data, lp=2):
    r"""
    Finds the nearest neighbors
    returns back the full list of neighbors and indices in sorted order
    """
    dist = [np.linalg.norm(validation[0] - t_data[i],lp) for i in range(len(t_data))]
    sorted_dist = np.argsort(dist)
    for i in range(1,len(validataion)):
        d = [np.linalg.norm(validation[i]-t_data[j],lp) for j in range(len(t_data))]
        dist = np.vstack((dist,d))
        sorted_dist = np.vstack((sorted_dist,np.argsort(d)))
    return dist, sorted_dist

def train(train_data,train_clas,v_data, v_clas):
    r"""
    Trains the KNN
    """
    dist, s_dist = neighbors(train_data)
    #v = [
    for k in range(1,100):
        pass


def main():
    f,c = read_data("data.csv")
    # Training data and classification
    t_data = f[:6]
    t_clas = c[:6]
    # Validation data and classification
    v_data = f[6:]
    v_clas = c[6:]
    knn(1,t_data)
    #d = read_toTrain("unclassified.csv")
    #dist = knn(i,d,t_data,t_clas)
    #k,classification = knn(d,t_data,t_clas,v_data,v_clas)
    print(np.min(dist))

if __name__ == '__main__':
    main()
