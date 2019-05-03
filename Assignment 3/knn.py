import numpy as np

def read_data(datafile,d=','):
    r"""
    Reads the datafile and returns features and classification
    """
    data = np.genfromtxt(datafile,delimiter=d)
    data = data[1:]
    #print(data[:][3]/ 1000)
    data[:,4] /= 1000
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

def knn(x0, train_data, train_clas,v_data, v_class,lp=2):
    r"""
    """
    dist = [np.linalg.norm(x0[0] - train_data[j],lp) for j in range(len(train_data))]
    for i in range(1,len(x0)):
        dist = np.vstack((dist,[np.linalg.norm(x0[i] - train_data[j],lp) for j in range(len(train_data))]))
    sorted_dist = [sorted(dist[i]) for i in range(len(dist))]
    #for i,k in zip(range(len(dist)),range(1,100)):
    classification = np.zeros(len(dist))
    #k = train(dist,train_clas)
    for k in range(2,100):
        for i in range(len(dist)):
            index = [np.where(dist==sorted_dist[i][p])[0][0] for p in range(k)]
            print(index)
            if np.sum([train_clas[index[p]] for p in range(k)]) >= k/2.:
                print("Classified")
                classification[i] = 1
        if validate(k,v_data,v_class):
            return k,classification
        
    print("FAILED")
    return -1,-1

#def train(dist,train_clas):
#    sorted_dist = [sorted(dist[i]) for i in range(len(dist))]
#    for i in range(len(dist)):
#        index = [np.where(dist==sorted_dist[p])[0][0] for p in range(k)]
#        if np.sum([train_clas[index[p]] for i in range(k)]) >= 1./k:
#            classification[i] = 1

def validate(k, v_data, v_class,lp=2):
    v_sorted = [sorted(v_data[i]) for i in range(len(v_data))]
    c = np.zeros(len(v_class))
    for i in range(len(v_data)):
        index = [np.where(v_data==v_sorted[p])[0][0] for p in range(k)]
        if np.sum([v_class[index[p]] for p in range(k)]) >= k/2.:
            print("encoded!")
            c[i] = 1
    print(c)
    print(v_class)
    if c is v_class:
        return True
    return False

def main():
    f,c = read_data("data.csv")
    # Training data and classification
    t_data = f[:6]
    t_clas = c[:6]
    # Validation data and classification
    v_data = f[6:]
    v_clas = c[6:]
    d = read_toTrain("unclassified.csv")
    #dist = knn(i,d,t_data,t_clas)
    k,classification = knn(d,t_data,t_clas,v_data,v_clas)
    print(np.min(dist))

if __name__ == '__main__':
    main()
