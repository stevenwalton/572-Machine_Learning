#!/usr/bin/python
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd
#
#
import sys
import re
# Node class for the decision tree
import node
import math

r"""
ID3 algorithm (Examples, Target_Attributes, Attributes)
    - Create root node for the tree
    - If all examples are positive, return single-node tree Root with label +
    - If all examples are negative, return single-node tree Root with label -
    - If number of predicting attributes is empty, then return single node tree
        root with label = most common value of target attribute in examples
    - Else: Begin
        - A <- Attribute that best classifies examples
        - Decision tree attribute for root = A
        - For each possible value, v_i, of A
            - Add a new tree branch below root, corresponding to test A = v_i
            - Let examples(v_i) be the subset of examples that have the 
                value v_i for A
            - If: Examples(v_i) is empty
                - Below this new branch add a leaf node with label = most common
                    target value in the examples
            - Else: Below this new branch add subtree ID3(Examples(v_i), 
                    Target_Attribute, Attributes - {A})
    - End
    -Return Root
"""


train=None
varnames=None
test=None
testvarnames=None
root=None

def entropy(p): # -> float
    r"""
    Calculates the entropy given to it.
    p: parameter \in [0,1]
    Formula:
       -p*np.log2(p) - (1.-p)*np.log2(1.-p)
    """
    if (p < 0 or p > 1):
        print(p)
    assert(p >= 0 and p <= 1),"Input needs to be between 0 and 1"
    # Check for nans
    if (p == 0) or (p == 1):
        return 0
    e = ((-p*math.log(p,2)) - ((1.-p)*(math.log(1.-p,2))))
    assert(e >= 0 and e <= 1),"Entropy is bounded on [0,1]"
    return e

def infogain(py_pxi, pxi, py, total):
    r"""
    Returns the information gain of a particular step
    py_pxi: array containing the number of occurrences of y=1 w/ x_i=1 forall i=1,...,n
    pxi   : array containing the number of occurrences of x_i = 1
    py    : number of occurrences of y=1
    total : total length of data
    eg:
            [+9,-5]
            Humidity
           /        \
        [+3,-4]   [+6,-1]
         High       Low
        Information Gain = 
            (-9/14 log2(9/14) - 5/14 log2(5/14) )
             - 7/14( -3/7 log2(3/7) - 4/7 log2(4/7) )
             - 7/14( -6/7 log2(6/7) - 1/7 log2(1/7) )

        py_pxi = [3,6]
        pxi    = [7,7]
        py     = 9
        total  = 14
    """
    frac = float(py)/float(total)
    S = entropy(frac)
    if type(pxi) is list:
        l = len(pxi)
    else:
        l = 1
    max_ig = 1
    ig = 0
    for i in range(l):
        try:
            current_ent = entropy(float(py_pxi[i])/float(pxi[i]))
        except ZeroDivisionError:
            current_ent = 0
        if current_ent < max_ig:
            ig = current_ent
    return S-ig

def counts(data, constraint=None):
    r"""
    Pass in the data and get returned the number of instances and how
    many times class == 1
    returns array [[+x,total]]
    """
    if type(data[0]) is list:
        arr_py_pxi = [[0]*2 for i in range(len(data[0])-1)]
        arr_pxi    = [[0]*2 for i in range(len(data[0])-1)]
    else:
        arr_py_pxi = [[0,0]]
        arr_pxi    = [[0,0]]
    py = 0
    total = 0
    for row in range(len(data)):
        total += 1
        if data[row][-1] == 1: py += 1
        for col in range(len(data[0])-1):
            if col == constraint: continue
            elif data[row][col] == 1:
                arr_pxi[col][0] += 1
                if data[row][-1] == 1:
                    arr_py_pxi[col][0] += 1
            elif data[row][col] == 0:
                arr_pxi[col][1] += 1
                if data[row][-1] == 1:
                    arr_py_pxi[col][1] += 1
            else:
                print "Shouldn't be here"
                exit(1)
    return arr_py_pxi, arr_pxi,py,total

def partition_data(data,split_on=None):
    r"""
    returns the left and right partition
    binary split
    """
    left = []
    right = []
    for row in range(len(data)):
        if data[row][split_on] == 1:
            right.append(data[row])
        elif data[row][split_on] == 0:
            left.append(data[row])
        else:
            print "Can't be here"
            exit(1)
    return left,right

def highest_info_gain(array_py_pxi, array_pxi, py, total, used=[]):
    r"""
    Pass in a entropy and counts list and get returned the variable with the highest info 
    gain
    S = entropy
    c = counts list
    """
    highest_infoGain = -1
    var = -1
    for i in range(len(array_pxi)):
        if used and i in used: continue
        else:
            ig = infogain(array_py_pxi[i],array_pxi[i],py,total)
            #print "i:",i,"has ig:",ig
            if ig > highest_infoGain:
                highest_infoGain = ig
                var = i
    #raw_input("Pause")
    assert(var != -1),"VAR IS -1!!!!"
    return var

def root_node(data, varnames):
    r"""
    Returns the index of the root node
    """
    highest_entropy = -1
    var = -1
    row_count = 0
    yes = 0
    best_yes = -1
    best_col = -1
    for col in range(len(data[0])-1):
        for row in range(len(data)):
            if data[row][col] == 1:
                row_count += 1
                if data[row][-1] == 1:
                    yes += 1
        current_entropy = entropy(float(yes)/float(row_count))
        if current_entropy > highest_entropy:
            highest_entropy = current_entropy
            var = col
            best_col = row_count
            best_yes = yes
    return var,best_yes,best_col

def get_data_prime(data,root_node,pos):
    r"""
    Modify the data so that it only contains values for root node
    data is the original data
    root_node is the index of the variable of the root node
    pos is either 1 or 0, returning the cases where we have positive
    instances or negative instances (splitting left and right)
    """
    assert(pos == 1 or pos == 0)
    itr = 0
    dp = np.zeros(len(data[0])-1)
    # Hacky as shit, fix later, maybe
    for i in range(len(data)):
        if data[i][root_node] == pos:
            dp = np.hstack((data[i][:root_node],data[i][(root_node+1):]))
            itr = i+1
            break

    for i in range(itr,len(data)):
        if data[i][root_node] == pos:
            try:
                tmp = np.hstack((data[i][:root_node],data[i][(root_node+1):]))
            except:
                print "data[i][:root_node]",data[i][:root_node]
                print "data[i][root_node+1:]",data[i][root_node+1:]
                exit(1)
            dp = np.vstack((dp,tmp))
    return dp

    

# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable

# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)

# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)

# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames,used_attributes=[]):
    r"""
    Function used to build tree in a top down manner. 
    data: array of arrays containing the data [[row1],[row2],...,[rown]]
    varnames: array with variable names [header info]
    """
    arr_py_pxi, arr_pxi,py,total = counts(data)
    # Array of info gains
    if len(used_attributes) == len(arr_pxi):
        ind = used_attributes[-1]
        s = 0
        total = 0
        for row in range(len(data)):
            s += data[row][ind]
            total += data[row][-1]
        if s >= 0.5*total:
            x = 1
        else:
            x = 0
        return node.Leaf(varnames,x)
    split_on = highest_info_gain(arr_py_pxi, arr_pxi, py, total, used_attributes)
    used_attributes.append(split_on)
    #print "Split on",varnames[split_on]
    left, right = partition_data(data,split_on)

    pure_left = all((left[i][-1] == 1) for i in range(len(left))) or \
                all((left[i][-1] == 0) for i in range(len(left)))
    pure_right = all((right[i][-1] == 1) for i in range(len(right))) or \
                all((right[i][-1] == 0) for i in range(len(right)))

    if pure_left or pure_right:
        if left == []:
            lef = 0
        else:
            lef = left[0][-1]
        if right == []:
            rig = 0
        else:
            rig = right[0][-1]
        if pure_left and pure_right:
            return node.Split(varnames, split_on, node.Leaf(varnames,lef), node.Leaf(varnames,rig))
        elif pure_left:
            return node.Split(varnames, split_on, node.Leaf(varnames,lef), build_tree(right, varnames, used_attributes))
        else:
            return node.Split(varnames, split_on, build_tree(left,varnames, used_attributes), node.Leaf(varnames, rig))

    
    return node.Split(varnames,split_on,build_tree(left,varnames,used_attributes), build_tree(right,varnames, used_attributes))

# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS,testS,modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    global used 
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)

def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct)/len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print 'Usage: id3.py <train> <test> <model>'
        sys.exit(2)
    loadAndTrain(argv[0],argv[1],argv[2])

    acc = runTest()
    print "Accuracy: ",acc

if __name__ == "__main__":
    main(sys.argv[1:])
