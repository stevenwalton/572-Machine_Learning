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

# Needed for math
import numpy as np

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

Rewritten:
    Run counts(data) to get a matrix holding positive examples and total examples
    if c[x,0] == c[x,1]: # All examples are positive
        return single-node tree Root with label +
    elif c[x,0] == 0: # All examples are negative
        return single-node tree Root with label -
    else:



Calc best attr
find one with most infogain
recurse
        
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
    e = ((-p*np.log2(p)) - ((1.-p)*(np.log2(1.-p))))
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
    if (frac > 1 or frac < 0):
        print frac
        print(py)
        print(total)
    S = entropy(frac)
    #assert(len(py_pxi) == len(pxi)),"py_pxi and pxi are not the same length"
    for i in range(len(pxi)):
        S -= pxi[i]/total * entropy(float(py_pxi[i])/float(pxi[i]))
    #S -= pxi/denom * entropy(float(py_pxi)/float(pxi))
    return S

# Rewrite with only data def counts(data):
# def counts(data):
def counts(data, used, constraint=None):
    r"""
    Pass in the data and get returned the number of instances and how
    many times class == 1
    returns array [[+x,total]]
    """
    c = np.zeros([len(data[0])-1, 2]) # rows, columns
    if constraint == None: # unconstrained
        for row in range(len(data)):
            for col in range(len(data[0])-1):
                if used[col] == 0 and data[row][col] == 1:
                    c[col][1] += 1
                    if data[row][-1] == 1:
                        c[col][0] += 1
    else: # Constrained (for info gain)
        for row in range(len(data)):
            for col in range(len(data[0])-1):
                if used[col] == 0 and data[row][constraint] == 1 and data[row][col] == 1:
                    c[col][1] += 1
                    if data[row][-1] == 1:
                        c[col][0] += 1
    #print(c)
    #raw_input("Pause")
    return c

def highest_info_gain(S,c):
    r"""
    Pass in a entropy and counts list and get returned the variable with the highest info 
    gain
    S = entropy
    c = counts list
    """
    highest_entropy = 0 
    for i in range(len(c)):
        #current_entropy = S - entropy(float(c[i,1])/float(c[i,0]))
        current_entropy = S - entropy(float(c[i,0])/float(c[i,1]))
        if current_entropy > highest_entropy:
            highest_entropy = current_entropy
            var = i
    for i in range(len(varnames)):
        if varnames[i] == var:
            used[i] = 1
    return var

#def determine_next_node(counts,data,used):
def determine_next_node(data,given=[]):
    r"""
    returns the variable with the highest entropy
    We can actually cheat a bit. Since the formula is S - sum S_x
    we can actually look for the smallest entropy given a condition
    We do the inverse of highest_info_gain
    """
    c = counts(data,used,given)
    lowest_entropy = 1 # Because this is max
    for i in range(len(c)):
        #current_entropy = entropy(float(c[i,1])/float(c[i,0]))
        current_entropy = entropy(float(c[i,0])/float(c[i,1]))
        if current_entropy < lowest_entropy:
            lowest_entropy = current_entropy
            var = i
    for i in range(len(varnames)-1):
        if varnames[i] == var:
            used[i] = 1
    #print varnames[i]
    return var
    #return 0

def next_node(c,data,given):
    r"""
    Take in the previous counts, all data, and the given variable
    Calculate the entropy for the each variable given the previous 
    condition and then return the variable with the highest information gain.
    """
    cnt = c[given]
    py     = cnt[0]
    total  = cnt[1]

    cnt = counts(data,used,given)
    #print(cnt)
    py_pxi = cnt[:,0]
    pxi    = cnt[:,1]
    S = np.zeros(len(data[0]))
    for i in range(len(data[0])-1):
        S[i] = infogain(py_pxi[i], pxi[i], py, total)
    #print S
    #print(np.max(S))
    for i in range(len(S)):
        if S[i] == np.max(S):
            #print i
            return i
    return 0

def root_node(data, varnames):
    r"""
    Returns the variable for the root node
    """
    rows = 0
    yes  = 0
    for row in range(len(data)):
        if data[row][-1] == 1:
            yes += 1
        rows += 1
    S = entropy(float(yes)/float(rows))
    c = counts(data, used)
    var = highest_info_gain(S,c)
    print(varnames[var])
    return var

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
    dp = np.zeros(len(data[0]))
    # Hacky as shit, fix later, maybe
    for i in range(len(data)):
        if data[i,root_node] == pos:
            dp = data[i]
            itr = i
            break

    for i in range(itr,len(data)):
        if data[i,root_node] == pos:
            np.vstack((dp,data[i]))

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
def build_tree(data, varnames):
    r"""
    Function used to build tree in a top down manner. 
    data: array of arrays containing the data [[row1],[row2],...,[rown]]
    varnames: array with variable names [header info]
    """
    base_counts = counts(data, used)
    print base_counts
    rn = root_node(data,varnames)
    print rn
    raw_input("Pause")
    root_name = varnames[rn]
    next_node(base_counts, data, rn)
    #left = determine_next_node(data,rn)
    #right= determine_next_node(data,rn)
    return node.Leaf(varnames, 1)
    #return node.Leaf(varnames, root_node)


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
    used = np.zeros(len(varnames)-1)

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)
    print("Current tree:")
    root.write(sys.stdout,0)

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
