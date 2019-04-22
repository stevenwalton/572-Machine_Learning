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


train=None
varnames=None
test=None
testvarnames=None
root=None

# Helper function computes entropy of Bernoulli distribution with
# parameter p
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


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
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
    assert(len(py_pxi) == len(pxi)),"py_pxi and pxi are not the same length"
    denom = float(np.sum(pxi))
    #print("pxi:",pxi)
    #print("len:",len(pxi))
    for i in range(len(pxi)):
        #print("py_pxi[",i,"] =",py_pxi[i])
        #print("pxi[",i,"] =",pxi[i])
        #print(float(py_pxi[i])/float(pxi[i]))
        S -= pxi[i]/denom * entropy(float(py_pxi[i])/float(pxi[i]))
    return S

def counts(data, used, constraint=None):
    r"""
    Pass in the data and get returned the number of instances and how
    many times class == 1
    c holds the counts and is an array of arrays. [[# of 1's, # class == 1],...]
    """
    c = np.zeros([len(data[0])-1, 2]) # rows, columns
    if constraint == None: # unconstrained
        for row in range(len(data)):
            for col in range(len(data[0])-1):
                if used[col] == 0 and data[row][col] == 1:
                    c[col][0] += 1
                    if data[row][-1] == 1:
                        c[col][1] += 1
    else: # Constrained (for info gain)
        for row in range(len(data)):
            for col in range(len(data[0])-1):
                if used[col] == 0 and data[row][constraint] == 1 and data[row][col] == 1:
                    c[col][0] += 1
                    if data[row][-1] == 1:
                        c[col][1] += 1
    #print(c)
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
        current_entropy = S - entropy(float(c[i,1])/float(c[i,0]))
        if current_entropy > highest_entropy:
            highest_entropy = current_entropy
            var = i
    for i in range(len(varnames)):
        if varnames[i] == var:
            used[i] = 1
    print varnames[var]
    return var
    #return 0

#def determine_next_node(counts,data,used):
def determine_next_node(data,given=[]):
    r"""
    returns the variable with the highest entropy
    """
    c = counts(data,used,given)
    print c
    return 0
    #var = -1 
    #running_entropy = 0
    #current_entropy = 0
    #for i in range(len(counts)):
    #    if used[i] == 0:
    #        e = float(counts[i][1])/float(counts[i][0])
    #        current_entropy = entropy(e)
    #        if current_entropy > running_entropy:
    #            running_entropy = current_entropy
    #            var = i
    #    else:
    #        current_entropy = 0
    #assert(var != -1)
    #assert(running_entropy != 0),"Highest entropy is 0"
    #print("Highest entropy on var: ",var)
    #h = highest_info_gain(counts,var,data,used)
    #print("Highest info gain from: ",h)
    ##print("\n")
    #return var

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
    #highest_entropy = 0
    ##var = "" 
    #for i in range(len(c)):
    #    #print("Frac:" + str(c[i,1]) + "/" + str(c[i,0]) + "=" +str(c[i,1]/c[i,0]))
    #    current_entropy = S - entropy(c[i,1]/c[i,0])
    #    #print("Current entropy: " + str(current_entropy))
    #    if current_entropy > highest_entropy:
    #        highest_entropy = current_entropy
    #        var = i
    #        #print var

    #for i in range(len(varnames)):
    #    if varnames[i] == var:
    #        used[i] = 1

    #print "[+" + str(yes) + ",-" + str(rows-yes) +"]"
    #print "Total: " + str(rows)
    #print "Len data:" + str(len(data))
    #print "Num vars:" + str(len(data[0])) + " " + str(len(varnames))
    #print "Total entropy: " + str(total_entropy)
    #raw_input("Pause")
    return var

        
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
    rn = root_node(data,varnames)
    root_name = varnames[rn]
    left = determine_next_node(data,rn)
    #print(rn)
    # >>>> YOUR CODE GOES HERE <<<<
    # For now, always return a leaf predicting "1":
    #used = np.zeros(len(varnames)-1)
    #used[4] = 1
    #cnt = counts(data, used, 4)
    #cnt = counts(data, used)
    #root_node = determine_next_node(cnt, data, used)
    #node.Leaf(varnames, root_node)
    #used[root_node] = 1
    #print cnt
    #for i in range(len(varnames)-1):
    #    #if used[4] == 1: print "4 is used"
    #    #else: print "4 is not used"
    #    next_node = determine_next_node(cnt,used)
    #    used[next_node] = 1
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
