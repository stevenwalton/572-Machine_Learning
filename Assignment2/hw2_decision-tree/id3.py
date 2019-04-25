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
import copy


train=None
varnames=None
test=None
testvarnames=None
root=None

# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
    if p == 0 or p == 1:
        return 0
    return -((p*math.log(p,2)) + ((1.-p)*(math.log((1.-p),2))))


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    S = entropy(py/float(total))
    try:
        left = ((total-pxi)/float(total)) * entropy(float(py-py_pxi)/float(total - pxi))
    except ZeroDivisionError:
        left = 0
    try:
        right = (pxi/float(total))*entropy(py_pxi/float(pxi))
    except ZeroDivisionError:
        right = 0
    return S - left - right

# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable
def highest_info_gain(data,used=[]):
    highest = -1
    var = -1
    total = len(data)
    py = len([1 for row in data if (row[-1] == 1)])
    for col in range(len(data[0])-1):
    #for col in range(len(data[0])):
        if col not in used:
            pxi = len([1 for row in data if (row[col] == 1)])
            py_pxi = len([1 for row in data if ((row[col] == 1) and (row[-1] == 1))])
            ig = infogain(py_pxi, pxi, py, total)
            if ig > highest:
                highest = ig
                var = col
    #print "split on",var
    return var

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
def build_tree(data, varnames, used_attributes=[]):
    split_on = highest_info_gain(data,used_attributes)
    if split_on == -1:
        s = 0
        total = 0
        #ind = used_attributes[-1]
        ind = split_on
        for row in range(len(data)):
            s += data[row][ind]
            total += data[row][-1]
        if s>= 0.5*total:
            x = 1
        else:
            x = 0
        return node.Leaf(varnames,x)
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
            return node.Split(varnames, \
                              split_on, \
                              node.Leaf(varnames,lef), \
                              node.Leaf(varnames,rig))
        elif pure_left:
            return node.Split(varnames, \
                              split_on, \
                              node.Leaf(varnames,lef), \
                              build_tree(right, \
                                         varnames, \
                                         copy.deepcopy(used_attributes)))
        else:
            return node.Split(varnames, \
                              split_on, \
                              build_tree(left,\
                                         varnames, \
                                         copy.deepcopy(used_attributes)), \
                              node.Leaf(varnames, rig))

    return node.Split(varnames,\
                      split_on,\
                      build_tree(left,\
                                 varnames,\
                                 copy.deepcopy(used_attributes)), \
                      build_tree(right,\
                                 varnames, \
                                 copy.deepcopy(used_attributes)))


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
