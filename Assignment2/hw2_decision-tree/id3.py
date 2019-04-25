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
import math # For log
import copy # for deepcopy

train=None
varnames=None
test=None
testvarnames=None
root=None

def entropy(p):
    r"""
    Calculate the entropy given a value that is between 0 and 1
    """
    if p == 0 or p == 1:
        return 0
    assert(type(p) is float),"Our fraction isn't a float!!!!!"
    return ((-p*math.log(p,2)) - ((1.-p)*(math.log((1.-p),2))))


def infogain(py_pxi, pxi, py, total):
    r"""
    Calculate the information gain from a particular branch
    We are exploiting the fact that this is a binomial tree
    """
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

def highest_info_gain(data,used):
    r"""
    Calls info gain and loops through all the remaining variables.
    Will return the variable that should be split on: ie the one with
    the highest info gain
    """
    highest = -1
    var = -1
    total = len(data)
    py = len([1 for row in data if (row[-1] == 1)])
    for col in range(len(data[0])-1):
        if col not in used:
            pxi = len([1 for row in data if (row[col] == 1)])
            py_pxi = len([1 for row in data if ((row[col] == 1) and (row[-1] == 1))])
            ig = infogain(py_pxi, pxi, py, total)
            if ig > highest:
                highest = ig
                var = col
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

def find_bigger(data):
    pos = 0
    neg = 0
    for i in range(len(data)):
        if data[i][-1] == 1:
            pos += 1
        else:
            neg += 1
    if pos > neg:
        return 1
    else:
        return 0

def check_if_pure(data):
    z = 0
    o = 0
    for i in range(len(data)):
        if data[i][-1] == 0:
            z += 1
        else:
            o += 1
    if z == len(data):
        return True,0
    elif o == len(data):
        return True,1
    else:
        return False,-1


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
    if len(data[0]) == len(used_attributes):
        return node.Leaf(varnames,find_bigger(data))

    else:
        split_on = highest_info_gain(data,used_attributes)
        used_attributes.append(split_on)
        new_used = copy.deepcopy(used_attributes)
        #print "Split on(",split_on,",\t",varnames[split_on],")"
        left, right = partition_data(data,split_on)
        l_pure, l_val = check_if_pure(left)
        r_pure, r_val = check_if_pure(right)
        if l_pure and r_pure:
            return node.Split(varnames, \
                              split_on, \
                              node.Leaf(varnames, \
                                        l_val),
                              node.Leaf(varnames, \
                                        r_val))
        elif l_pure:
            return node.Split(varnames, \
                              split_on, \
                              node.Leaf(varnames, \
                                        l_val),
                              build_tree(right, \
                                         varnames,  \
                                         copy.deepcopy(used_attributes)))
        elif r_pure:
            return node.Split(varnames, \
                              split_on, \
                              build_tree(left, \
                                         varnames, \
                                         copy.deepcopy(used_attributes)), \
                              node.Leaf(varnames, \
                                        r_val))

        else:
            return node.Split(varnames,\
                              split_on,
                              build_tree(left,\
                                         varnames,\
                                         copy.deepcopy(used_attributes)),\
                              build_tree(right,\
                                         varnames,\
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
    used = []
    used.append(len(varnames)-1)
    root = build_tree(train, varnames,used)
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
