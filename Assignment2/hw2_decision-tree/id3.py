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

# Needed for math
#import numpy as np

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
    #e = ((-p*np.log2(p)) - ((1.-p)*(np.log2(1.-p))))
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
    for i in range(len(pxi)):
        if pxi[i] == 0 and py_pxi[i] == 0: continue
        else:
            S -= pxi[i]/total * entropy(float(py_pxi[i])/float(pxi[i]))
    return S

# Rewrite with only data def counts(data):
# def counts(data):
#def counts(data, used, constraint=None):
def counts(data, constraint=None):
    r"""
    Pass in the data and get returned the number of instances and how
    many times class == 1
    returns array [[+x,total]]
    """
    arr_py_pxi = [[0]*2 for i in range(len(data[0])-1)]
    arr_pxi    = [[0]*2 for i in range(len(data[0])-1)]
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
    ##c = [[0]*2]*len(data[0]-1)
    ##py_pxi = [0]*(len(data[0])-1)
    ##pxi    = [0]*(len(data[0])-1)
    #for row in range(len(data)):
    #    for col in range(len(data[0])-1):
    #        if col == constraint: continue
    #        elif data[row][col] == 1:
    #            pxi[col] += 1
    #            if data[row][-1] == 1:
    #                py_pxi[col] += 1
    #print "py_pxi"
    #print py_pxi
    #print "pxi"
    #print pxi
    #return py_pxi,pxi

            
    #   c = np.zeros([len(data[0])-1, 2]) # rows, columns
    #   #try:
    #   #    c = np.zeros([len(data[0])-1, 2]) # rows, columns
    #   #except TypeError:
    #   #    print "Data[0] is an integer"
    #   #    print data
    #   #    exit(1)
    #   #for row in range(len(data)):
    #   #    for col in range(len(data[0])-1):
    #   #        if used[col] == 0 and data[row][col] == 1:
    #   #            c[col][1] += 1
    #   #            if data[row][-1] == 1:
    #   #                c[col][0] += 1
    #   for row in range(len(data)):
    #       for col in range(len(data[0])-1):
    #           if col != constraint and data[row][col] == 1:
    #               c[col][1] += 1
    #               if data[row][-1] == 1:
    #                   c[col][0] += 1
    #   return c
    #   #else: # Constrained (for info gain)
    #   #    for row in range(len(data)):
    #   #        for col in range(len(data[0])-1):
    #   #            if used[col] == 0 and data[row][constraint] == 1 and data[row][col] == 1:
    #   #                c[col][1] += 1
    #   #                if data[row][-1] == 1:
    #   #                    c[col][0] += 1
    #   #print(c)
    #   #raw_input("Pause")
    #   #return c

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



#def highest_info_gain(S,c):
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
            if ig > highest_infoGain:
                highest_infoGain = ig
                var = i
    assert(var != -1),"VAR IS -1!!!!"
    return var

    #   highest_entropy = -1
    #   #print S
    #   #print c
    #   for i in range(len(c)):
    #       #current_entropy = S - entropy(float(c[i,1])/float(c[i,0]))
    #       #print c[i,0]
    #       #print c[i,1]
    #       #print(entropy(float(c[i,0])/float(c[i,1])))
    #       if c[i,1] == 0:
    #           continue
    #       current_entropy = S - entropy(float(c[i,0])/float(c[i,1]))
    #       #print current_entropy
    #       if current_entropy > highest_entropy:
    #           highest_entropy = current_entropy
    #           var = i
    #   #if current_entropy == 0:
    #   #    print "Current entropy:",current_entropy
    #   #    print "S:",S
    #   #    print "num:",c[i,0]
    #   #    print "denom:",c[i,1]
    #   #    print entropy(float(c[i,0])/float(c[i,1]))
    #   #    print "var:",var
    #   for i in range(len(varnames)):
    #       if varnames[i] == var:
    #           used[i] = 1
    #   return var

#def determine_next_node(counts,data,used):
#def determine_next_node(data,given=[]):
#    r"""
#    returns the variable with the highest entropy
#    We can actually cheat a bit. Since the formula is S - sum S_x
#    we can actually look for the smallest entropy given a condition
#    We do the inverse of highest_info_gain
#    """
#    c = counts(data,used,given)
#    lowest_entropy = 1 # Because this is max
#    for i in range(len(c)):
#        #current_entropy = entropy(float(c[i,1])/float(c[i,0]))
#        current_entropy = entropy(float(c[i,0])/float(c[i,1]))
#        if current_entropy < lowest_entropy:
#            lowest_entropy = current_entropy
#            var = i
#    for i in range(len(varnames)-1):
#        if varnames[i] == var:
#            used[i] = 1
#    #print varnames[i]
#    return var
#    #return 0

#def next_node(c,data,given):
#    r"""
#    Take in the previous counts, all data, and the given variable
#    Calculate the entropy for the each variable given the previous 
#    condition and then return the variable with the highest information gain.
#    """
#    cnt = c[given]
#    py     = cnt[0]
#    total  = cnt[1]
#
#    cnt = counts(data,used,given)
#    #print(cnt)
#    py_pxi = cnt[:,0]
#    pxi    = cnt[:,1]
#    S = np.zeros(len(data[0]))
#    for i in range(len(data[0])-1):
#        S[i] = infogain(py_pxi[i], pxi[i], py, total)
#    #print S
#    #print(np.max(S))
#    for i in range(len(S)):
#        if S[i] == np.max(S):
#            #print i
#            return i
#    return 0

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
    #rows = 0
    #yes  = 0
    #for row in range(len(data)):
    #    if data[row][-1] == 1:
    #        yes += 1
    #    rows += 1
    #S = entropy(float(yes)/float(rows))
    #c = counts(data)
    #if S == 0:
    #    return 0
    #var = highest_info_gain(S,c)
    ##print(varnames[var])
    #print(varnames)
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
            #dp = data[i][:root_node] + data[i][(root_node+1):]
            itr = i+1
            break

    #print(len(dp))
    #print dp
    for i in range(itr,len(data)):
        if data[i][root_node] == pos:
            try:
                tmp = np.hstack((data[i][:root_node],data[i][(root_node+1):]))
            except:
                print "data[i][:root_node]",data[i][:root_node]
                print "data[i][root_node+1:]",data[i][root_node+1:]
                exit(1)
            #dp = np.vstack((dp,data[i][:root_node] + data[i][(root_node+1):]))
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
    #try:
    #    print "len:",len(data[0])
    #    print "len data",len(data)
    #    if len(data) <= 3:
    #        print data
    #except:
    #    print "Data is a vector"
    #    print data
    #    exit(1)
    #try:
    #    len(data)
    #    len(data[0])
    #except:
    #    print "data",data
    #    #print "data[0]",data[0]
    #    print "t1:",type(data)
    #    print "type:",type(data[0])
    #    exit(1)
    #if type(data[0]) == int or not type(data[0]):
    #if data == [] or data[0] == [] or type(data[0]) == int:
    if len(data)==1:
        print "Len data == 1"
        #print "data",data
        #print "data[0]",data[0]
        s = sum(data[0])
        #print data[-1]
        if data[0][-1] == 1:
            print "Returning 1"
            return node.Leaf(varnames,1)
        else:
            print "Returning 0"
            return node.Leaf(varnames,0)
    arr_py_pxi, arr_pxi,py,total = counts(data)
    # Array of info gains
    print "used",used_attributes
    split_on = highest_info_gain(arr_py_pxi, arr_pxi, py, total, used_attributes)
    used_attributes.append(split_on)
    print "Split on",varnames[split_on]
    left, right = partition_data(data,split_on)
    pure_left = all((left[i][-1] == 1) or (left[i][-1] == 0) for i in range(len(left)))
    pure_right = all((right[i][-1] == 1) or (right[i][-1] == 0) for i in range(len(right)))
    if pure_left or pure_right:
        if pure_left and pure_right:
            return node.Split(varnames, split_on, node.Leaf(varnames,left[0][-1]), node.Leaf(varnames,right[0][-1]))
        elif pure_left:
            return node.Split(varnames, split_on, node.Leaf(varnames,left[0][-1]), build_tree(right, varnames, used_attributes))
        else:
            return node.Split(varnames, split_on, build_tree(left,varnames, used_attributes), node.Leaf(varnames, right[0][-1]))

    
    return node.Split(varnames,split_on,build_tree(left,varnames,used_attributes), build_tree(right,varnames, used_attributes))
    #raw_input("Pause")
    #left_all_pos = False
    #left_all_neg = False
    #right_all_pos = False
    #right_all_neg = False

    # node.Split(names, var, left,right)
    #return node.Split(varnames,split_on,build_tree(left,varnames,split_on), build_tree(right,varnames, split_on))
    #if all(sum(left[i])==len(left[i]) for i in range(len(left))):
    #    left_all_pos == True
    #elif all(sum(left[i])==0 for i in range(len(left))):
    #    left_all_neg == True
    #if all(sum(right[i])==len(right[i]) for i in range(len(right))):
    #    right_all_pos == True
    #elif all(sum(right[i])==0 for i in range(len(right))):
    #    right_all_neg == True

    #if left_all_pos and right_all_pos:
    #    return Node.Split(varnames,split_on,node.Leaf(varnames,1),node.Leaf(varnames,1))
    #elif left_all_neg and right_all_neg:
    #    return Node.Split(varnames,split_on,node.Leaf(varnames,0),node.Leaf(varnames,0))
    #elif left_all_pos and right_all_neg:
    #    return Node.Split(varnames,split_on,node.Leaf(varnames,1),node.Leaf(varnames,0))
    #elif left_all_neg and right_all_pos:
    #    return Node.Split(varnames,split_on,node.Leaf(varnames,0),node.Leaf(varnames,1))
    #elif left_all_neg:
    #    return Node.Split(varnames,split_on,node.Leaf(varnames,0),build_tree(right,varnames,used_attributes))
    #elif left_all_pos:
    #    return Node.Split(varnames,split_on,node.Leaf(varnames,1),build_tree(right,varnames,used_attributes))
    #elif right_all_neg:
    #    return Node.Split(varnames,split_on,build_tree(right,varnames,used_attributes),node.Leaf(varnames,0))
    #elif right_all_pos:
    #    return Node.Split(varnames,split_on,build_tree(right,varnames,used_attributes),node.Leaf(varnames,1))
    #else:
    #    return node.Split(varnames,split_on,build_tree(left,varnames,used_attributes), build_tree(right,varnames, used_attributes))











    #py_pxi, pxi = counts(data,rn)
    #ig = infogain(py_pxi,pxi,py,total)
    

    #   #print base_counts
    #   # First find the root node
    #   rn = root_node(data,varnames)
    #   #print varnames
    #   #raw_input("PAuse")
    #   base_counts = counts(data)
    #   #print "Base data"
    #   #print data
    #   #print "Root node:",rn,varnames[rn]
    #   #raw_input("Pause")

    #   assert(len(base_counts) != 0), "Base counts is 0"
    #   if len(base_counts) == 1:
    #       if base_counts[0][0] >= 0.5*base_counts[0][1]:
    #           return node.Leaf(varnames,1)
    #       else: # More negative examples
    #           return node.Leaf(varnames,0)

    #   # Find the new data sets for left and right
    #   data_right = get_data_prime(data,rn,1)
    #   #print "Right data"
    #   #print data_right
    #   #raw_input("Pause")
    #   data_left  = get_data_prime(data,rn,0)
    #   #print "Left data"
    #   #print data_left
    #   #raw_input("Pause")
    #   # Get the counts
    #   #counts_right = counts(data_right,rn)
    #   #counts_left  = counts(data_left,rn)
    #   new_varnames = varnames[:rn] + varnames[(rn+1):]
    #   # Check returns
    #   # only split right: left terminates
    #   #if len(data_left) <= len(varnames):
    #   if type(data_left[0]) == np.int64 and type(data_right[0]) == np.int64:
    #       return node.Leaf(new_varnames,1)
    #   if type(data_left[0]) == np.float64 and type(data_right[0]) == np.float64:
    #       return node.Leaf(new_varnames,1)
    #   if type(data_left[0]) == np.float64 or type(data_left[0]) == np.int64:
    #       #print data_left
    #       counts_right = counts(data_right,rn)
    #       #if counts_left[0][0] >= 0.5*counts_left[0][1]:
    #       #if counts_left[:-1] == 1:
    #       if data_left[-1] == 1:
    #           l = 1
    #       else:
    #           l = 0
    #       return node.Split(data, \
    #                         rn,\
    #                         node.Leaf(new_varnames,l),\
    #                         build_tree(data_right,new_varnames))

    #       # only split left: right terminates
    #   #elif len(data_right) == len(data[0]) or len(counts_right) <= len(varnames):
    #   #elif len(data_right) <= len(varnames):
    #   elif type(data_right[0]) == np.float64 or type(data_right[0]) == np.int64:
    #       #counts_right = counts(data_right,rn)
    #       #if counts_right[0][0] >= 0.5*counts_right[0][1]:
    #       counts_left = counts(data_left,rn)
    #       #print data_right
    #       #if counts_right[:-1] == 1:
    #       if data_right[-1] == 1:
    #           l = 1
    #       else:
    #           l = 0
    #       return node.Split(data, \
    #                         rn,\
    #                         build_tree(data_left,new_varnames), \
    #                         node.Leaf(new_varnames,l))
    #   # Continue both ways
    #   counts_right = counts(data_right,rn)
    #   counts_left  = counts(data_left,rn)
    #   return node.Split(data, \
    #                     rn,\
    #                     build_tree(data_left,new_varnames),\
    #                     build_tree(data_right,new_varnames))



    #   #print rn
    #   #raw_input("Pause")
    #   #root_name = varnames[rn]
    #   #next_node(base_counts, data, rn)
    #   #left = determine_next_node(data,rn)
    #   #right= determine_next_node(data,rn)
    #   #return node.Leaf(varnames, 1)
    #   #return node.Leaf(varnames, root_node)


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
    #used = np.zeros(len(varnames)-1)

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)
    #print("Current tree:")
    #root.write(sys.stdout,0)

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
