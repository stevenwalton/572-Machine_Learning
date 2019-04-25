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

def log_2(x):
    return math.log(x)/math.log(2)

def entropy(p):
    # no diff
    #e = 0
    #if p > 0 and p < 1:
    #    e = (-p * log_2(p)) - ((1 - p) * log_2(1-p))
    #return e
    if p == 0 or p == 1:
        return 0
    assert(type(p) is float),"Our fraction isn't a float!!!!!"
    return ((-p*math.log(p,2)) - ((1.-p)*(math.log((1.-p),2))))


def infogain(py_pxi, pxi, py, total):
    # no diff
    b = float(float(py) / float(total))
    total_ent = entropy(b)
    ent_left = 0

    if pxi != 0:
        c = float(float(py_pxi) / float(pxi))
        ent_left = entropy(c)

    ent_right = 0

    if total - pxi !=0:
        d = float((float(py) - float(py_pxi)) / float(float(total) - float(pxi)))
        ent_right = entropy(d)

    information_gain = total_ent

    if total != 0:
        l_frac = float(float(pxi)/float(total))
        r_frac = float(((float(total) - float(pxi)) / float(total)))
        l_side = l_frac * ent_left
        r_side = r_frac * ent_right

        information_gain -= (l_side + r_side)

#    print information_gain
    return information_gain
    #   S = entropy(py/float(total))
    #   try:
    #       left = ((total-pxi)/float(total)) * entropy(float(py-py_pxi)/float(total - pxi))
    #   except ZeroDivisionError:
    #       left = 0
    #   try:
    #       right = (pxi/float(total))*entropy(py_pxi/float(pxi))
    #   except ZeroDivisionError:
    #       right = 0
    #   return S - (left + right)

# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable
def find_stuff(data, att):
    # return list [(# of data lines), (# of data lines class == 1), (# lines where att == 0), (# lines att == 0 && class == 1)

    return_list = []
    return_list.append(len(data))

    pos_lines = 0
    left_att = 0
    left_att_pos = 0

    for i in range(len(data)):
        if data[i][-1] == 1:
            pos_lines += 1

        if data[i][att] == 0:
            left_att += 1
            if data[i][-1] == 1:
                left_att_pos += 1

    return_list.append(pos_lines)
    return_list.append(left_att)
    return_list.append(left_att_pos)

    return return_list
def highest_info_gain(data,used):
    max_gain = -1.0
    max_var = -1

    for i in range(len(data[0])-1):

        if i not in used:
            #print "checking " + str(i)
            stuff = find_stuff(data, i)

            #print "revised values - py_pxi = " + str(stuff[3]) + " pxi = " + str(stuff[2]) + " py = " + str(stuff[1]) + " total = " + str(stuff[0])
            this_gain = infogain(stuff[3], stuff[2], float(stuff[1]), float(stuff[0]))

            if this_gain > max_gain and (i not in used):
                max_gain = this_gain
                max_var = i

            #print "info gain claculated: " + str(this_gain)
    #print "max gain: " + str(max_gain)
    #print max_var
    return max_var
    #highest = -1
    #var = -1
    #total = len(data)
    #py = len([1 for row in data if (row[-1] == 1)])
    #for col in range(len(data[0])-1):
    ##for col in range(len(data[0])):
    #    if col not in used:
    #        pxi = len([1 for row in data if (row[col] == 1)])
    #        py_pxi = len([1 for row in data if ((row[col] == 1) and (row[-1] == 1))])
    #        ig = infogain(py_pxi, pxi, py, total)
    #        if ig > highest:
    #            highest = ig
    #            var = col
    ##print "split on",var
    #return var

def partition_data(data,split_on=None):
    r"""
    returns the left and right partition
    binary split
    """
    negative_datas = []
    positive_data = []

    for i in range(len(data)):
        if data[i][split_on] == 1:
            positive_data.append(data[i])

        elif data[i][split_on] == 0:
            negative_datas.append(data[i])

    return [negative_datas, positive_data]
    #left = []
    #right = []
    #for row in range(len(data)):
    #    if data[row][split_on] == 1:
    #        right.append(data[row])
    #    elif data[row][split_on] == 0:
    #        left.append(data[row])
    #    else:
    #        print "Can't be here"
    #        exit(1)
    #return left,right

def find_bigger(data):
    num_pos = 0
    num_neg = 0
    ret = 0
    for i in range(len(data)):
        if data[i][-1] == 1:
            num_pos += 1
        else:
            num_neg += 1

    if num_pos > num_neg:
        ret = 1

    return ret
    #pos = 0
    #neg = 0
    #for i in range(len(data)):
    #    if data[i][-1] == 1:
    #        pos += 1
    #    else:
    #        neg += 1
    #if pos > neg:
    #    return 1
    #else:
    #    return 0

def check_if_pure(data):
    zero_sum = 0
    one_sum = 0

    for i in range(len(data)):
        if data[i][-1] == 0:
            zero_sum += 1
        else:
            one_sum += 1

    if zero_sum == len(data):
        return [True, 0]
    elif one_sum == len(data):
        return  [True, 1]
    else:
        return [False, -1]
    #   z = 0
    #   o = 0
    #   for i in range(len(data)):
    #       if data[i][-1] == 0:
    #           z += 1
    #       else:
    #           o += 1
    #   if z == len(data):
    #       return True,0
    #   elif o == len(data):
    #       return True,1
    #   else:
    #       return False,-1


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
        l_check = check_if_pure(left)
        r_check = check_if_pure(right)
        if l_check[0] == True and r_check[0] == True:
            #print "two pures for " + varnames[split_on]
            return node.Split(varnames, \
                              split_on, \
                              node.Leaf(varnames, \
                                        l_check[1]), \
                              node.Leaf(varnames, \
                                        r_check[1]))
        elif l_check[0] == True:
            #print "left pure for " + varnames[split_on]
            return node.Split(varnames, \
                              split_on, \
                              node.Leaf(varnames, \
                                        l_check[1]), \
                              build_tree(right, \
                                         varnames,  \
                                         copy.deepcopy(used_attributes)))
        elif r_check[0] == True:
            #print "right pure for " + varnames[split_on]
            return node.Split(varnames, \
                              split_on, \
                              build_tree(left, \
                                         varnames, \
                                         copy.deepcopy(used_attributes)), \
                              node.Leaf(varnames, \
                                        r_check[1]))

        else:
            #print "split on" + varnames[split_on]

            ln = build_tree(left, \
                            varnames, \
                            copy.deepcopy(used_attributes))

            rd = build_tree(right, \
                            varnames,\
                            copy.deepcopy(used_attributes))

            return node.Split(varnames, split_on, ln, rd)
    #right, left= partition_data(data,split_on)

    #pure_left = all((left[i][-1] == 1) for i in range(len(left))) or \
    #            all((left[i][-1] == 0) for i in range(len(left)))
    #pure_right = all((right[i][-1] == 1) for i in range(len(right))) or \
    #            all((right[i][-1] == 0) for i in range(len(right)))
    #   pure_left,l_val = check_if_pure(left)
    #   pure_right,r_val= check_if_pure(right)
    #   if pure_left and pure_right:
    #       #print "two pures for " + varnames[split_on]
    #       return node.Split(varnames, \
    #                         split_on, \
    #                         node.Leaf(varnames, \
    #                                   l_val), \
    #                         node.Leaf(varnames, \
    #                                   r_val))
    #   elif pure_left:
    #       #print "left pure for " + varnames[split_on]
    #       return node.Split(varnames, \
    #                         split_on, \
    #                         node.Leaf(varnames, \
    #                                   l_val), \
    #                         build_tree(right, \
    #                                    varnames,  \
    #                                    copy.deepcopy(used_attributes)))
    #   elif pure_right:
    #       #print "right pure for " + varnames[split_on]
    #       return node.Split(varnames, \
    #                         split_on, \
    #                         build_tree(left, \
    #                                    varnames, \
    #                                    copy.deepcopy(used_attributes)), \
    #                         node.Leaf(varnames, \
    #                                   r_val))

    #   else:
    #       #print "split on" + varnames[rn]

    #       ln = build_tree(left, \
    #                       varnames, \
    #                       copy.deepcopy(used_attributes))

    #       rd = build_tree(right, \
    #                       varnames,\
    #                       copy.deepcopy(used_attributes))

    #       return node.Split(varnames, split_on, ln, rd)

    #if pure_left and pure_right:
    #    return node.Split(varnames,\
    #                      split_on,\
    #                      node.Leaf(varnames,l_val),\
    #                      node.Leaf(varnames,r_val))
    #elif pure_left:
    #    return node.Split(varnames,\
    #                      split_on,\
    #                      node.Leaf(varnames,l_val),\
    #                      build_tree(right,\
    #                                 varnames,\
    #                                 copy.deepcopy(used_attributes)))
    #elif pure_right:
    #    return node.Split(varnames,\
    #                      split_on,\
    #                      build_tree(left,\
    #                                 varnames,\
    #                                 copy.deepcopy(used_attributes)),
    #                      node.Leaf(varnames,r_val))
    #else:
    #    return node.Split(varnames,\
    #                      split_on,\
    #                      build_tree(left,\
    #                                 varnames,\
    #                                 copy.deepcopy(used_attributes)),
    #                      build_tree(right,\
    #                                 varnames,\
    #                                 copy.deepcopy(used_attributes)))


    #if pure_left or pure_right:
    #    if left == []:
    #        lef = 0
    #    else:
    #        lef = left[0][-1]
    #    if right == []:
    #        rig = 0
    #    else:
    #        rig = right[0][-1]
    #    if pure_left and pure_right:
    #        return node.Split(varnames, \
    #                          split_on, \
    #                          node.Leaf(varnames,lef), \
    #                          node.Leaf(varnames,rig))
    #    elif pure_left:
    #        return node.Split(varnames, \
    #                          split_on, \
    #                          node.Leaf(varnames,lef), \
    #                          build_tree(right, \
    #                                     varnames, \
    #                                     copy.deepcopy(used_attributes)))
    #    else:
    #        return node.Split(varnames, \
    #                          split_on, \
    #                          build_tree(left,\
    #                                     varnames, \
    #                                     copy.deepcopy(used_attributes)), \
    #                          node.Leaf(varnames, rig))

    #return node.Split(varnames,\
    #                  split_on,\
    #                  build_tree(left,\
    #                             varnames,\
    #                             copy.deepcopy(used_attributes)), \
    #                  build_tree(right,\
    #                             varnames, \
    #                             copy.deepcopy(used_attributes)))


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
    used.append(20)
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
