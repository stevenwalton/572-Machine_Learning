#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100
epsilon = 0.0001

# Load data from a file
def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  namehash = {}
  for l in f:
    example = [int(x) for x in p.split(l.strip())]
    x = example[0:-1]
    y = example[-1]
    data.append( (x,y) )
  return (data, varnames)

def sigmoid(z):
    try:
        return 1./(1.+exp(-z))
    except OverflowError:
        return 0.

# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
   numvars = len(data[0][0])
   w = [0.0] * numvars
   b = 0.0
   for itr in range(MAX_ITERS):
      #print("itr:",itr)
      # Some loop for all (x,y) in D do
      g_w = [0.]*numvars
      g_b = 0.
      for i in range(len(data)):
          (x,y) = data[i]
          a = b
          for d in range(numvars):
              a += w[d]*x[d]
          grad = -sigmoid(-y*a)*y
          for j in range(numvars):
              g_w[j] += x[j] * grad
          g_b += grad
      mag_g_w = 0
      for d in range(numvars):
          mag_g_w += g_w[d] * g_w[d]
      mag_g_b = g_b * g_b
      if (sqrt(mag_g_w + mag_g_b)) < epsilon:
          return (w,b)
      for d in range(numvars):
          w[d] -= eta * (g_w[d] + l2_reg_weight * w[d])
      b -= eta * g_b
   return (w,b)

# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
  (w,b) = model
  a = b
  for i in range(len(x)):
      a += w[i]*x[i]
  return sigmoid(a)


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  if (len(argv) != 5):
    print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  eta = float(argv[2])
  lam = float(argv[3])
  modelfile = argv[4]

  # Train model
  (w,b) = train_lr(train, eta, lam)

  # Write model file
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in range(len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  for (x,y) in test:
    prob = predict_lr( (w,b), x )
    print(prob)
    if (prob - 0.5) * y > 0:
      correct += 1
  acc = float(correct)/len(test)
  print("Accuracy: ",acc)

if __name__ == "__main__":
  main(sys.argv[1:])
