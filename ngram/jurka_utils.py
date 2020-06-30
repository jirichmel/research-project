#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import sklearn
import json
import ast

def matrix_coding(mat):
    """Optimizing the need storage space for lower triangular matrix with a uniform diagonal.
    Returns a vector containing concatenated sequences of rows
    starting from the second row and without the diagonal.
    
    Keyword arguments:
    mat -- input matrix
    """
    dim = mat.shape
    assert dim[0] == dim[1], "Error: The matrix is not a square matrix."
    vec = []
    diag_val = float(mat[0,0])
    vec.append(diag_val)
    for i in range(dim[0]):
        for j in range(i):
            vec.append(float(mat[i,j]))
    return vec

def matrix_decoding(vec):
    """Making a lower triangular matrix with the same value on the whole diagonal.
    Returns a numpy array.
    Keyword arguments:
    vec -- python list of numbers (output of matrix_coding)
    """
    dim = int(.5*(1 + np.sqrt(8*len(vec)-7))) # square matrix dimensions
    diag_val = vec[0]
    mat = np.diag(np.zeros(dim))
    n = 0
    for i in range(dim):
        for j in range(i):
            n += 1
            mat[i,j] = vec[n]
    return mat
    
def save_as_json(name, dic, indent):
    """Save the python dictionary as json file.
    
    Keyword arguments:
    name -- name of the file
    dic -- the dictionary containing the data
    indent -- the desired indent of the data in the file
    """
    out_file = open(name, "w")  
    json.dump(dic, out_file, indent=indent)  
    out_file.close()  

def parse_input():
    """Parse input from a text file in pipeline.
    """
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip().startswith("#"):
            try:
                contents.append(ast.literal_eval(line))
            except ValueError:
                raise SystemExit("Input not valid. Check the input file.")
    return contents

def execution_time(func):
    """Decorator the time the execution time of a function.
    
    Keyword arguments:
    func -- function to be decorated.
    """
    def wrap(*args, **kwargs):
        start = time.time()
        executed = func(*args, **kwargs)
        end = time.time()
        print("Execution time of " + str(func) + ":", end-start, "s.")
        return executed
    return wrap
    
def rmse(true, pred):
    """Calculate root mean squared error from true and pred."""
    return np.sqrt(sklearn.metrics.mean_squared_error(true, pred))

def rmsle(true, pred):
    """Calculate root mean squared logarithmic error from true and pred."""
    #return np.sqrt(np.mean(np.square(np.log1p(pred) - np.log1p(true))))
    return np.sqrt(sklearn.metrics.mean_squared_log_error(true, pred))
def mae(true, pred):
    """Calculate root mean squared logarithmic error from true and pred."""
    return sklearn.metrics.mean_absolute_error(true, pred)
