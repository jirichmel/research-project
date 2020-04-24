#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import networkx as nx
import json
import ray
import time
import ngram

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
    diag_val = int(mat[0,0])
    vec.append(diag_val)
    for i in range(dim[0]):
        for j in range(i):
            vec.append(int(mat[i,j]))
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

@ray.remote(num_return_vals=1)
def calculate_chunk(df_frac, df_latt, R_ions, hypar, chunk):
    """Calculate the data for the given chunk.
    
    Keyword arguments:
    df_frac -- dataframe of the fractional atomic coordinates
    df_latt -- dataframe of the lattice vectors
    R_ions -- dictionary of atomic radii
    hypar -- hyperparameter which contributes to the amount of neighboring atoms
    """
    ajdi_dict = {}
    for ajdi in chunk:
        rsn_dict = {}
        # Get the all the relaxation_step_number values in an array of the given id and iterate thru them.
        for rsn in df_latt.loc[df_latt["id"] == ajdi].loc[:,["relaxation_step_number"]].to_numpy().reshape(1,-1)[0]:
            mdm, sv = ngram.minimal_distance(df_frac,df_latt,ajdi,rsn)
            plm = ngram.get_hood(ngram.gengraph(mdm, sv, R_ions, hypar))
            plm = matrix_coding(plm)
            rsn_dict.update({str(rsn): plm})
        ajdi_dict[str(ajdi)] = rsn_dict 
    print("Chunk done.")
    return ajdi_dict


def save_as_json(name,dic,indent):
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
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip().startswith("#"):
            try:
                contents.append(eval(line))
            except SyntaxError:
                contents.append(line)
            except NameError:
                contents.append(line)
    return contents

start = time.time()

R_O, R_Al, R_Ga, R_In, hypar, NUM_CPUS, lattice_vector_relaxation_path, atoms_frac_xyz_relaxation_path, general_path, json_file_name = parse_input()

lattice_vector_relaxation = pd.read_csv(lattice_vector_relaxation_path)
atoms_frac_xyz_relaxation = pd.read_csv(atoms_frac_xyz_relaxation_path)
general = pd.read_csv(general_path)

R_ions = { "O" : R_O, "Al" : R_Al, "Ga" : R_Ga, "In" : R_In }

ray.init(num_cpus=NUM_CPUS)

chunks = np.array_split(general["id"].to_numpy(),NUM_CPUS)

object_ids = []
# Parallel forcycle:
for core in range(NUM_CPUS): # get the object ids into a list
    object_ids.append(calculate_chunk.remote(atoms_frac_xyz_relaxation, lattice_vector_relaxation, R_ions, hypar, chunks[core])) # remote function goes here

actual_parts = []

for core in range(NUM_CPUS): # get the actual parts from the object ids
    actual_parts.append(ray.get(object_ids[core]))

data = {}
for part in actual_parts:
    data.update(part)

save_as_json(json_file_name,data,4)

end = time.time()
print("Elapsed time:", end - start,"s")

ray.shutdown()
