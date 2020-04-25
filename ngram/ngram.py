#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import networkx as nx

#
# Crystal graph representation which converts 
# crystalline structures into features by counting
# the contiguous sequences of unique atomic sites of
# various lengths.
#

def lattmat(df,ajdi,relaxation_step_number):
    """Create a numpy matrix of lattice vectors. A column = a vector.
    
    Keyword arguments:
    df -- dataframe of the lattice vectors
    ajdi -- id of the material
    relaxation_step_number -- self explanatory
    """
    return np.transpose(df.loc[(df["id"]==ajdi) & (df["relaxation_step_number"]==relaxation_step_number)].loc[:,["lattice_vector_1_x" ,"lattice_vector_1_y" ,"lattice_vector_1_z" ,"lattice_vector_2_x" ,"lattice_vector_2_y" ,"lattice_vector_2_z" ,"lattice_vector_3_x" ,"lattice_vector_3_y", "lattice_vector_3_z"]].to_numpy().reshape(3,3))

def posvecs(df,ajdi,relaxation_step_number):
    """Create a numpy array of vectors of position of the atoms of the material with the given id and relaxation_step_number.
    
    Keyword arguments:
    df -- dataframe of xyz coordinates of atoms
    ajdi -- id of the material
    relaxation_step_number -- self explanatory
    """
    return df.loc[(df["id"]==ajdi) & (df["relaxation_step_number"]==relaxation_step_number)].loc[:,["x [A]" ,"y [A]" ,"z [A]"]].to_numpy()

def xyzcoord(lattice_matrix,frac_vector):
    """Calculate the xyz coordinates (numpy vector) of a frac vector in the given lattice.
    
    Keyword arguments:
    lattice_matrix -- 3x3 matrix of the lattice vectors
    frac_vector -- frac positional vector of the atom
    """
    return np.dot(lattice_matrix,frac_vector)

#atoms_frac_xyz_relaxation includes reduced coordinates.
#So we have the whole dataframe of the data we need already.

#To calculate the actual distance of the atoms in the crystal structure
#we must account for the fact that we have data only of one unit cell. 
#But the distance of the atom i from the atom j can be the smallest
#if we compare it with the atom j in the neighboring cell.
#Therefore it is necessary to test all the neighboring cells as well.
#That means to do 9 comparisons in total (8 neighboring cells and the
#actual cell we are in).

def length(vector):
    """Calculate the euclidean distance of a vector.
    
    Keyword arguments:
    vector -- input numpy vector
    """
    return np.linalg.norm(vector,ord=2)

def minimal_distance(df_frac, df_lattice, ajdi, relaxation_step_number):
    """Get the shortest distances between atoms of the given material id and relaxation_step_number.
    Outputs a matrix of distances and an array of species.
    
    Keyword arguments:
    df_frac -- dataframe of fractional coordinates of atoms
    df_lattice -- dataframe of lattice vectors
    ajdi -- id of the material
    relaxation_step_number -- self explanatory
    """
    # get the slice of the atoms for the given id and relarelaxation_step_number
    atoms = df_frac.loc[(df_frac["id"]==ajdi) & (df_frac["relaxation_step_number"]==relaxation_step_number)]
    
    # Species vector:
    species_vector = atoms["species"].to_numpy()
    
    number_of_total_atoms = len(atoms.index.values)
    
    # A symmetrical matrix which holds the distances of i-th and j-th atoms, the diagonal is zeros.
    minimal_distance_matrix = np.zeros((number_of_total_atoms,number_of_total_atoms))
    
    #the lattice vector matrix:
    A = lattmat(df_lattice,ajdi,relaxation_step_number)
    
    for i in range(number_of_total_atoms):
        for j in range(i):
            
            # The difference of fractional vectors:
            ij = atoms.loc[atoms.index.values[i],["L1","L2","L3"]].to_numpy() - atoms.loc[atoms.index.values[j],["L1","L2","L3"]].to_numpy()
            
            # Initialize the dummy variables for this iteration:
            minimal_distance = np.inf # minimal distance between two atoms
            
                # Seek through the neighboring unit cells:
            for k in range(-1, 2):
                for l in range(-1, 2):
                    for m in range(-1, 2):

                        # difference of ij the neighboring cell shift:
                        r = ij + np.array([k,l,m])

                        # euclidean distance of the vectors which are recalculated back:
                        R = xyzcoord(A,r)
                        distance = length(R)

                        if (distance < minimal_distance):
                            minimal_distance = distance
                            
            # save the values for given i,j into the matrix:
            minimal_distance_matrix[i,j] = minimal_distance
            
    return minimal_distance_matrix, species_vector

def gengraph(minimal_distance_matrix, species_vector, ionic_radii, hypar=1.5):
    """Generates the crystal graph.
    
    Keyword arguments:
    minimal_distance_matrix -- matrix of minimal distances
    species_vector -- the names of the atoms in the unit cell
    ionic_radii -- dictionary of ionic radii for the atoms spicies
    hypar -- hyperparameter which tunes the decision distances, default value is 1.5
    """
    number_of_total_atoms = len(minimal_distance_matrix)
    
    G = nx.Graph() # Empty graph.
    
    for i in range(number_of_total_atoms):
        namei = species_vector[i]
        for j in range(i):
            namej = species_vector[j]
            
            # exclude oxygen connections:
            if (namei == "O" and namej != "O") or (namei != "O" and namej == "O"):
                
                nodei = "".join([namei," ",str(i)])
                nodej = "".join([namej," ",str(j)])
                
                # The decision:
                if minimal_distance_matrix[i,j] < hypar * (ionic_radii[namei]+ionic_radii[namej]):
                    G.add_edge(nodei,nodej)
    return G

def get_neighbors(G):
    """Get neigbors of every atom in the graph.
    
    Keyword arguments:
    G -- the crystal graph.
    """
    atoms = G.number_of_nodes()
    node_labels = sorted(dict(G.nodes.items()).keys(), key=lambda x: int(x.split()[1]))
    dic = {}
    for i in range(atoms):
        dic[node_labels[i]] = list(G[node_labels[i]]) # list(G.neighbors(node_labels[i]))
    return dic

def get_hood(G):
    """Calculates the least amount of inbetween atoms between two atoms.
    
    Keyword argumets:
    G -- the crystal graph.
    """
    atoms = G.number_of_nodes()
    path_length_matrix = np.zeros((atoms,atoms), dtype = int)
    node_labels = sorted(dict(G.nodes.items()).keys(), key = lambda x: int(x.split()[1]))
    for i in range(atoms):
        for j in range(i):
            try:
                path_length_matrix[i, j] = nx.shortest_path_length(G, node_labels[i], node_labels[j])
            except nx.NetworkXNoPath:
                print("There is no path between some atoms.")
                print("Atoms:")
                print(node_labels)
    return path_length_matrix
