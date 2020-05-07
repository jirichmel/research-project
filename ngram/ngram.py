#!/usr/bin/env python
# coding: utf-8

# Version info:
# 6.5.2020 -- deprecation custom function in facor of pymatgen package.

import numpy as np
import pandas as pd
import networkx as nx

from pymatgen import core as core

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
#Therefore it is necessary to test a beforehand unknown amount of
#neighboring cells because sometimes, there are problems with
#(e.g. spacegroup 227) the choice of the nearest instance of an atom.

def init_material(df_frac, df_latt, ajdi, rsn):
    """Initiate the IStructure class of pymatgen using the data.
    Returns an instance of IStructure for the given material data.

    Keyword arguments:
    df_frac -- dataframe of fractional coordinates of atoms
    df_latt -- dataframe of lattice vectors
    ajdi -- id of the material
    rsn -- relaxation step number of the material
    """
    lattice = np.transpose(lattmat(df_latt,ajdi,rsn)) # pymatgen compatible lattice vector
    species = df_frac[(df_frac["id"]==ajdi) & (df_frac["relaxation_step_number"]==rsn)].species.values
    coords = df_frac[(df_frac["id"]==ajdi) & (df_frac["relaxation_step_number"]==rsn)][df_frac.columns[-3:]].values

    material = core.IStructure(lattice=lattice, species=species, coords=coords, to_unit_cell=True, coords_are_cartesian=False)
    return material


def gen_graph(material, graph_type, hypar=1.5):
    """Generates the crystal graph.
    
    Keyword arguments:
    material -- instance of IStructure
    graph_type -- options: metal_oxygen_cons, all_cons, metal_metal_cons
    hypar -- hyperparameter which tunes the decision distances, default value is 1.5
    """
    number_of_atoms = len(material.distance_matrix)
    
    species_vector = [str(atom) for atom in material.species]    

    # Shannon radii for coordination IV of O and coordination VI of Al, Ga, In

    R_O = core.Element.O.data["Shannon radii"]["-2"]["II"][""]["ionic_radius"]
    R_Al = core.Element.Al.data["Shannon radii"]["3"]["VI"][""]["ionic_radius"]
    R_Ga = core.Element.Ga.data["Shannon radii"]["3"]["VI"][""]["ionic_radius"]
    R_In = core.Element.In.data["Shannon radii"]["3"]["VI"][""]["ionic_radius"]

    radii = { "O" : R_O, "Al" : R_Al, "Ga" : R_Ga, "In" : R_In }

    G = nx.Graph() # Empty graph.
    
    metal_oxygen_cons = """
if (namei == "O" and namej != "O") or (namei != "O" and namej == "O"):
                
    nodei = "".join([namei," ",str(i)])
    nodej = "".join([namej," ",str(j)])
                
    # The decision:
    if material.distance_matrix[i,j] < hypar * (radii[namei]+radii[namej]):
        G.add_edge(nodei, nodej, distance=material.distance_matrix[i,j])
"""

    all_cons = """                
nodei = "".join([namei," ",str(i)])
nodej = "".join([namej," ",str(j)])
                
# The decision:
if material.distance_matrix[i,j] < hypar * (radii[namei]+radii[namej]):
    G.add_edge(nodei, nodej, distance=material.distance_matrix[i,j])
"""

    metal_metal_cons = """
if (namei != "O" and namej != "O"):
                
    nodei = "".join([namei," ",str(i)])
    nodej = "".join([namej," ",str(j)])
                
    # The decision:
    if material.distance_matrix[i,j] < hypar * (radii[namei]+radii[namej]):
        G.add_edge(nodei, nodej, distance=material.distance_matrix[i,j])
"""

    options = {
            "metal_oxygen_cons": metal_oxygen_cons,
            "all_cons": all_cons,
            "metal_metal_cons": metal_metal_cons,
            }


    for i in range(number_of_atoms):
        namei = species_vector[i]
        for j in range(i):
            namej = species_vector[j]
            
            exec(options[graph_type])
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

def get_coordinations(G):
    """Get coordination numbers of every atom in the graph.

    Keyword arguments:
    G -- the crystal graph.
    """
    atoms = G.number_of_nodes()
    node_labels = sorted(dict(G.nodes.items()).keys(), key=lambda x: int(x.split()[1]))
    dic = {}
    for i in range(atoms):
        dic[node_labels[i]] = len(list(G[node_labels[i]]))
    return dic
