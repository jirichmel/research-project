#!/usr/bin/env python
# coding: utf-8

# Version info:
# 6.5.2020 -- deprecation custom function in facor of pymatgen package.
# 1.7.2020 -- cumulative distance function added.

import numpy as np
import pandas as pd
import networkx as nx
import itertools as it
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

def get_hypar(df_frac, df_latt, df_gen, ajdi, rsn):
    """Optimize hyperparametr of ngram based on spacegroup and gamma angle.
    
    Keyword arguments:
    df_frac -- fractional coordinates dataframe
    df_latt -- lattice dataframe
    df_gen -- general data dataframe
    ajdi -- id of the material
    rsn -- relaxation step number of the material
    """
    spacegroup = df_gen["spacegroup"][df_gen["id"]==ajdi].values[0]
    material = init_material(df_frac, df_latt, ajdi, rsn)
    gamma = material.lattice.gamma
    if spacegroup == 12:
        return 1.4
    elif spacegroup == 33:
        return 1.4
    elif spacegroup == 167:
        return 1.5
    elif spacegroup == 194:
        return 1.3
    elif spacegroup == 206:
        return 1.5
    elif spacegroup == 227:
        if gamma < 60:
            return 1.4
        else:
            return 1.5

def gen_graph(mdm, material, graph_type, hypar=1.5):
    """Generates the crystal graph.
    
    Keyword arguments:
    mdm -- minimal distance matrix
    material -- instance of IStructure
    graph_type -- options: metal_oxygen_cons, all_cons, metal_metal_cons
    hypar -- hyperparameter which tunes the decision distances, default value is 1.5
    """
    number_of_atoms = len(mdm)
    
    species_vector = [str(atom) for atom in material.species]    

    # Shannon radii for coordination IV of O and coordination VI of Al, Ga, In
    # old choices of radii:
    
    #R_O = core.Element.O.data["Shannon radii"]["-2"]["II"][""]["ionic_radius"]
    #R_Al = core.Element.Al.data["Shannon radii"]["3"]["VI"][""]["ionic_radius"]
    #R_Ga = core.Element.Ga.data["Shannon radii"]["3"]["VI"][""]["ionic_radius"]
    #R_In = core.Element.In.data["Shannon radii"]["3"]["VI"][""]["ionic_radius"]
    
    # new choices: aritmetic averages (almost the same as geometric averages)
    
    sum_radius= \
    core.Element.O.data["Shannon radii"]["-2"]["II"][""]["ionic_radius"] + \
    core.Element.O.data["Shannon radii"]["-2"]["III"][""]["ionic_radius"] + \
    core.Element.O.data["Shannon radii"]["-2"]["IV"][""]["ionic_radius"] + \
    core.Element.O.data["Shannon radii"]["-2"]["VI"][""]["ionic_radius"] + \
    core.Element.O.data["Shannon radii"]["-2"]["VIII"][""]["ionic_radius"]
    R_O = (sum_radius)*(1/5)
    
    sum_radius= \
    core.Element.Al.data["Shannon radii"]["3"]["IV"][""]["ionic_radius"] + \
    core.Element.Al.data["Shannon radii"]["3"]["V"][""]["ionic_radius"] + \
    core.Element.Al.data["Shannon radii"]["3"]["VI"][""]["ionic_radius"]
    R_Al = (sum_radius)*(1/3)
    
    
    sum_radius= \
    core.Element.Ga.data["Shannon radii"]["3"]["IV"][""]["ionic_radius"] + \
    core.Element.Ga.data["Shannon radii"]["3"]["V"][""]["ionic_radius"] + \
    core.Element.Ga.data["Shannon radii"]["3"]["VI"][""]["ionic_radius"]
    R_Ga = (sum_radius)*(1/3)
    
    sum_radius= \
    core.Element.In.data["Shannon radii"]["3"]["IV"][""]["ionic_radius"] + \
    core.Element.In.data["Shannon radii"]["3"]["VI"][""]["ionic_radius"] + \
    core.Element.In.data["Shannon radii"]["3"]["VIII"][""]["ionic_radius"]
    R_In = (sum_radius)*(1/3)
    
    radii = { "O" : R_O, "Al" : R_Al, "Ga" : R_Ga, "In" : R_In }

    G = nx.Graph() # Empty graph.
    
    metal_oxygen_cons = """
if (namei == "O" and namej != "O") or (namei != "O" and namej == "O"):
                
    nodei = "".join([namei," ",str(i)])
    nodej = "".join([namej," ",str(j)])
                
    # The decision:
    if mdm[i,j] < hypar * (radii[namei]+radii[namej]):
        G.add_edge(nodei, nodej, distance=mdm[i,j])
"""

    all_cons = """                
nodei = "".join([namei," ",str(i)])
nodej = "".join([namej," ",str(j)])
                
# The decision:
if mdm[i,j] < hypar * (radii[namei]+radii[namej]):
    G.add_edge(nodei, nodej, distance=mdm[i,j])
"""

    metal_metal_cons = """
if (namei != "O" and namej != "O"):
                
    nodei = "".join([namei," ",str(i)])
    nodej = "".join([namej," ",str(j)])
                
    # The decision:
    if mdm[i,j] < hypar * (radii[namei]+radii[namej]):
        G.add_edge(nodei, nodej, distance=mdm[i,j])
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

GA_COORDINATIONS = ['Ga-0', 'Ga-1', 'Ga-2', 'Ga-3', 'Ga-4', 'Ga-5', 'Ga-6', 'Ga-7', 'Ga-8', 'Ga-9'] # , 'Ga-10'
AL_COORDINATIONS = ['Al-0', 'Al-1', 'Al-2', 'Al-3', 'Al-4', 'Al-5', 'Al-6', 'Al-7', 'Al-8', 'Al-9'] # , 'Al-10'
IN_COORDINATIONS = ['In-0', 'In-1', 'In-2', 'In-3', 'In-4', 'In-5', 'In-6', 'In-7', 'In-8', 'In-9'] # , 'In-10'
O_COORDINATIONS = ['O-0', 'O-1', 'O-2', 'O-3', 'O-4', 'O-5', 'O-6', 'O-7', 'O-8', 'O-9'] # , 'O-10'

ALL_COORDINATIONS_UNIGRAM = AL_COORDINATIONS + GA_COORDINATIONS + IN_COORDINATIONS + O_COORDINATIONS
ALL_COORDINATIONS_BIGRAM = []
ALL_COORDINATIONS_TRIGRAM = []
ALL_COORDINATIONS_QUADGRAM = []

# general function. run only once
def gen_all_coordinations(targeted_coordination_list, parts):
    """Generate the list of all possible 
    coordination combinations for ngram of 
    given length in a easy to read format.
    Keyword arguments:
    targeted_coordination_list -- the output list
    parts -- list of all combinations "of atom-coordination"
    """
    for thing in parts:
        srted = sorted(thing)
        desc = ""
        for i in srted:
            desc += i + "/"
        desc = desc[:-1]
        targeted_coordination_list.append(desc)
        
        
# here we gooo

def gen_desc(targeted_desc_list, G, n):
    """Generate ngram descriptors.
    Keyword arguments:
    targeted_desc_list -- the output list
    G -- the graph of the material
    n -- the length of the subgraph
    """
    coordinations = get_coordinations(G)
    for i in list(it.combinations(list(G.nodes()),n)):
        H = G.subgraph(i)
        subgraph_nodes = list(H.nodes())
        subgraph_edges = list(H.edges())
        if not len(subgraph_edges) < len(subgraph_nodes)-1:
            parts = []
            for node in subgraph_nodes:
                parts.append(node.split()[0] + "-" + str(coordinations[node]))
            parts = sorted(parts)
            desc = ""
            for i in parts:
                desc += i + "/"
            desc = desc[:-1]
            targeted_desc_list.append(desc)

def gen_matrix_row(coordinations, ALL_COORDINATIONS_NGRAM):
    """Generates the matrix row for the given material
    Keyword arguments:
    coordinations -- the material coordination information
    ALL_COORDINATIONS_NGRAM -- the list of all possible ngram features
    """
    hist = {}
    for i in ALL_COORDINATIONS_NGRAM:
        hist.update({i:0})
    for coord in coordinations:
        hist.update({coord:hist[coord]+1})
    return hist            

def gen_ngram_matrix(flag, df_frac, df_latt, df_gen, n, ALL_COORDINATIONS_NGRAM):
    """Generates the ngram matrix.
    
    Keyword arguments:
    flag -- type of dataset used. options: "relaxation", "final", "vegard"
    df_frac -- dataframe of fractional coordinates
    df_latt -- dataframe of lattice vectors
    df_gen -- dataframe of general data
    n -- integer, order of the n-gram
    ALL_COORDINATIONS_UNIGRAM -- list of coordinations ranging from 0 to 10
    """
    if flag=="final":
        ajdis = df_gen["id"]
        try:
            indexing = df_latt[["id", "relaxation_step_number"]]
        except KeyError:
            df_latt.insert(1, "relaxation_step_number", [-1 for i in range(len(df_latt))])
            df_frac.insert(1, "relaxation_step_number", [-1 for i in range(len(df_frac))])
            indexing = df_latt[["id", "relaxation_step_number"]]
    if flag=="relaxation" or flag=="vegard":
        ajdis = df_gen["id"]
        indexing = df_latt[["id", "relaxation_step_number"]]
        
    X = []
    for ajdi in ajdis:
        for rsn in indexing[indexing["id"]==ajdi].relaxation_step_number.values:

            material = init_material(df_frac, df_latt, ajdi, rsn)
            mdm = material.distance_matrix

            G = gen_graph(mdm, material, "metal_oxygen_cons", hypar=get_hypar(df_frac, df_latt, df_gen, ajdi, rsn))
            mat_n = []
            gen_desc(mat_n, G, n)
            row = np.array(list(gen_matrix_row(mat_n, ALL_COORDINATIONS_NGRAM).values()))/material.lattice.volume
            X.append(row)
        print(ajdi, "ended.")
    
    return np.array(X)

# generating ALL_COORDINATIONS_BI TRI QUAD GRAM lists
gen_all_coordinations(ALL_COORDINATIONS_BIGRAM, it.combinations_with_replacement(ALL_COORDINATIONS_UNIGRAM, 2))

gen_all_coordinations(ALL_COORDINATIONS_TRIGRAM, it.combinations_with_replacement(ALL_COORDINATIONS_UNIGRAM, 3))

gen_all_coordinations(ALL_COORDINATIONS_QUADGRAM, it.combinations_with_replacement(ALL_COORDINATIONS_UNIGRAM, 4))

def cumulative_distance_functions(df_frac, df_latt, df_gen, ajdi, rsn, ALL_COORDINATIONS_NGRAM):
    """Cumulative distance functions descriptors. CURRENTLY WORKS ONLY FOR UNIGRAMS!!!
    Keyword arguments:
    df_frac -- fractional coordinates dataframe
    df_latt -- lattice dataframe
    df_gen -- general data dataframe
    ajdi -- id of the material
    rsn -- relaxation step number of the material
    ALL_COORDINATIONS_NGRAM -- all coordinations of the given ngram
    """
    value = {
        "1": 0, 
        "r^1": 1, 
        "r^2": 2, 
        "r^3": 3,
        "r^4": 4,
        "r^5": 5,
        "r^6": 6,
        "r^7": 7,
        "r^8": 8,
        "r^9": 9,
        "r^10": 10,
        "r^11": 11,
        "r^12": 12,
        "r^-1": 13, 
        "r^-2": 14, 
        "r^-3": 15, 
        "r^-4": 16, 
        "r^-5": 17,
        "r^-6": 18,
        "r^-7": 19,
        "r^-8": 20,
        "r^-9": 21,
        "r^-10": 22,
        "r^-11": 23,
        "r^-12": 24,
    }

    value = {
        "1": 0,
        "r^1": 1,
    }

    counter = dict(zip(ALL_COORDINATIONS_NGRAM, [i for i in range(len(ALL_COORDINATIONS_NGRAM))]))

    # rows are the descriptors in value dict for givin coordination
    matrix = np.zeros((len(ALL_COORDINATIONS_NGRAM), len(value)), dtype="float64")

    material = init_material(df_frac, df_latt, ajdi, rsn)
    G = gen_graph(material.distance_matrix, material, "metal_oxygen_cons", hypar=get_hypar(df_frac, df_latt, df_gen, ajdi, rsn))
    neighbors = get_neighbors(G)

    for atom in neighbors:
        unigram = atom.split()[0] + "-" + str(len(list(neighbors[atom])))
        for nei in neighbors[atom]:
            for i in range(1, 2):
                matrix[counter[unigram], value["r^" + str(i)]] += G.get_edge_data(atom, nei)["distance"]**i
                #matrix[counter[unigram], value["r^-" + str(i)]] += G.get_edge_data(atom, nei)["distance"]**(-i)
            matrix[counter[unigram], value["1"]] += 1
            
    return matrix.reshape(1,-1)

def cumulative_distance_functions_matrix(df_frac, df_latt, df_gen, ALL_COORDINATIONS_NGRAM):
    """Cumulative distance functions descriptors. CURRENTLY WORKS ONLY FOR UNIGRAMS!!!
    Keyword arguments:
    df_frac -- fractional coordinates dataframe
    df_latt -- lattice dataframe
    df_gen -- general data dataframe
    ALL_COORDINATIONS_NGRAM -- all coordinations of the given ngram
    """
    ls = []
    try:
        indexing = df_latt[["id", "relaxation_step_number"]]
    except KeyError:
        df_latt.insert(1, "relaxation_step_number", [-1 for i in range(len(df_latt))])
        df_frac.insert(1, "relaxation_step_number", [-1 for i in range(len(df_frac))])
        indexing = df_latt[["id", "relaxation_step_number"]]
    for ajdi in indexing["id"].values:
        for rsn in indexing[indexing["id"]==ajdi]["relaxation_step_number"].values:
            ls.append(cumulative_distance_functions(df_frac, df_latt, df_gen, ajdi, rsn, ALL_COORDINATIONS_NGRAM))
    return np.concatenate(ls, axis=0)
