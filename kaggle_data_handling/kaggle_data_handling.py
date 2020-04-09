#!/usr/bin/env python
# coding: utf-8

# # Kaggle Dataset

# In[1]:


import numpy as np
import pandas as pd
import os
import ray


# In[2]:


# Multiprocessing is required otherwise we're gonna have a bad time
num_cpus = 15
ray.init(num_cpus = 15)


# In[3]:


# The following function finds a specific line and get data in this single line
def match_line(raw_data_file, *args): # the path to the file and the sought strings
    lines = ""
    with open(raw_data_file, 'r') as file:
        for line in file: # iterate through the text file
            for arg in args: # for all strings passed to the function...
                if arg in line: # if the current line contains the sought string, then:
                    lines += line # Will add the matched text to the lines string
    return lines

# Gets all the numerical values from all the lines with the given match_string
# and outputs them in a list of three numerical values which can then be processed further
def get_line_data_with_species(input_string, keyword):
    vector_DATA = []
    l = []
    species = []
    for line in input_string:
        if keyword in line:
            l = []
            for t in line.split():
                try:
                    l.append(float(t))
                except ValueError:
                    pass
            vector_DATA.append(l)
            # get the element from the end of the string and use the same data format as vector_DATA:
            species.append(line[-2:].strip())
    return vector_DATA, species


# In[4]:


def directories(where,number_of_directories):
    print("Make sure empty",where,"folder exists in the given path!")
    os.mkdir("".join(["../", where,"/directory_tree"]))
    os.mkdir("".join(["../", where,"/final"]))
    os.mkdir("".join(["../", where,"/relaxation"]))
    for i in range(1,number_of_directories+1):
        os.mkdir("".join(["../", where,"/directory_tree/",str(i)]))
    print("In","".join(["../",where]),"created final, relaxation and directory_tree folders,",number_of_directories,"directories in",where,"/directory_tree folder were created.")


# In[6]:


# A Ray remote function
@ray.remote(num_return_vals=9)
def iterate_thru_all(where, list_of_filenumbers, *args): # number of files is the number of files from train_ or test_
    
    #Loading the OLS oxide energies:
    oxides = pd.read_csv("../OLS_oxide_energies").to_numpy()
    
    #############################
    
    # Need to load this to get the spacegroup column:
    df_work_test_or_train_DATA = pd.read_csv("".join(["../", where,".csv/", where,".csv"]))
    
    #############################
    
    ## General test or train data:
    df_general_DATA = pd.DataFrame(columns = ["id", "spacegroup", "number_of_total_atoms", 
                                                    "percent_atom_al", "percent_atom_ga", "percent_atom_in"]) 
    
    
    ## It is worth it to store the final values right now during the extraction:
    # Final lattice vectors:
    df_lattice_vector_DATA = pd.DataFrame(columns = ["id", 
                                                      "lattice_vector_1_x", "lattice_vector_1_y", "lattice_vector_1_z", 
                                                      "lattice_vector_2_x", "lattice_vector_2_y", "lattice_vector_2_z",
                                                      "lattice_vector_3_x", "lattice_vector_3_y", "lattice_vector_3_z"])
    # Final atomic coordinates:
    df_atoms_xyz_DATA = pd.DataFrame(columns = ["id", "species", "x [A]", "y [A]", "z [A]"])
    
    # Final fractional atomic coordinates:
    df_atoms_frac_xyz_DATA = pd.DataFrame(columns = ["id", "species", "L1", "L2", "L3"])
    
    # Final energy:
    df_energy_DATA = pd.DataFrame(columns = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"])
    
    
    
    ## Relaxation step data:
    # lattice_vector data of relaxation steps:
    df_lattice_vector_relaxation_DATA = pd.DataFrame(columns = ["id", "relaxation_step_number", 
                                                      "lattice_vector_1_x", "lattice_vector_1_y", "lattice_vector_1_z", 
                                                      "lattice_vector_2_x", "lattice_vector_2_y", "lattice_vector_2_z",
                                                      "lattice_vector_3_x", "lattice_vector_3_y", "lattice_vector_3_z"])
    # atomic coordinates during relaxation steps:
    df_atoms_xyz_relaxation_DATA = pd.DataFrame(columns = ["id", "relaxation_step_number", "species", "x [A]", "y [A]", "z [A]"])

    # Fractional atomic coordinates during relaxation steps:
    df_atoms_frac_xyz_relaxation_DATA = pd.DataFrame(columns = ["id", "relaxation_step_number", "species", "L1", "L2", "L3"])

    # Energies during relaxation steps:
    df_energy_relaxation_DATA = pd.DataFrame(columns = ["id", "relaxation_step_number", "formation_energy_ev_natom", "bandgap_energy_ev"])
    
    ########################
    
    k = 0 # a counter
    l = 0 # another counter
    m = 0 # another counter
    n = 0 # another counter
    
    # Global iterators:
    global_iterator_0 = 0
    global_iterator_1 = 0
    global_iterator_2 = 0
    
    
    #########################
    
    for i in list_of_filenumbers:

        print("Extracting from bandfile number ", str(i), ".", " Calculating...")
        raw_data_file = "".join(["../", where, "_just_bandfiles/", str(i), "/band.out"]) # specify the path to the file in an iterable form
        output = match_line(raw_data_file, *args)
        output = output.splitlines() # convert the string into a list using the line breakers
        
        
        # Get kation percetages right:
        LINE_DATA = get_line_data_with_species(output[output.index("  Final atomic structure:"):],"            atom")
        for j in LINE_DATA[1]:
            if j == "O":
                k = k + 1
            elif j == "Al":
                l = l + 1
            elif j == "Ga":
                m = m + 1
            elif j == "In":
                n = n + 1
        NATOMS = k + l + m + n
        # Kation percentages (notice the absense of oxygen in the following three lines)
        percent_atom_al = l/(l+m+n)
        percent_atom_ga = m/(l+m+n)
        percent_atom_in = n/(l+m+n)
        # General data is stored right here:
        df_general_DATA.loc[i-1] = [str(i), df_work_test_or_train_DATA.at[i-1, "spacegroup"], str(NATOMS), percent_atom_al, percent_atom_ga, percent_atom_in]
        # Clearing some variables for future use:
        k = 0
        l = 0
        m = 0
        n = 0
        
        ## Final output dataset:
        # Preparing the final output in case no relaxation data are present and to get the final values 
        FINAL = output[output.index("  Final atomic structure:")-1:]
        
        # folder for the final data values:
        os.mkdir("".join(["../", where,"/directory_tree/", str(i),"/",str(i)]))
        
        # Lattice vectors:
        LINE_DATA_lattice = get_line_data_with_species(FINAL[2:5], "  lattice_vector")[0]
        
        # Open the file to save the lattice datapoint into a its folder:
        f = open("".join(["../", where,"/directory_tree/", str(i),"/",str(i),"/lattice_vector.csv"]),'w')
        f.write("id,lattice_vector_1_x,lattice_vector_1_y,lattice_vector_1_z,lattice_vector_2_x,lattice_vector_2_y,lattice_vector_2_z,lattice_vector_3_x,lattice_vector_3_y,lattice_vector_3_z\n")
        f.write(",".join(map(str,[i] + LINE_DATA_lattice[0] + LINE_DATA_lattice[1] + LINE_DATA_lattice[2]))+"\n")
        f.close()
        
        df_lattice_vector_DATA.loc[i-1] = [str(i)] + LINE_DATA_lattice[0] + LINE_DATA_lattice[1] + LINE_DATA_lattice[2]

        # Energies:
        LINE_DATA_gap = get_line_data_with_species([FINAL[0]],"  ESTIMATED overall HOMO-LUMO gap:")[0][0][0]
        LINE_DATA_energy = np.around(get_line_data_with_species([FINAL[-1]],"  | Total energy of the DFT / Hartree-Fock s.c.f. calculation      :")[0][0][0],decimals=8)
        # Formation calculation:
        LINE_DATA_energy = -2.5*(percent_atom_al*oxides[0][0] + percent_atom_ga*oxides[0][1] + percent_atom_in*oxides[0][2] - LINE_DATA_energy/NATOMS)
        # Open the file to save the energy datapoint into a its folder:
        f = open("".join(["../", where,"/directory_tree/", str(i),"/",str(i),"/energy.csv"]),'w')
        f.write("id,formation_energy_ev_natom,bandgap_energy_ev\n")
        f.write(",".join(map(str,[i] + [LINE_DATA_energy, LINE_DATA_gap]))+"\n")
        f.close()
        
        df_energy_DATA.loc[i-1] = [str(i),LINE_DATA_energy,LINE_DATA_gap]

        # Atomic positions:
        LINE_DATA_atom = get_line_data_with_species(FINAL,"            atom") # returns a list of two lists
        LINE_DATA_frac = get_line_data_with_species(FINAL, "       atom_frac") # returns a list of two lists
        
        f = open("".join(["../", where,"/directory_tree/", str(i),"/",str(i),"/atoms_xyz.csv"]),'w')
        g = open("".join(["../", where,"/directory_tree/", str(i),"/",str(i),"/atoms_frac_xyz.csv"]),'w')
        f.write("id,species,x [A],y [A],z [A]\n")
        g.write("id,species,L1,L2,L3\n")
        
        for j in range(NATOMS):
            f.write(",".join(map(str,[i, LINE_DATA_atom[1][j]] + LINE_DATA_atom[0][j]))+"\n")
            
            df_atoms_xyz_DATA.loc[global_iterator_0] = [str(i),str(LINE_DATA_atom[1][j])] + LINE_DATA_atom[0][j]

            g.write(",".join(map(str,[i, LINE_DATA_frac[1][j]] + LINE_DATA_frac[0][j]))+"\n")
            
            df_atoms_frac_xyz_DATA.loc[global_iterator_0] = [str(i),str(LINE_DATA_frac[1][j])] + LINE_DATA_frac[0][j]
            
            global_iterator_0 += 1
            
        f.close()
        g.close()
        
        # removes also the line before "  Final atomic structure:" which contains HOMO-LUMO energy and its value is reported two lines above this which we are about to remove:
        del output[output.index("  Final atomic structure:")-1:] 
        
        while k < len(output) and "  Relaxation step number" not in output[k]: # remove the selfconsistency output data before the Relaxation steps (can remove the whole list)
            k = k + 1
        del output[:k]
        k = 0
        
        if output != []: # if there are any Relaxation data then...
            LINE_DATA_energy = []
            LINE_DATA_gap = []
            # lattice_vectors of Relaxation + energies of Relaxation:
            LINE_DATA = get_line_data_with_species(output, "  lattice_vector")[0]
            
            l = 1 # relaxation step number starts with 1
            
            #the first folder for the first relaxation data
            os.mkdir("".join(["../", where,"/directory_tree/", str(i),"/",str(i),".", str(l)]))
            
            l = 2
            for j in range(1,len(output)): # omit the first line which includes the first Relaxation step number string
                if "Relaxation step number" in output[j]: # right above relaxation step number are two numerical values we want
                    
                    LINE_DATA_energy.append(get_line_data_with_species([output[j-2]],"  | Total energy                  :")[0][0][1])
                    LINE_DATA_gap.append(get_line_data_with_species([output[j-1]],"  ESTIMATED overall HOMO-LUMO gap:")[0][0][0])
                    
                    os.mkdir("".join(["../", where,"/directory_tree/", str(i),"/",str(i),".", str(l)]))
                    l = l + 1
            
            # Add the last relaxation step values:
            LINE_DATA_energy.append(get_line_data_with_species([output[-1]],"  | Total energy                  :")[0][0][1])
            LINE_DATA_gap.append(get_line_data_with_species([output[-2]],"  ESTIMATED overall HOMO-LUMO gap:")[0][0][0])
            
            
            l = 1 # relaxation step number starts with 1
            while LINE_DATA != []: # using the list of lattice vectors until we empty it
                
                # Open the file to save the lattice datapoint into a its folder:
                f = open("".join(["../", where,"/directory_tree/", str(i),"/",str(i),".", str(l),"/lattice_vector.csv"]),'w')
                f.write("id,relaxation_step_number,lattice_vector_1_x,lattice_vector_1_y,lattice_vector_1_z,lattice_vector_2_x,lattice_vector_2_y,lattice_vector_2_z,lattice_vector_3_x,lattice_vector_3_y,lattice_vector_3_z\n")
                f.write(",".join(map(str,[i, l] + LINE_DATA[0] + LINE_DATA[1] + LINE_DATA[2]))+"\n")
                f.close()
                
                df_lattice_vector_relaxation_DATA.loc[global_iterator_1] = [str(i),str(l)] + LINE_DATA[0] + LINE_DATA[1] + LINE_DATA[2]
                
                #Calculating the formation energy:
                LINE_DATA_energy[0] = -2.5*(percent_atom_al*oxides[0][0] + percent_atom_ga*oxides[0][1] + percent_atom_in*oxides[0][2] - LINE_DATA_energy[0]/NATOMS)
                
                # Open the file to save the energy datapoint into a its folder:
                f = open("".join(["../", where,"/directory_tree/", str(i),"/",str(i),".", str(l),"/energy.csv"]),'w')
                f.write("id,relaxation_step_number,formation_energy_ev_natom,bandgap_energy_ev\n")
                f.write(",".join(map(str,[i, l] + [LINE_DATA_energy[0], LINE_DATA_gap[0]]))+"\n")
                f.close()
                
                
                df_energy_relaxation_DATA.loc[global_iterator_1] = [str(i),str(l),LINE_DATA_energy[0],LINE_DATA_gap[0]]
                
                
                del LINE_DATA[:3] # delete the three values used
                del LINE_DATA_energy[0]
                del LINE_DATA_gap[0]
                
                l = l + 1 # counter for relaxation step
                global_iterator_1 += 1
            l = 0
            
            # Atomic positions of Relaxation + Fractional coordinates of Relaxation:
            LINE_DATA = get_line_data_with_species(output, "            atom")
            LINE_DATA_frac = get_line_data_with_species(output, "       atom_frac") # returns a list of two lists
            k = 1
            while LINE_DATA != ([], []):
                
                f = open("".join(["../", where,"/directory_tree/", str(i),"/",str(i),".", str(k),"/atoms_xyz.csv"]),'w')
                g = open("".join(["../", where,"/directory_tree/", str(i),"/",str(i),".", str(k),"/atoms_frac_xyz.csv"]),'w')
                f.write("id,relaxation_step_number,species,x [A],y [A],z [A]\n")
                g.write("id,relaxation_step_number,species,L1,L2,L3\n")
                
                for j in range(NATOMS):
                    
                    f.write(",".join(map(str,[i, k, LINE_DATA[1][j]] + LINE_DATA[0][j]))+"\n")
                    df_atoms_xyz_relaxation_DATA.loc[global_iterator_2] = [str(i),str(k),str(LINE_DATA[1][j])] + LINE_DATA[0][j]
                    g.write(",".join(map(str,[i, k, LINE_DATA_frac[1][j]] + LINE_DATA_frac[0][j]))+"\n")
                    df_atoms_frac_xyz_relaxation_DATA.loc[global_iterator_2] = [str(i),str(k),str(LINE_DATA_frac[1][j])] + LINE_DATA_frac[0][j]
                    global_iterator_2 += 1
                
                f.close()
                g.close()
                
                del LINE_DATA_frac[0][:NATOMS]
                del LINE_DATA_frac[1][:NATOMS]
                
                del LINE_DATA[0][:NATOMS]
                del LINE_DATA[1][:NATOMS]
                
                k += 1
            k = 0
        else: # if there are no relaxation data, then the final atomic structure is the only data point of the given material.
            # This is easily handled since the structure of the Final is the same everywhere
            # The relevant data is already in variables.
            # Lattice vectors:
            l = 0 # no relaxation means we assign zero to relaxation step number
            
            df_lattice_vector_relaxation_DATA.loc[global_iterator_1] = [str(i),str(l)] + LINE_DATA_lattice[0] + LINE_DATA_lattice[1] + LINE_DATA_lattice[2]
            
            # Energies:
            df_energy_relaxation_DATA.loc[global_iterator_1] = [str(i),str(l),LINE_DATA_energy,LINE_DATA_gap]
            
            global_iterator_1 += 1
            
            # Atomic positions:
            for j in range(NATOMS):
                
                df_atoms_xyz_relaxation_DATA.loc[global_iterator_2] = [str(i),str(l),str(LINE_DATA_atom[1][j])] + LINE_DATA_atom[0][j]
                df_atoms_frac_xyz_relaxation_DATA.loc[global_iterator_2] = [str(i),str(l),str(LINE_DATA_frac[1][j])] + LINE_DATA_frac[0][j]
                
                global_iterator_2 += 1
            
    # return the created dataframes
    print("Done with files",min(list_of_filenumbers), "-", max(list_of_filenumbers), ".")
    return df_general_DATA, df_lattice_vector_relaxation_DATA, df_energy_relaxation_DATA, df_atoms_xyz_relaxation_DATA, df_atoms_frac_xyz_relaxation_DATA, df_lattice_vector_DATA, df_energy_DATA, df_atoms_xyz_DATA, df_atoms_frac_xyz_DATA


# In[6]:


# Create parent directories of the materials
directories("train",2400)


# In[10]:


# Create parent directories of the materials
directories("test",600)


# In[8]:


# train set
list_of_file_numbers = []
for k in range(num_cpus):
    list_of_file_numbers.append(list(range(1 +k*160,160+1 +k*160)))
    
## Cleaning the train set:

# Problematic datafiles:
print("Removing material id",list_of_file_numbers[2][list_of_file_numbers[2].index(464)])
del list_of_file_numbers[2][list_of_file_numbers[2].index(464)] # 464. is problematic and will not be included in the dataset
print("Removing material id",list_of_file_numbers[13][list_of_file_numbers[13].index(2189)])
del list_of_file_numbers[13][list_of_file_numbers[13].index(2189)] # 2189. is problematic and will not be included in the dataset

# Removing duplicates from the train set:
print("Removing material id",list_of_file_numbers[0][list_of_file_numbers[0].index(126)])
del list_of_file_numbers[0][list_of_file_numbers[0].index(126)] # id 395/126 is duplicate. One removed.
print("Removing material id",list_of_file_numbers[7][list_of_file_numbers[7].index(1215)])
del list_of_file_numbers[7][list_of_file_numbers[7].index(1215)] # id 1215/1886 is duplicate. One removed.
print("Removing material id",list_of_file_numbers[2][list_of_file_numbers[2].index(353)])
del list_of_file_numbers[2][list_of_file_numbers[2].index(353)] # id 2075/353 is duplicate. One removed.
print("Removing material id",list_of_file_numbers[1][list_of_file_numbers[1].index(308)])
del list_of_file_numbers[1][list_of_file_numbers[1].index(308)] # id 308/2154 is duplicate. One removed.
print("Removing material id",list_of_file_numbers[3][list_of_file_numbers[3].index(531)])
del list_of_file_numbers[3][list_of_file_numbers[3].index(531)] # id 531/1379 is duplicate. One removed.
print("Removing material id",list_of_file_numbers[14][list_of_file_numbers[14].index(2319)])
del list_of_file_numbers[14][list_of_file_numbers[14].index(2319)] # id 2319/2337 is duplicate. One removed.
print("Removing material id",list_of_file_numbers[14][list_of_file_numbers[14].index(2370)])
del list_of_file_numbers[14][list_of_file_numbers[14].index(2370)] # id 2370/2333 is duplicate. One removed.

object_ids = []

# Parallel forcycle:
for core in range(num_cpus): # get the object ids into a list
    object_ids.append( iterate_thru_all.remote("train",list_of_file_numbers[core],"  Relaxation step number", "  lattice_vector ", "            atom", "       atom_frac", 
                 "ESTIMATED overall HOMO-LUMO gap:", "  | Total energy                  :", "  Final atomic structure:", 
                 "  | Total energy of the DFT / Hartree-Fock s.c.f. calculation      :")
)
dataframes = []
for core in range(num_cpus): # get the actual dataframes from the object ids
    dataframes.append(ray.get(object_ids[core]))

# concatenate the dataframes
df_general_DATA = pd.concat([dataframes[core][0] for core in range(num_cpus)], ignore_index=True)
df_lattice_vector_relaxation_DATA = pd.concat([dataframes[core][1] for core in range(num_cpus)], ignore_index=True)
df_energy_relaxation_DATA = pd.concat([dataframes[core][2] for core in range(num_cpus)], ignore_index=True)
df_atoms_xyz_relaxation_DATA = pd.concat([dataframes[core][3] for core in range(num_cpus)], ignore_index=True)
df_atoms_frac_xyz_relaxation_DATA = pd.concat([dataframes[core][4] for core in range(num_cpus)], ignore_index=True)

# save the datasets:
df_general_DATA.to_csv("../train/relaxation/general.csv", index = False)
df_lattice_vector_relaxation_DATA.to_csv("../train/relaxation/lattice_vector_relaxation.csv", index = False)
df_energy_relaxation_DATA.to_csv("../train/relaxation/energy_relaxation.csv", index = False)
df_atoms_xyz_relaxation_DATA.to_csv("../train/relaxation/atoms_xyz_relaxation.csv", index = False)
df_atoms_frac_xyz_relaxation_DATA.to_csv("../train/relaxation/atoms_frac_xyz_relaxation.csv", index = False)

df_lattice_vector_DATA = pd.concat([dataframes[core][5] for core in range(num_cpus)], ignore_index=True)
df_energy_DATA = pd.concat([dataframes[core][6] for core in range(num_cpus)], ignore_index=True)
df_atoms_xyz_DATA = pd.concat([dataframes[core][7] for core in range(num_cpus)], ignore_index=True)
df_atoms_frac_xyz_DATA = pd.concat([dataframes[core][8] for core in range(num_cpus)], ignore_index=True)

df_lattice_vector_DATA.to_csv("../train/final/lattice_vector.csv", index = False)
df_energy_DATA.to_csv("../train/final/energy.csv", index = False)
df_atoms_xyz_DATA.to_csv("../train/final/atoms_xyz.csv", index = False)
df_atoms_frac_xyz_DATA.to_csv("../train/final/atoms_frac_xyz.csv", index = False)


# In[7]:


# test set
list_of_file_numbers = []
for k in range(num_cpus):
    list_of_file_numbers.append(list(range(1 +k*40,40+1 +k*40)))
object_ids = []

# Parallel forcycle:
for core in range(num_cpus): # get the object ids into a list
    object_ids.append( iterate_thru_all.remote("test",list_of_file_numbers[core],"  Relaxation step number", "  lattice_vector ", "            atom", "       atom_frac", 
                 "ESTIMATED overall HOMO-LUMO gap:", "  | Total energy                  :", "  Final atomic structure:", 
                 "  | Total energy of the DFT / Hartree-Fock s.c.f. calculation      :")
)
dataframes = []
for core in range(num_cpus): # get the actual dataframes from the object ids
    dataframes.append(ray.get(object_ids[core]))
    
    
# concatenate the dataframes
df_general_DATA = pd.concat([dataframes[core][0] for core in range(num_cpus)], ignore_index=True)
df_lattice_vector_relaxation_DATA = pd.concat([dataframes[core][1] for core in range(num_cpus)], ignore_index=True)
df_energy_relaxation_DATA = pd.concat([dataframes[core][2] for core in range(num_cpus)], ignore_index=True)
df_atoms_xyz_relaxation_DATA = pd.concat([dataframes[core][3] for core in range(num_cpus)], ignore_index=True)
df_atoms_frac_xyz_relaxation_DATA = pd.concat([dataframes[core][4] for core in range(num_cpus)], ignore_index=True)

df_general_DATA.to_csv("../test/relaxation/general.csv", index = False)
df_lattice_vector_relaxation_DATA.to_csv("../test/relaxation/lattice_vector_relaxation.csv", index = False)
df_atoms_xyz_relaxation_DATA.to_csv("../test/relaxation/atoms_xyz_relaxation.csv", index = False)
df_atoms_frac_xyz_relaxation_DATA.to_csv("../test/relaxation/atoms_frac_xyz_relaxation.csv", index = False)
df_energy_relaxation_DATA.to_csv("../test/relaxation/energy_relaxation.csv", index = False)


df_lattice_vector_DATA = pd.concat([dataframes[core][5] for core in range(num_cpus)], ignore_index=True)
df_energy_DATA = pd.concat([dataframes[core][6] for core in range(num_cpus)], ignore_index=True)
df_atoms_xyz_DATA = pd.concat([dataframes[core][7] for core in range(num_cpus)], ignore_index=True)
df_atoms_frac_xyz_DATA = pd.concat([dataframes[core][8] for core in range(num_cpus)], ignore_index=True)

df_lattice_vector_DATA.to_csv("../test/final/lattice_vector.csv", index = False)
df_energy_DATA.to_csv("../test/final/energy.csv", index = False)
df_atoms_xyz_DATA.to_csv("../test/final/atoms_xyz.csv", index = False)
df_atoms_frac_xyz_DATA.to_csv("../test/final/atoms_frac_xyz.csv", index = False)


# In[8]:


# Ending parallel calculation
ray.shutdown()


# In[ ]:




