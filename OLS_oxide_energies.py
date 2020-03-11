#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import linear_model


# In[ ]:


# Ordinary Least Squares to get a more accurate value of formation energies of input oxides in the formation energy calculation:
ols = linear_model.LinearRegression(fit_intercept=False)
df_general = pd.read_csv("../train/relaxation/general.csv")
df = pd.read_csv("../train.csv/train.csv") # load the kaggle train dataset
x = df["formation_energy_ev_natom"] # load the kaggle formation energy
natoms = df["number_of_total_atoms"]
x = x.drop([463, 2188, 125, 1214, 352, 307, 530, 2318, 2369])
x = x.to_numpy()
natoms = natoms.drop([463, 2188, 125, 1214, 352, 307, 530, 2318, 2369])
natoms = natoms.to_numpy()
df = pd.read_csv("../train/final/energy.csv") # load our energy data
y = df["formation_energy_ev_natom"].to_numpy() # the total energy is stored here!!
assert len(y)==len(x),"The lengths don't match."
b = np.divide(y,natoms) - 0.4*x # the right side
A = df_general[["percent_atom_al", "percent_atom_ga", "percent_atom_in"]].to_numpy()
ols.fit(A,b)
[x, y, z] = ols.coef_
print("The formation energies of the oxides calculated using OLS are:",x,y,z)
f = open("../OLS_oxide_energies", "w")
f.write("# The formation energies of the oxides calculated using OLS: E_Al2O3,E_Ga2O3,E_In2O3\n")
f.write(",".join([str(x),str(y),str(z)])+"\n")
f.close()

