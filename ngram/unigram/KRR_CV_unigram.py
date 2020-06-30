#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import json
import sklearn
import ngram
import jurka_utils

def gen_unigram_matrix(flag, df_frac, df_latt, df_gen, ALL_COORDINATIONS_UNIGRAM, ALL_COORDINATIONS_UNIGRAM_USEFUL):
    """Generates the unigram matrix.
    
    Keyword arguments:
    flag -- type of dataset used. options: "relaxation", "final", "vegard"
    mat -- loaded json object
    df_frac -- dataframe of fractional coordinates
    df_latt -- dataframe of lattice vectors
    df_gen -- dataframe of general data
    ALL_COORDINATIONS_UNIGRAM -- list of coordinations ranging from 0 to 10
    ALL_COORDINATIONS_UNIGRAM_USEFUL -- list of coordinations wanted in the matrix
    """
    if flag=="final":
        ajdis = df_gen["id"]
        try:
            indexing = df_latt[["id", "relaxation_step_number"]]
        except KeyError:
            #df_latt["relaxation_step_number"] = -1
            df_latt.insert(1, "relaxation_step_number", [-1 for i in range(len(df_latt))])
            #df_frac["relaxation_step_number"] = -1
            df_frac.insert(1, "relaxation_step_number", [-1 for i in range(len(df_frac))])
            indexing = df_latt[["id", "relaxation_step_number"]]
    if flag=="relaxation" or flag=="vegard":
        ajdis = df_gen["id"]
        indexing = df_latt[["id", "relaxation_step_number"]]
    X = []
    for ajdi in ajdis:
        for rsn in indexing[indexing["id"]==ajdi].relaxation_step_number.values:

            material = ngram.init_material(df_frac, df_latt, ajdi, rsn)
            mdm = material.distance_matrix

            #tridiag = jurka_utils.matrix_decoding(mat[str(ajdi)][str(rsn)])
            #mdm = tridiag + np.transpose(tridiag)

            G = ngram.gen_graph(mdm, material, "metal_oxygen_cons", hypar=ngram.get_hypar(df_frac, df_latt, df_gen, ajdi, rsn))
            coordinations = ngram.get_coordinations(G)
            hist = {}
            row = []
            for i in ALL_COORDINATIONS_UNIGRAM:
                hist.update({i:0})
            for i in coordinations.keys():
                name = i.split()[0]
                coord = name + "-" + str(coordinations[i])
                hist.update({coord:hist[coord]+1})
            for i in ALL_COORDINATIONS_UNIGRAM_USEFUL:
                row.append(hist[i]/material.lattice.volume)
            X.append(row)
        print(ajdi, "ended.")
    return np.array(X)

[  #distance_matrix_path, 
    unigram_matrix_path, 
    dependent_variable_matrix_path,
    unigram_matrix_test_path,
    dependent_variable_matrix_test_path,
    frac_train_path,
    latt_train_path,
    gen_train_path,
    egy_train_path,
    frac_test_path,
    latt_test_path,
    gen_test_path,
    egy_test_path,
] = jurka_utils.parse_input()

df_frac_train = pd.read_csv(frac_train_path)
df_latt_train = pd.read_csv(latt_train_path)
df_gen_train = pd.read_csv(gen_train_path)
df_egy_train = pd.read_csv(egy_train_path)

df_frac_test = pd.read_csv(frac_test_path)
df_latt_test = pd.read_csv(latt_test_path)
df_gen_test = pd.read_csv(gen_test_path)
df_egy_test = pd.read_csv(egy_test_path)

ALL_COORDINATIONS_UNIGRAM = [
    'Al-0',
    'Al-1',
    'Al-2',
    'Al-3',
    'Al-4', 
    'Al-5', 
    'Al-6', 
    'Al-7',
    'Al-8',
    'Al-9',
    'Al-10',
    'Ga-0',
    'Ga-1',
    'Ga-2',
    'Ga-3', 
    'Ga-4', 
    'Ga-5', 
    'Ga-6', 
    'Ga-7', 
    'Ga-8',
    'Ga-9',
    'Ga-10',
    'In-0',
    'In-1',
    'In-2',
    'In-3', 
    'In-4', 
    'In-5', 
    'In-6', 
    'In-7', 
    'In-8',
    'In-9',
    'In-10',
    'O-0',
    'O-1',
    'O-2', 
    'O-3', 
    'O-4', 
    'O-5', 
    'O-6',
    'O-7',
    'O-8',
    'O-9',
    'O-10',
    ]

ALL_COORDINATIONS_UNIGRAM_USEFUL = [ 
    'Al-4', 
    'Al-5', 
    'Al-6',  
    'Ga-4', 
    'Ga-5', 
    'Ga-6',  
    'In-4', 
    'In-5', 
    'In-6', 
    'O-2', 
    'O-3', 
    'O-4', 
    'O-5', 
    ]

#with open(distance_matrix_path) as jsonfile:
#    mat = json.load(jsonfile)
    
#X = np.loadtxt(unigram_matrix_path, delimiter=',')
#y = np.loadtxt(dependent_variable_matrix_path, delimiter=',')

#X_test = np.loadtxt(unigram_matrix_test_path, delimiter=',')
#y_test = np.loadtxt(dependent_variable_matrix_test_path, delimiter=',')

X = gen_unigram_matrix("relaxation", df_frac_train, df_latt_train, df_gen_train, ALL_COORDINATIONS_UNIGRAM, ALL_COORDINATIONS_UNIGRAM)
y = df_egy_train["formation_energy_ev_natom"].to_numpy()

X_test = gen_unigram_matrix("relaxation", df_frac_test, df_latt_test, df_gen_test, ALL_COORDINATIONS_UNIGRAM, ALL_COORDINATIONS_UNIGRAM)
y_test = df_egy_test["formation_energy_ev_natom"].to_numpy()

#sklearn.metrics.mean_squared_error

krr = KernelRidge() # alpha=1.0, kernel="rbf", gamma=1.0

base = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])

parameters = [{'kernel': ['rbf'],
               'gamma': np.concatenate(tuple(base*i for i in range(1,2))),
               'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]},
               ]

# RMSE after grid search 5-fold CV: 0.048602587588671274
# 
# best_parameters = [{'alpha': [0.001], 'gamma': [10.0], 'kernel': ['rbf']}]


parameters1 = [{'kernel': ['rbf'],
               'gamma': [float(2*i) for i in range(90, 201)],
               'alpha': [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01]},
               ]
 
clf = GridSearchCV(krr, 
                   parameters, 
                   n_jobs=-1, 
                   refit=True, 
                   cv=None, 
                   verbose=0, 
                   pre_dispatch='2*n_jobs', 
                   error_score=np.nan, 
                   return_train_score=False
                  )

clf.fit(X, y)

y_pred = clf.predict(X_test)

rmse = jurka_utils.rmse(y_test, y_pred)

rmsle = jurka_utils.rmsle(y_test, y_pred)

mae = jurka_utils.mae(y_test, y_pred)

print("RMSE after grid search 5-fold CV:", rmse)
print("RMSLE after grid search 5-fold CV:", rmsle)
print("MAE after grid search 5-fold CV:", mae)
print()
print("Best parameters:", clf.best_params_)

with open('KRR_CV_unigram_relaxation_dataset.output', 'w') as f:
    f.write("RMSE after grid search 5-fold CV:\n")
    f.write(str(rmse) + "\n")

    f.write("RMSLE after grid search 5-fold CV:\n")
    f.write(str(rmsle) + "\n")

    f.write("MAE after grid search 5-fold CV:\n")
    f.write(str(mae) + "\n")

    f.write("Best parameters:\n")
    for key in clf.best_params_.keys():
        f.write("%s,%s\n"%(key,clf.best_params_[key]))

