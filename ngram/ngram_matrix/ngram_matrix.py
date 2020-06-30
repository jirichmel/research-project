#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import itertools as it
import jurka_utils
import ngram
import ray

[   frac_train_path,
    latt_train_path,
    gen_train_path,
    egy_train_path,
    frac_test_path,
    latt_test_path,
    gen_test_path,
    egy_test_path,
    NUM_CPUS,
] = jurka_utils.parse_input()

df_frac_train = pd.read_csv(frac_train_path)
df_latt_train = pd.read_csv(latt_train_path)
df_gen_train = pd.read_csv(gen_train_path)
df_egy_train = pd.read_csv(egy_train_path)

df_frac_test = pd.read_csv(frac_test_path)
df_latt_test = pd.read_csv(latt_test_path)
df_gen_test = pd.read_csv(gen_test_path)
df_egy_test = pd.read_csv(egy_test_path)

#matrix = ngram.gen_ngram_matrix("relaxation", df_frac_train, df_latt_train, df_gen_train, 1, ngram.ALL_COORDINATIONS_UNIGRAM)
#print("Dimensions: ", matrix.shape)
#np.savetxt("unigram_relaxation_train.csv", matrix, delimiter=",")

#y = df_egy_train[["formation_energy_ev_natom", "bandgap_energy_ev"]].to_numpy()

#matrix = ngram.gen_ngram_matrix("relaxation", df_frac_test, df_latt_test, df_gen_test, 1, ngram.ALL_COORDINATIONS_UNIGRAM)
#print("Dimensions: ", matrix.shape)
#np.savetxt("unigram_relaxation_test.csv", matrix, delimiter=",")

#y = df_egy_test[["formation_energy_ev_natom", "bandgap_energy_ev"]].to_numpy()


#matrix = ngram.gen_ngram_matrix("relaxation", df_frac_train, df_latt_train, df_gen_train, 3, ngram.ALL_COORDINATIONS_TRIGRAM)
#print("Dimensions: ", matrix.shape)
#np.savetxt("trigram_relaxation_train.csv", matrix, delimiter=",")

#matrix = ngram.gen_ngram_matrix("relaxation", df_frac_test, df_latt_test, df_gen_test, 3, ngram.ALL_COORDINATIONS_TRIGRAM)
#print("Dimensions: ", matrix.shape)
#np.savetxt("trigram_relaxation_test.csv", matrix, delimiter=",")

ray.init(num_cpus=NUM_CPUS)

#df_gen_train = df_gen_train[df_gen_train["id"]<601]

#df_gen_train = df_gen_train[(df_gen_train["id"]<1201) & (df_gen_train["id"]>=601)]

df_gen_train = df_gen_train[(df_gen_train["id"]<1801) & (df_gen_train["id"]>=1201)]

#df_gen_train = df_gen_train[df_gen_train["id"]>=1801]

def calc_parts(name, NUM_CPUS, func, type_of_calc, df_frac, df_latt, df_gen, n, coordinations):
    chunks = np.array_split(df_gen, NUM_CPUS)
    object_ids = []
    decorated_func = ray.remote(func)
    # Parallel forcycle:
    for chunk in chunks: # get the object ids into a list
        object_ids.append(decorated_func.remote(type_of_calc, df_frac, df_latt, chunk, n, coordinations)) # remote function goes here

    actual_parts = []

    for obj in object_ids: # get the actual parts from the object ids
        actual_parts.append(ray.get(obj))
    
    matrix = np.concatenate(actual_parts)
    print("Dimensions: ", matrix.shape)
    np.savetxt(name, matrix, delimiter=",")

calc_parts("quadgram_relaxation_train_3.csv", NUM_CPUS, ngram.gen_ngram_matrix, "relaxation", df_frac_train, df_latt_train, df_gen_train, 4, ngram.ALL_COORDINATIONS_QUADGRAM)

#calc_parts("quadgram_relaxation_test.csv", NUM_CPUS, ngram.gen_ngram_matrix, "relaxation", df_frac_test, df_latt_test, df_gen_test, 4, ngram.ALL_COORDINATIONS_QUADGRAM)


ray.shutdown()
