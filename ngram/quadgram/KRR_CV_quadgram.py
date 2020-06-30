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

[   ngram_train_path,
    ngram_test_path,
    egy_train_path,
    egy_test_path,
] = jurka_utils.parse_input()

X = np.loadtxt(open(ngram_train_path, "r"), delimiter=",")
X_test = np.loadtxt(open(ngram_test_path, "r"), delimiter=",")

df_egy_train = pd.read_csv(egy_train_path)
df_egy_test = pd.read_csv(egy_test_path)

y = df_egy_train["formation_energy_ev_natom"].to_numpy()
y_test = df_egy_test["formation_energy_ev_natom"].to_numpy()

krr = KernelRidge() # alpha=1.0, kernel="rbf", gamma=1.0

base = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])

parameters = [{'kernel': ['rbf'],
               'gamma': np.concatenate(tuple(base*i for i in range(1, 2))),
               'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]},
               ]

parameters1 = [{'kernel': ['rbf'],
               'gamma': [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
               'alpha': [0.1, 0.05, 0.01, 0.005, 0.001]},
               ]
 
clf = GridSearchCV(krr, 
                   parameters1, 
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

with open('KRR_CV_quadgram_final_dataset.output', 'w') as f:
    f.write("RMSE after grid search 5-fold CV:\n")
    f.write(str(rmse) + "\n")

    f.write("RMSLE after grid search 5-fold CV:\n")
    f.write(str(rmsle) + "\n")

    f.write("MAE after grid search 5-fold CV:\n")
    f.write(str(mae) + "\n")

    f.write("Best parameters:\n")
    for key in clf.best_params_.keys():
        f.write("%s,%s\n"%(key,clf.best_params_[key]))

