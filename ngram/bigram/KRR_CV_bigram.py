#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
from shutil import rmtree
import sklearn
from time import time
import ngram
import jurka_utils

start = time()
[   unigram_matrix_train_path_VEGARD,
    unigram_matrix_test_path_VEGARD,
    unigram_cdf_train_path_VEGARD,
    unigram_cdf_test_path_VEGARD,
    df_egy_train_path_VEGARD,
    df_egy_test_path_VEGARD,
    unigram_matrix_train_path_RELAX,
    unigram_matrix_test_path_RELAX,
    unigram_cdf_train_path_RELAX,
    unigram_cdf_test_path_RELAX,
    df_egy_train_path_RELAX,
    df_egy_test_path_RELAX,
    unigram_matrix_train_path_FINAL,
    unigram_matrix_test_path_FINAL,
    unigram_cdf_train_path_FINAL,
    unigram_cdf_test_path_FINAL,
    df_egy_train_path_FINAL,
    df_egy_test_path_FINAL,
    dataset_label
] = jurka_utils.parse_input()

if dataset_label=="vegard":
    unigram_matrix_train_path = unigram_matrix_train_path_VEGARD
    unigram_matrix_test_path = unigram_matrix_test_path_VEGARD
    unigram_cdf_train_path = unigram_cdf_train_path_VEGARD
    unigram_cdf_test_path = unigram_cdf_test_path_VEGARD
    df_egy_train_path = df_egy_train_path_FINAL
    df_egy_test_path = df_egy_test_path_FINAL

if dataset_label=="relaxation":
    unigram_matrix_train_path = unigram_matrix_train_path_RELAX
    unigram_matrix_test_path = unigram_matrix_test_path_RELAX
    unigram_cdf_train_path = unigram_cdf_train_path_RELAX
    unigram_cdf_test_path = unigra_cdf_test_path_RELAX
    df_egy_train_path = df_egy_train_path_RELAX
    df_egy_test_path = df_egy_test_path_RELAX

if dataset_label=="final":
    unigram_matrix_train_path = unigram_matrix_train_path_FINAL
    unigram_matrix_test_path = unigram_matrix_test_path_FINAL
    unigram_cdf_train_path = unigram_cdf_train_path_FINAL
    unigram_cdf_test_path = unigram_cdf_test_path_FINAL
    df_egy_train_path = df_egy_train_path_FINAL
    df_egy_test_path = df_egy_test_path_FINAL

if dataset_label=="vegardfinal":
    unigram_matrix_train_path = unigram_matrix_train_path_FINAL
    unigram_matrix_test_path = unigram_matrix_test_path_VEGARD
    unigram_cdf_train_path = unigram_cdf_train_path_FINAL
    unigram_cdf_test_path = unigram_cdf_test_path_VEGARD
    df_egy_train_path = df_egy_train_path_FINAL
    df_egy_test_path = df_egy_test_path_FINAL

if dataset_label=="weirdest":
    unigram_matrix_train_path = unigram_matrix_train_path_RELAX
    unigram_matrix_test_path = unigram_matrix_test_path_VEGARD
    unigram_cdf_train_path = unigram_cdf_train_path_RELAX
    unigram_cdf_test_path = unigram_cdf_test_path_VEGARD
    df_egy_train_path = df_egy_train_path_RELAX
    df_egy_test_path = df_egy_test_path_FINAL

X = np.loadtxt(unigram_matrix_train_path, delimiter=',')
train_shape = X.shape

X_test = np.loadtxt(unigram_matrix_test_path, delimiter=',')
test_shape = X_test.shape

print("X: ", X.shape)
print("X_test: ", X_test.shape)

X_cdf = np.loadtxt(unigram_cdf_train_path, delimiter=",")
X_cdf_test = np.loadtxt(unigram_cdf_test_path, delimiter=",")
print("X_cdf: ", X_cdf.shape)
print("X_cdf_test: ", X_cdf_test.shape)

#scale = StandardScaler().fit(X_cdf)
#X_cdf = (X_cdf - scale.mean_)/scale.scale_

#X_cdf_test = (X_cdf_test - scale.mean_)/scale.scale_

X = np.concatenate([X,X_cdf],axis=1)
X_test = np.concatenate([X_test,X_cdf_test],axis=1)

X = np.concatenate([X, X_test])

#############################################################
#X = ngram.reduce_unigram_matrix(X, 5, reduce=True)
#print("Reduced unigram dim:", X.shape)
X = ngram.remove_empty(X)
print("X after empty removed:", X.shape)
#############################################################

X_test = X[train_shape[0]:, :]
print("X_test dim:", X_test.shape)

X = X[:train_shape[0], :]
print("X dim:", X.shape)

df_egy_train = pd.read_csv(df_egy_train_path)
df_egy_test = pd.read_csv(df_egy_test_path)

y_form = df_egy_train["formation_energy_ev_natom"].to_numpy()
y_form_test = df_egy_test["formation_energy_ev_natom"].to_numpy()

y_gap = df_egy_train["bandgap_energy_ev"].to_numpy()
y_gap_test = df_egy_test["bandgap_energy_ev"].to_numpy()

#sklearn.metrics.mean_squared_error

krr = KernelRidge() # alpha=1.0, kernel="rbf", gamma=1.0

# logarithmic lambda grid and finer sigma grid
base = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e0 ,1e-1, 1e1, 1e2, 1e3, 1e4, 1e5])

parameters = [{'krr__kernel': ['laplacian', 'rbf'],
               'krr__gamma': base, #np.concatenate(tuple(base*i for i in range(1,10))),
               'krr__alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]},
               ]

# RMSE after grid search 5-fold CV: 0.048602587588671274
# 
# best_parameters = [{'alpha': [0.001], 'gamma': [10.0], 'kernel': ['rbf']}]


parameters1 = [{'kernel': ['rbf'],
               'gamma': [float(2*i) for i in range(90, 201)],
               'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]},
               ]
#####################################
location = './cachedir_' + dataset_label
memory = joblib.Memory(location = location)
pipe = Pipeline([('scaler', StandardScaler()), ('krr', KernelRidge())], memory=memory)
#pipe = Pipeline([ ('krr', KernelRidge())], memory=memory)

#base = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
#parameters = [{'krr__kernel': ['laplacian', 'rbf'],
#               'krr__gamma': np.concatenate(tuple(base*i for i in range(1, 10))),#1 5
#               'krr__alpha': [0.001, 0.01, 0.1, 1, 10 ]}, # 0.000001, 0.00001, 0.0001, 
#               ]

#scale = StandardScaler().fit(X)
#X_test = (X_test - scale.mean_)/scale.scale_

if dataset_label in ("relaxation", "weirdest"):
    groups = df_egy_train["id"].to_numpy()
    group_kfold = GroupKFold(n_splits=5)
    cv = group_kfold.split(X, groups=groups)
    cv = list(cv)
else:
    cv = 5
####################################
clf = GridSearchCV(pipe,#krr 
                   param_grid=parameters, 
                   n_jobs=-1, 
                   refit="neg_mean_squared_error", 
                   cv=cv, 
                   verbose=0, 
                   pre_dispatch='2*n_jobs', 
                   error_score=np.nan,
                   return_train_score=False,# should be True but too expensive for relax
                   scoring=["neg_mean_squared_error", "neg_mean_absolute_error"]
                  )

clf.fit(X, y_form)
y_form_pred = clf.predict(X_test)
rmse_form = jurka_utils.rmse(y_form_test, y_form_pred)
rmsle_form = jurka_utils.rmsle(y_form_test, y_form_pred)
mae_form = jurka_utils.mae(y_form_test, y_form_pred)
form_best_params = clf.best_params_
pickle.dump(clf.cv_results_ , open( 'KRR_CV_bigram_cdf_2_competition_s_' + dataset_label + '.p.form', "wb" ) )

clf.fit(X, y_gap)
y_gap_pred = clf.predict(X_test)
rmse_gap = jurka_utils.rmse(y_gap_test, y_gap_pred)
rmsle_gap = jurka_utils.rmsle(y_gap_test, y_gap_pred)
mae_gap = jurka_utils.mae(y_gap_test, y_gap_pred)
gap_best_params = clf.best_params_
pickle.dump(clf.cv_results_ , open( 'KRR_CV_bigram_cdf_2_competition_s_' + dataset_label + '.p.gap', "wb" ) )

print("RMSE_form after grid search 5-fold CV:", rmse_form)
print("RMSLE_from after grid search 5-fold CV:", rmsle_form)
print("MAE_form after grid search 5-fold CV:", mae_form)
print()
print("Best parameters form:", form_best_params)

print("RMSE_gap after grid search 5-fold CV:", rmse_gap)
print("RMSLE_gap after grid search 5-fold CV:", rmsle_gap)
print("MAE_gap after grid search 5-fold CV:", mae_gap)
print()
print("Best parameters gap:", gap_best_params)

end = time()
print("Time elapsed:", (end - start)/60, " min.")

with open('KRR_CV_bigram_cdf_2_competition_s_' + dataset_label + '.output', 'w') as f:
    f.write("Time elapsed: " + str((end - start)/60) + " min.\n")
    f.write("X_test dim: " + str(X_test.shape) + "\n")
    f.write("X dim: " + str(X.shape) + "\n")

    f.write("RMSE_form after grid search 5-fold CV:\n")
    f.write(str(rmse_form) + "\n")
    f.write("RMSLE_form after grid search 5-fold CV:\n")
    f.write(str(rmsle_form) + "\n")
    f.write("MAE_form after grid search 5-fold CV:\n")
    f.write(str(mae_form) + "\n")
    for key in form_best_params.keys():
        f.write("%s,%s\n"%(key,form_best_params[key]))
        
    f.write("RMSE_gap after grid search 5-fold CV:\n")
    f.write(str(rmse_gap) + "\n")
    f.write("RMSLE_gap after grid search 5-fold CV:\n")
    f.write(str(rmsle_gap) + "\n")
    f.write("MAE_gap after grid search 5-fold CV:\n")
    f.write(str(mae_gap) + "\n")
    f.write("Best parameters:\n")
    for key in gap_best_params.keys():
        f.write("%s,%s\n"%(key,gap_best_params[key]))

##################
memory.clear(warn=False)
rmtree(location)
