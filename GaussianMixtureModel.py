#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 20:13:02 2018

@author: ashwath
"""
###############################################################################
#Import Libraries

import itertools 
from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GMM
import pandas as pd
import matplotlib as mpl
import math

###############################################################################
#Import Dataset 
dataset_old = pd.read_csv('BH11D.csv')
dataset_old = dataset_old.iloc[:,[1,2,3,8,9] ]
dataset_old = dataset_old.dropna(how='any',axis=0)
dataset = pd.read_csv('BH11D.csv')
dataset = dataset.iloc[:,[1,2,3,8,9] ]
dataset = dataset.dropna(how='any',axis=0)
dataset['mean'] = dataset_old.mean(axis=1)
mean_df = dataset.iloc[:,4 ]
total_size=len(mean_df)
train_size=math.floor(0.75*total_size) 

#training dataset
train=mean_df.head(train_size)
train_all = dataset_old.head(train_size)
#test dataset
test=mean_df.tail(len(mean_df) -train_size)
test_all = dataset_old.tail(len(mean_df) -train_size)
test_one = test_all.iloc[[420,421],:]

X_train = train.values
X_train_all = train_all.values
X_train = X_train.reshape(-1, 1)

X_test = test.values
X_test_all = test_all.values
X_test = X_test.reshape(-1, 1)

X_test_one = test_one.values
X_test_one = X_test_one.reshape(-1, 1)


###############################################################################
lowest_bic = np.infty
bic = []
n_components_range = range(1,200 ) # specifying maximum number of clusters
cv_types = ['spherical', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a mixture of Gaussians with EM
        gmm = GMM(n_components=n_components, covariance_type=cv_type)
        gmm.fit(X_train_all)
        bic.append(gmm.bic(X_train_all))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
# plot the BIC
bic = np.array(bic)
clf = best_gmm
print(clf)
prob = clf.predict_proba(dataset_old)
gmm_predict = clf.predict(dataset_old)

GMM_Predicted_Probabilities = pd.DataFrame(prob)
GMM_Predicted_Probabilities.to_csv("GMM_Predicted_Probabilities", index = False, encoding='utf-8',  header=None)

GMM_Predicted_Components = pd.DataFrame(gmm_predict)
GMM_Predicted_Components.to_csv("GMM_Predicted_Components", index = False, encoding='utf-8',  header=None)

BIC = pd.DataFrame(bic)
BIC.to_csv("BIC_200", index = False, encoding='utf-8',  header=None)

FinalGMMSelectedModel=open('./GMMFinalModelDetailsfile', 'w+')
print(clf, file=FinalGMMSelectedModel)
FinalGMMSelectedModel.close()
