#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 22:28:14 2022

Description:
Logistic_Regression 模型的训练

@author: xzyzdstql
"""

# %% import modules
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# %% 装载数据集，训练
data = pd.read_csv("./data/train/train_all.csv")

train_all = (data-data.min())/(data.max()-data.min())

X = train_all.drop(['target'], axis=1)
y = train_all.pop('target')

X_train, X_check, y_train, y_check = train_test_split(
    X, y, random_state=1, test_size=0.2, stratify=y)


# %% model definition

rfc = RandomForestClassifier()
parameters = {
    'n_estimators': range(30, 80, 10),
    'max_depth': range(3, 10, 2),
    'min_samples_leaf': [5, 6, 7],
    'max_features': [1, 2, 3]
}

grid_rfc = GridSearchCV(rfc, parameters, scoring='roc_auc')
grid_rfc.fit(X_train, y_train)
print(grid_rfc.best_params_, grid_rfc.best_score_)


#%% manual check
rfc.fit(X_train, y_train)

# predict_array 为预测概率矩阵。我希望取出第一列。
predict_array = rfc.predict_proba(X_check)
y_predict = predict_array[:, 1]

test_auc = metrics.roc_auc_score(y_check, y_predict)  # 我分出来的验证集上的auc值

print("manual AUC(%):", test_auc)
