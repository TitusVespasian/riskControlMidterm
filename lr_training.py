#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 22:28:14 2022

Description:
Logistic_Regression 模型的训练

@author: xzyzdstql
"""

# %% import

from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



import numpy as np
import pandas as pd
import pickle

# 如果使用GridSearchCV，把下面这一项设置为True
use_gridsearch = True

# %% 参数设置

param_grid = [
    {
        "penalty": ['l2'],
        "class_weight": ["balanced"],  # 平衡权重
        "random_state": [80],  # 固定随机数种子
        "solver": ['sag'],
        "max_iter": [800],
        "n_jobs": [-1]  # 使用全部的CPU。但是liblinear的时候没用
    }
]

# %% 装载数据集，训练

train_all = pd.read_csv("data/train/train_all.csv")
X = train_all.drop(['target'], axis=1)
y = train_all.pop('target')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.2)


# %% 模型的保存，载入与检查

lr = LogisticRegression()
clf = GridSearchCV(lr, param_grid, cv=5)
clf.fit(X_train, y_train)

print(clf.best_params_)
best_model = clf.best_estimator_

y_pred = best_model.predict(X_test)
print('accuracy', accuracy_score(y_test, y_pred))


# %%  保存文件
outfile = open("./saved_model/lr_model.pickle", "wb")
pickle.dump(lr, outfile)
outfile.close()
