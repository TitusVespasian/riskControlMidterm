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
from sklearn.model_selection import StratifiedKFold  # 交叉验证

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 归一化需要
from sklearn.preprocessing import MinMaxScaler


import numpy as np
import pandas as pd
import pickle


# %% 装载数据集，训练
data = pd.read_csv("./data/train/train_all.csv")

train_all = (data-data.min())/(data.max()-data.min())



X = train_all.drop(['target'], axis=1)
y = train_all.pop('target')

X_train, X_check, y_train, y_check = train_test_split(
    X, y, random_state=8, test_size=0.2, stratify=y)


# %% 模型的保存，载入与检查

lr = LogisticRegression(
    penalty='l2',
    class_weight="balanced",  # 平衡权重
    random_state=1,  # 固定随机数种子
    max_iter=1600,
    n_jobs=-1  # 使用全部的CPU。但是liblinear的时候没用
)

param_grid = {
    'C': [0.6, 0.8, 1.0],
    'solver': ['sag', 'saga'],
}

kflod = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

grid_search = GridSearchCV(
    lr, param_grid, scoring='roc_auc', n_jobs=-1, cv=kflod)

grid_result = grid_search.fit(
    X_train, y_train)  # 运行网格搜索

print(grid_result.best_score_, grid_search.best_params_)

# %% 模型训练与检查
lr.fit(X_train, y_train)

# predict_array 为预测概率矩阵。我希望取出第一列。
predict_array = lr.predict_proba(X_check)
y_predict = predict_array[:, 1]

test_auc = metrics.roc_auc_score(y_check, y_predict)  # 我分出来的验证集上的auc值

print("manual AUC(%):", test_auc)

# %%  保存文件
outfile = open("./saved_model/lr_model.pickle", "wb")
