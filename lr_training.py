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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 归一化需要
from sklearn.preprocessing import MinMaxScaler


import numpy as np
import pandas as pd
import pickle


# %% 装载数据集，训练
train_all = pd.read_csv("./data/train/train_all.csv")

scaler = MinMaxScaler()  # 实例化
scaler.fit_transform(train_all)

#%% 训练数据

X = train_all.drop(['target'], axis=1)
y = train_all.pop('target')


# %% 模型的保存，载入与检查

lr = LogisticRegression(
    penalty='l2',
    class_weight="balanced",  # 平衡权重
    random_state=80,  # 固定随机数种子
    solver='sag',
    max_iter=1600,
    n_jobs=-1  # 使用全部的CPU。但是liblinear的时候没用
)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print(lr.score(X_test, y_test))

# %%  保存文件
outfile = open("./saved_model/lr_model.pickle", "wb")
