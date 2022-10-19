#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import modules

import pandas as pd
import matplotlib.pylab as plt

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split

import pickle


# %% 装载数据集，训练

# %% 装载数据集，训练
train_all = pd.read_csv("data/train/train_all.csv")
X = train_all.drop("target", axis=1)
y = train_all.pop('target')

# 正样本的数目显著少于负样本，故计算权重
negative_num = y.value_counts()[0]
positive_num = y.value_counts()[1]
adjusted_weight = round(negative_num / positive_num, 2)  # 正例的权值，保留2位小数

X_train, X_check, y_train, y_check = train_test_split(
    X, y, random_state=1, test_size=0.2, stratify=y)

# %% train_model-1

xgb1 = XGBClassifier(
    learning_rate=0.2,
    # n_estimators =,
    max_depth=5,
    min_child_weight=0.5,
    gamma=0.3,
    # subsample=,
    nthread=-1,
    scale_pos_weight=adjusted_weight,
    # tree_method='gpu_hist'
)

# %% 模型的保存，载入与检查
xgb1.fit(X_train, y_train)

# %% check

print(xgb1.score(X_check, y_check))
