#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import modules

import pandas as pd
import matplotlib.pylab as plt

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import sklearn.preprocessing as preprocessing

import pickle

# %% train_model-1

xgb1 = XGBClassifier(
    learning_rate=0.2,
    # n_estimators =,
    max_depth=5,
    min_child_weight=0.5,
    gamma=0.3,
    # subsample=,
    # colsample_bytree =,
    objective='binary:logistic',
    nthread=-1,
    # scale_pos_weight=,
    tree_method='gpu_hist'
)

# %% 装载数据集，训练
# todo: 合并完成之后，改成已经合并的数据集

train_all = pd.read_csv("data/train/Master_Training_Modified.csv")
y_train = train_all.pop('target')

xgb1.fit(train_all.values[:, 1:], y_train)

# %% 模型的保存，载入与检查

# 保存文件
outfile = open("./saved_model/lr_model.pickle", "wb")
pickle.dump(xgb1, outfile)
outfile.close()
