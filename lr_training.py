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
import numpy as np
import pandas as pd
import pickle

# 如果使用GridSearchCV，把下面这一项设置为True
use_gridsearch = False

# %% 参数设置

lr = LogisticRegression(
    penalty='l2',
    class_weight="balanced",  # 平衡权重
    random_state=80,  # 固定随机数种子
    # solver='liblinear',
    solver='sag',
    max_iter=800,
    n_jobs=-1,  # 使用全部的CPU。但是liblinear的时候没用
)

# %% 装载数据集，训练
# todo: 合并完成之后，改成已经合并的数据集

train_all = pd.read_csv("data/train/Master_Training_Modified.csv")
y_train = train_all.pop('target')
lr.fit(train_all.values[:, 1:], y_train)

# %% 模型的保存，载入与检查

# 保存文件
outfile = open("./saved_model/lr_model.pickle", "wb")
pickle.dump(lr, outfile)
outfile.close()

# 可以直接打印吗？我没试过
# print(lr)
