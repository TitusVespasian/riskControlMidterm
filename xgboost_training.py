#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import modules

import pandas as pd
import matplotlib.pylab as plt

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import BaggingClassifier
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold  # 交叉验证
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import pickle


# %% 装载数据集，训练

train_all = pd.read_csv("data/train/train_all.csv")
X = train_all.drop("target", axis=1)
y = train_all.pop('target')


# 正样本的数目显著少于负样本，故计算权重
negative_num = y.value_counts()[0]
positive_num = y.value_counts()[1]
adjusted_weight = round(negative_num / positive_num, 2)  # 正例的权值，保留2位小数

X_train, X_check, y_train, y_check = train_test_split(
    X, y, random_state=8, test_size=0.2, stratify=y)


# %% 装载数据集，训练

train_all = pd.read_csv("data/train/train_all.csv")
X = train_all.drop("target", axis=1)
y = train_all.pop('target')


# 正样本的数目显著少于负样本，故计算权重
negative_num = y.value_counts()[0]
positive_num = y.value_counts()[1]
adjusted_weight = round(negative_num / positive_num, 2)  # 正例的权值，保留2位小数


# _train后缀是训练集和验证集的
X_train, X_check, y_train, y_check = train_test_split(
    X, y, random_state=8, test_size=0.2, stratify=y)


# %% train_model-1


xgb1 = XGBClassifier(
    learning_rate=0.1,
    n_estimators=70,
    max_depth=3,
    min_child_weight=7,
    gamma=0,
    # subsample=,
    nthread=-1,
    scale_pos_weight=adjusted_weight,
    # 如果可以，在远端服务器运行，把下面的注释放开以获得显卡支持
    # tree_method='gpu_hist',
    random_state=1,
    use_label_encoder=False,
)
xgb_bagging=BaggingClassifier(base_estimator=xgb1)
# param_grid = {
#     'learning_rate': [0.05, 0.1, 0.2, 0.3],
#     'n_estimators': [10, 20, 30, 50, 70],
#     'min_child_weight': [3, 5, 7],
#     'gamma': [0, 0.1, 0.2, 0.35],
# }
#
#
# kflod = StratifiedKFold(n_splits=7, shuffle=True, random_state=1)
# # %% GridSearchCV fit
#
# grid_search = GridSearchCV(
#     xgb1, param_grid, scoring='roc_auc', n_jobs=-1, cv=kflod)
#
# grid_result = grid_search.fit(
#     X_train, y_train, eval_metric="auc", verbose=4)  # 运行网格搜索
#
# print(grid_result.best_score_, grid_search.best_params_)

# 如果需要每一组参数的评估值，放开下面的注释
# means = grid_result.cv_results_['mean_test_score']
# params = grid_result.cv_results_['params']
# for mean, param in zip(means, params):
#     print("%f [with] %r" % (mean, param))


# %% 模型训练与检查
xgb_bagging.fit(X_train, y_train)

# predict_array 为预测概率矩阵。我希望取出第一列。
predict_array = xgb_bagging.predict_proba(X_check)
y_predict = predict_array[:, 1]

test_auc = metrics.roc_auc_score(y_check, y_predict)  # 我分出来的验证集上的auc值

print("AUC(%):", test_auc)


# %% save model

outfile = open("./saved_model/lr_model.pickle", "wb")
pickle.dump(xgb_bagging, outfile)
outfile.close()
