#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 22:28:14 2022

Description:这一部分对应于原demo的“建模调参与优化”一节的后半部分。
主要用途是对于最终的模型进行训练。

在本文件的修改，请使用注释加以标记，写报告好写

@author: xzyzdstql
"""

# %% import modules

import pandas as pd

# 我都import了什么东西？
from sklearn import metrics
from sklearn.model_selection import cross_validate, GridSearchCV

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing as preprocessing

import matplotlib.pylab as plt

# %% read_data

train_all = pd.read_csv('data/train/train_all.csv')
y_train = train_all.pop('target')


# %% function definion

def modelfit(alg, dtrain, y_train, dtest=None, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain.values[:, 1:], label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()[
                          'n_estimators'], nfold=cv_folds, early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # 建模
    alg.fit(dtrain.values[:, 1:], y_train, eval_metric='auc')

    # 对训练集预测
    dtrain_predictions = alg.predict(dtrain.values[:, 1:])
    dtrain_predprob = alg.predict_proba(dtrain.values[:, 1:])[:, 1]

    # 输出模型的一些结果
    #
    #
    print(cvresult.shape[0])
    print("\n当前模型")
    print("准确率 : %.4g" % metrics.accuracy_score(y_train, dtrain_predictions))
    print("AUC 得分 (训练集): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    print(feat_imp.head(25))
    print(feat_imp.shape)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


# %% train_model-1

xgb1 = XGBClassifier(
    # learning_rate=,
    # n_estimators =,
    # max_depth=,
    # min_child_weight =,
    # gamma =,
    # subsample=,
    # colsample_bytree =,
    objective='binary:logistic',
    # nthread=,
    # scale_pos_weight=,
    # seed = ,
    tree_method='gpu_hist'
)

modelfit(xgb1, train_all, y_train)




# GridSearchCV对于所有的参数进行自动的组合，选出效果最好的
clf_lr = GridSearchCV(lr, parameters, cv=3)
print('开始训练')
clf_lr.fit(train_all.values[:, 1:], y_train)
print('模型训练结束')
clf_lr

# %% train_model-3

clf_lr_accuracy = clf_lr.score(train_all.values[:, 1:], y_train)
print(clf_lr_accuracy)
clf_lr.cv_results_, clf_lr.best_params_, clf_lr.best_score_
