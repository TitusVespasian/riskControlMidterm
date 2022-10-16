#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 21:22:13 2022

@author: xzytql,zlwtql
"""

# %% import
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn import metrics
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

# %% 特征分析函数


def modelfit(alg, dtrain, y_train, dtest=None, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain.values[:, :], label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # 建模
    alg.fit(dtrain.values[:, :], y_train, eval_metric='auc')

    # 对训练集预测
    dtrain_predictions = alg.predict(dtrain.values[:, :])
    dtrain_predprob = alg.predict_proba(dtrain.values[:, :])[:, 1]

    # 输出模型的一些结果
    # print(dtrain_predictions)
    # print(alg.predict_proba(dtrain.as_matrix()[: ,1:]))
    print(cvresult.shape[0])
    print("\n关于现在这个模型")
    print("准确率 : %.4g" % metrics.accuracy_score(y_train, dtrain_predictions))
    print("AUC 得分 (训练集): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    for i in range(20):
        print(dtrain.columns[int(feat_imp.index[i][1:])], feat_imp[i])
    print(feat_imp.shape)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


xgb1 = XGBClassifier(
    learning_rate=0.05,
    # n_estimators =,
    # max_depth=,
    # min_child_weight =,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27
)



# %% 开始分析特征

train_master = pd.read_csv(r"./data/train/Master_Training_Cleaned.csv")
y_train = train_master["target"].values

# 'UserInfo_9'
print(train_master['UserInfo_9'].unique())
# ['中国移动' '中国电信' '不详' '中国联通']
train_master=pd.get_dummies(train_master.UserInfo_9)

# 'UserInfo_22'
print(train_master['UserInfo_22'].unique())
# ['D' '未婚' '已婚' '不详' '离婚' '再婚' '初婚']
print(train_master['UserInfo_22'].value_counts())
"""
D     27867
未婚     1330
已婚      468
不详      296
离婚       34
再婚        4
初婚        1
Name: UserInfo_22, dtype: int64
"""
train_master.loc[train_master['UserInfo_22']=='初婚','UserInfo_22']="已婚"
train_master=pd.get_dummies(train_master.UserInfo_22)

# 'UserInfo_23'
print(train_master['UserInfo_23'].value_counts())
dummies_UserInfo_23=pd.get_dummies(train_master['UserInfo_23'], prefix='UserInfo_23')
modelfit(xgb1, dummies_UserInfo_23, y_train)

dummies_UserInfo_2 = pd.get_dummies(train_master['UserInfo_2'], prefix='UserInfo_2')


#modelfit(xgb1, dummies_UserInfo_2, y_train)
# modelfit(xgb1, dummies_UserInfo_2, y_train)
