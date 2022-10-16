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
import re

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 8, 6
#from matplotlib.font_manager import FontProperties
#myfont = FontProperties(fname=r"external-libraries/yahei.ttf",size=12)
rcParams['font.sans-serif'] = ['SimHei']


# %% 特征分析函数
def getFeaturename(name):
    try:
        rex_str=name[name.rfind('_')+1:]
    except:
        print('Not successful')
        print('name:'+name)
        return name
    return rex_str

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
    feat_list=[]
    for i in range(min(len(feat_imp),20)):
        temp_str=getFeaturename(dtrain.columns[int(feat_imp.index[i][1:])])
        print(temp_str, feat_imp[i])
        feat_list.append(temp_str)
    feat_imp_new=pd.DataFrame(data=feat_list,columns=['feature_name'],index=feat_imp.index)
    feat_temp = pd.DataFrame(data=feat_imp,columns=['feature_score'])
    feat_imp_new = pd.concat([feat_imp_new, feat_temp], axis=1)
    print(feat_imp.shape)
    feat_imp_new.plot(x='feature_name',y='feature_score',kind='barh',
                      title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    #plt.gcf().subplots_adjust(left=0.2)
    plt.show()


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
train_master=train_master.join(pd.get_dummies(train_master.UserInfo_9,prefix="UserInfo_9"))
train_master.drop('UserInfo_9',axis=1,inplace=True)
try:
    print(train_master.UserInfo_9)
except AttributeError:
    print("successfully droped!")

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
train_master=train_master.join(pd.get_dummies(train_master.UserInfo_22,prefix="UserInfo_22"))
train_master.drop('UserInfo_22',axis=1,inplace=True)
try:
    print(train_master.UserInfo_9)
except AttributeError:
    print("successfully droped!")

# 'UserInfo_23'
print(train_master['UserInfo_23'].value_counts())
dummies_UserInfo_23=pd.get_dummies(train_master['UserInfo_23'], prefix='UserInfo_23')
modelfit(xgb1, dummies_UserInfo_23, y_train)

dummies_UserInfo_2 = pd.get_dummies(train_master['UserInfo_2'], prefix='UserInfo_2')


#modelfit(xgb1, dummies_UserInfo_2, y_train)
# modelfit(xgb1, dummies_UserInfo_2, y_train)
