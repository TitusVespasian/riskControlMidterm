#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 21:22:13 2022

@author: xzytql,zlwtql
"""

import matplotlib.pylab as plt
# %% import
import pandas as pd
import xgboost as xgb
from matplotlib.pylab import rcParams
from sklearn import metrics
from xgboost.sklearn import XGBClassifier

rcParams['figure.figsize'] = 8, 4
# from matplotlib.font_manager import FontProperties
# myfont = FontProperties(fname=r"external-libraries/yahei.ttf",size=12)
rcParams['font.sans-serif'] = ['SimHei']


# %% 特征分析函数
def getFeaturename(name):
    try:
        rex_str = name[name.rfind('_') + 1:]
    except:
        print('Not successful')
        print('name:' + name)
        return name
    return rex_str

def plot_haves(train_master,feature_name,have_char):
    target_have = train_master.target[train_master[feature_name] == have_char]\
        .value_counts()
    target_not_have = train_master.target[train_master[feature_name] != have_char]\
        .value_counts()
    df_WeblogInfo_20 = pd.DataFrame({'no_have': target_have, 'have': target_not_have})
    df_WeblogInfo_20.plot(kind='bar', stacked=True)
    plt.title(u'有无'+feature_name+'对结果的影响')
    plt.xlabel(u'有无')
    plt.ylabel(u'违约情况')
    plt.savefig('./backup/'+feature_name+'_hnh.jpg')
    plt.show()
    plt.pause(6)  # 间隔的秒数：6s
    plt.close()

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
    if useTrainCV:
        print(cvresult.shape[0])
    print("\n关于现在这个模型")
    print("准确率 : %.4g" % metrics.accuracy_score(y_train, dtrain_predictions))
    print("AUC 得分 (训练集): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_list = []
    for i in range(min(len(feat_imp), 20)):
        temp_str = getFeaturename(dtrain.columns[int(feat_imp.index[i][1:])])
        print(temp_str, feat_imp[i])
        feat_list.append(temp_str)
    feat_imp_new = pd.DataFrame(data=feat_list, columns=['feature_name'], index=feat_imp.index)
    feat_temp = pd.DataFrame(data=feat_imp, columns=['feature_score'])
    feat_imp_new = pd.concat([feat_imp_new, feat_temp], axis=1)
    print(feat_imp.shape)
    feat_imp_new.plot(x='feature_name', y='feature_score', kind='barh',
                      title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    # plt.gcf().subplots_adjust(left=0.2)
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
train_master = train_master.join(pd.get_dummies(train_master.UserInfo_9, prefix="UserInfo_9"))
train_master.drop('UserInfo_9', axis=1, inplace=True)
try:
    print(train_master.UserInfo_9)
except AttributeError:
    print("successfully droped!")

# 'UserInfo_22'
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
train_master.loc[train_master['UserInfo_22'] == '初婚', 'UserInfo_22'] = "已婚"
train_master = train_master.join(pd.get_dummies(train_master.UserInfo_22, prefix="UserInfo_22"))
train_master.drop('UserInfo_22', axis=1, inplace=True)
try:
    print(train_master.UserInfo_22)
except AttributeError:
    print("successfully droped!")

# 'UserInfo_23'
print(train_master['UserInfo_23'].value_counts())
# len=27
dummies_UserInfo_23 = pd.get_dummies(train_master['UserInfo_23'], prefix='UserInfo_23')
train_master = train_master.join(pd.get_dummies(train_master.UserInfo_23, prefix="UserInfo_23"))
train_master.drop('UserInfo_23', axis=1, inplace=True)
# modelfit(xgb1, dummies_UserInfo_23, y_train)

# 'UserInfo_24'
print(train_master['UserInfo_24'].value_counts())
# len=1963
"""
print((train_master['UserInfo_24']=='D').value_counts())
True     27867
False     2133
Name: UserInfo_24, dtype: int64
"""
plot_haves(train_master,'UserInfo_24','D')
temp=train_master['UserInfo_24'].copy()
train_master['UserInfo_24']=temp.apply(lambda x : 1 if x=='D' else 0)

rest_col=['Education_Info2','Education_Info3','Education_Info4','Education_Info6','Education_Info7',
          'Education_Info8','WeblogInfo_19','WeblogInfo_20','WeblogInfo_21']
for i in rest_col:
    print(train_master[rest_col].value_counts())




dummies_UserInfo_2 = pd.get_dummies(train_master['UserInfo_2'], prefix='UserInfo_2')

# modelfit(xgb1, dummies_UserInfo_2, y_train)
