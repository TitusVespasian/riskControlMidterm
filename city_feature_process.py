#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 21:22:13 2022

@author: xzytql,zlwtql
"""
import matplotlib.pylab as plt
import pandas as pd
import xgboost as xgb
from matplotlib.pylab import rcParams
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
rcParams['figure.figsize'] = 8, 4
rcParams['font.sans-serif'] = ['SimHei']

def get_featurename(name):
    try:
        rex_str = name[name.rfind('_') + 1:]
    except Exception as err:
        print(err)
        print('Not successful')
        print('name:' + name)
        return name
    return rex_str


def modelfit(fname,alg, dtrain, _y_train, dtest=None, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    cvresult = 0
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain.values[:, :], label=_y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # 建模
    alg.fit(dtrain.values[:, :], _y_train, eval_metric='auc')

    # 对训练集预测
    dtrain_predictions = alg.predict(dtrain.values[:, :])
    dtrain_predprob = alg.predict_proba(dtrain.values[:, :])[:, 1]

    # 输出模型的一些结果
    # print(dtrain_predictions)
    # print(alg.predict_proba(dtrain.as_matrix()[: ,1:]))
    if useTrainCV:
        print(cvresult.shape[0])
    print("\n关于现在这个模型")
    print("准确率 : %.4g" % metrics.accuracy_score(_y_train, dtrain_predictions))
    print("AUC 得分 (训练集): %f" % metrics.roc_auc_score(_y_train, dtrain_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_list = []
    for i in range(len(feat_imp)):
        temp_str = get_featurename(dtrain.columns[int(feat_imp.index[i][1:])])
        if i < 20:
            print(temp_str, feat_imp[i])
        feat_list.append(temp_str)
    feat_imp_new = pd.DataFrame(data=feat_list, columns=['feature_name'], index=feat_imp.index)
    feat_temp = pd.DataFrame(data=feat_imp, columns=['feature_score'])
    feat_imp_new = pd.concat([feat_imp_new, feat_temp], axis=1)
    print(feat_imp_new.shape)
    feat_imp_new.plot(x='feature_name', y='feature_score', kind='bar',
                      title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    # plt.gcf().subplots_adjust(left=0.2)
    plt.savefig('./backup/' + fname + '.png')
    plt.show()
    plt.pause(6)
    plt.close()
    return feat_imp_new


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

# 省份信息处理 UserInfo_19, UserInfo_7
def province_selection(_train_master):

    dummies_19=pd.get_dummies(_train_master.UserInfo_19, prefix="UserInfo_19")
    score_imp=modelfit("UserInfo_19",xgb1, dummies_19, y_train)

    #_train_master["UserInfo_19"]=_train_master["UserInfo_19"].apply(lambda x:
    #                            x if (score_imp[score_imp['feature_score'] > 25]['feature_name']==x).any()
    #                            else "rest")
    # TODO: ajust 25 greater for xgboost lesser for lr
    for iname in score_imp['feature_name']:
        if (score_imp[score_imp['feature_score'] <= 25]['feature_name']==iname).any():
            _train_master.loc[_train_master["UserInfo_19"] == iname,"UserInfo_19"] ='rest'
    _train_master.join(dummies_19)
    _train_master.drop("UserInfo_19", axis=1, inplace=True)

    dummies_7 = pd.get_dummies(_train_master.UserInfo_7, prefix="UserInfo_7")
    modelfit("UserInfo_7",xgb1, dummies_7, y_train)
    # TODO: ajust 46 greater for xgboost lesser for lr
    for iname in score_imp['feature_name']:
        if (score_imp[score_imp['feature_score'] <= 46]['feature_name']==iname).any():
            _train_master.loc[_train_master["UserInfo_7"] == iname,"UserInfo_7"] ='rest'
    _train_master.join(dummies_7, prefix="UserInfo_7")
    _train_master.drop("UserInfo_7", axis=1, inplace=True)

    return _train_master

# 城市信息处理 UserInfo_19, UserInfo_7
def province_selection(_train_master):
    return None

if __name__=="__main__":
    train_master = pd.read_csv(r"./data/train/Master_Training_Cleaned_expCity.csv")
    y_train = train_master["target"].values
    train_master = province_selection(train_master)
    train_master.to_csv(r"./data/train/Master_Training_Modified.csv", index=False, sep=',')

