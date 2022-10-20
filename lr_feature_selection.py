#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 21:22:13 2022

@author: xzytql,zlwtql
"""


# %% import
import pandas as pd


# %% 特征分析函数
def get_featurename(name):
    try:
        rex_str = name[name.rfind('_') + 1:]
    except Exception as err:
        print(err)
        print('Not successful')
        print('name:' + name)
        return name
    return rex_str

# %% 开始分析特征
def lr_feature_select(_train_master):
    # 'UserInfo_9'
    print(_train_master['UserInfo_9'].unique())
    # ['中国移动' '中国电信' '不详' '中国联通']
    _train_master = _train_master.join(pd.get_dummies(_train_master.UserInfo_9, prefix="UserInfo_9"))
    _train_master.drop('UserInfo_9', axis=1, inplace=True)
    try:
        print(_train_master.UserInfo_9)
    except AttributeError:
        print("successfully droped!")

    # 'UserInfo_22'
    print(_train_master['UserInfo_22'].value_counts())
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
    _train_master.loc[_train_master['UserInfo_22'] == '初婚', 'UserInfo_22'] = "已婚"
    _train_master = _train_master.join(pd.get_dummies(_train_master.UserInfo_22, prefix="UserInfo_22"))
    _train_master.drop('UserInfo_22', axis=1, inplace=True)
    try:
        print(_train_master.UserInfo_22)
    except AttributeError:
        print("successfully droped!")

    # 'UserInfo_23'
    print(_train_master['UserInfo_23'].value_counts())
    # len=27
    _train_master = _train_master.join(pd.get_dummies(_train_master.UserInfo_23, prefix="UserInfo_23"))
    _train_master.drop('UserInfo_23', axis=1, inplace=True)
    # modelfit(xgb1, dummies_UserInfo_23, y_train)

    # 'UserInfo_24'
    print(_train_master['UserInfo_24'].value_counts())
    # len=1963
    """
    print((train_master['UserInfo_24']=='D').value_counts())
    True     27867
    False     2133
    Name: UserInfo_24, dtype: int64
    """
    #plot_haves(_train_master, 'UserInfo_24', 'D')
    temp = _train_master['UserInfo_24'].copy()
    _train_master['UserInfo_24'] = temp.apply(lambda x: 1 if x == 'D' else 0)

    #rest_col = ['Education_Info2', 'Education_Info3', 'Education_Info4', 'Education_Info6', 'Education_Info7',
    #            'Education_Info8', 'WeblogInfo_19', 'WeblogInfo_20', 'WeblogInfo_21']

    # 'Education_Info2'
    #plot_haves(_train_master, 'Education_Info2', 'E')
    temp = _train_master['Education_Info2'].copy()
    _train_master['Education_Info2'] = temp.apply(lambda x: 1 if x == 'E' else 0)

    # 'Education_Info3'
    print(_train_master['Education_Info3'].value_counts())
    # len=3
    _train_master = _train_master.join(pd.get_dummies(_train_master.Education_Info3, prefix="Education_Info3"))
    _train_master.drop('Education_Info3', axis=1, inplace=True)

    # 'Education_Info4'
    print(_train_master['Education_Info4'].value_counts())
    # len=6
    _train_master = _train_master.join(pd.get_dummies(_train_master.Education_Info4, prefix="Education_Info4"))
    _train_master.drop('Education_Info4', axis=1, inplace=True)

    # 'Education_Info6'
    print(_train_master['Education_Info6'].value_counts())
    # len=6
    _train_master = _train_master.join(pd.get_dummies(_train_master.Education_Info6, prefix="Education_Info6"))
    _train_master.drop('Education_Info6', axis=1, inplace=True)

    # 'Education_Info7'
    print(_train_master['Education_Info7'].value_counts())
    temp = _train_master['Education_Info7'].copy()
    _train_master['Education_Info7'] = temp.apply(lambda x: 1 if x == 'E' else 0)

    # 'Education_Info8'
    print(_train_master['Education_Info8'].value_counts())
    # len=7
    _train_master = _train_master.join(pd.get_dummies(_train_master.Education_Info8, prefix="Education_Info8"))
    _train_master.drop('Education_Info8', axis=1, inplace=True)

    # 'WeblogInfo_19'
    print(_train_master['WeblogInfo_19'].value_counts())
    # len=7
    _train_master = _train_master.join(pd.get_dummies(_train_master.WeblogInfo_19, prefix="WeblogInfo_19"))
    _train_master.drop('WeblogInfo_19', axis=1, inplace=True)

    # WeblogInfo_20
    # len=20
    # modelfit(xgb1, dummies_WeblogInfo_20, y_train)
    _train_master = _train_master.join(pd.get_dummies(_train_master.WeblogInfo_20, prefix="WeblogInfo_20"))
    _train_master.drop('WeblogInfo_20', axis=1, inplace=True)

    # WeblogInfo_21
    _train_master = _train_master.join(pd.get_dummies(_train_master.WeblogInfo_21, prefix="WeblogInfo_21"))
    _train_master.drop('WeblogInfo_21', axis=1, inplace=True)

    return _train_master


# modelfit(xgb1, dummies_UserInfo_2, y_train)
if __name__ == "__main__":
    train=False
    if train:
        train_master = pd.read_csv(r"./data/train/Master_Training_Cleaned.csv")
        y_train = train_master["target"].values
        train_master = lr_feature_select(train_master)
        train_master.to_csv(r"./data/train/Master_Training_Cleaned_expCity.csv", index=False, sep=',')
    else:
        all_master = pd.read_csv(r"./data/all/all_set.csv")
        all_master = lr_feature_select(all_master)
        all_master.to_csv(r"./data/all/all_Cleaned_expCity.csv", index=False, sep=',')
