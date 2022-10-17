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
rcParams['figure.figsize'] = 8, 4
rcParams['font.sans-serif'] = ['SimHei']

# 省份信息处理 UserInfo_19, UserInfo_7
def province_selection(_train_master):
    gt=_train_master.groupby('UserInfo_19')

    return _train_master






if __name__=="__main__":
    train_master = pd.read_csv(r"./data/train/Master_Training_Cleaned_expCity.csv")
    y_train = train_master["target"].values
    train_master = province_selection(train_master)
    train_master.to_csv(r"./data/train/Master_Training_Modified.csv", index=False, sep=',')

