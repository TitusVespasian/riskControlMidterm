#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 22:28:14 2022

Description:这一部分对应于原demo的“建模调参与优化”一节的前半部分（XGBoost之前）。
主要用途是将三张修改过后的csv文件恰当合并。

在本文件的修改，请使用注释加以标记，写报告好写

@author: xzyxzstql
"""


# %% import modules


import pandas as pd
import numpy as np

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4 # 绘图相关设置

# demo使用了下面的两个语句以抑制Warnings。我把它们放开了。
# import warnings
# warnings.filterwarnings("ignore")

# %% load_data

train_master = pd.read_csv('data/train/Master_Training_Modified.csv', encoding='utf-8')
train_userupdateinfo = pd.read_csv('data/train/userupdate_df.csv', encoding='utf-8')
train_loginfo = pd.read_csv('data/train/loginfo_df.csv', encoding='utf-8')

# %% merge

# 将上述三张表的信息进行合并

train_all = pd.merge(train_master, train_userupdateinfo, how='left', on='Idx')
train_all = pd.merge(train_all, train_loginfo, how='left', on='Idx')
train_all.isnull().sum().sort_values(ascending=False).head(10)

"""
kind_of_L1,kind_of_L2,num_of_logins,between_early,between_late    1013
dates, times, categorys, numbers     5
"""
# 现在对于缺少的值进行填充
# 填充方法目前为0 0 0 0 0/0填充
missing_list_0=["dates","times","categorys","numbers","kind_of_L1","kind_of_L2","num_of_logins"]
missing_list_mean=['between_early','between_late']
# missing_list=list(train_all.columns[train_all.isnull().sum() > 0])

for column in missing_list_0:
    # mean_val = train_all[column].mean()
    train_all[column].fillna(0, inplace=True)

for column in missing_list_mean:
    mean_val = train_all[column].mean()
    train_all[column].fillna(mean_val, inplace=True)

# 至此，缺失的项应当全部被填充。

#%% 将train_all全部数值化，最后输出

train_all['Idx'] = train_all['Idx'].astype(np.int64)
train_all['target'] = train_all['target'].astype(np.int64)

train_all = pd.get_dummies(train_all)
# train_all.head()
train_all.to_csv('data/train/train_all.csv', encoding='utf-8', index=False)
y_train = train_all.pop('target')

