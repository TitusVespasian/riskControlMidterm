#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 21:22:13 2022

@author: xzytql
"""


# %% 引入模块
import numpy as np
import pandas as pd
import re

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

# %% 对数据进行读入

# 双引号前面的r表示不转义
# train_loginfo = pd.read_csv(r"./data/train/LogInfo_Training_Set.csv", encoding=("gb18030"))
train_master = pd.read_csv(r"./data/train/Master_Training_Set.csv", encoding=("gb18030"))


# %% 去除显著无用列

n_null_rate = train_master.isnull().sum().sort_values(ascending=False)/len(train_master)

train_master.drop(['WeblogInfo_1', 'WeblogInfo_3'], axis=1, inplace=True)

# %% 判断是否需要去除更多的列

# TODO: 解释这一部分的意义？
attrib = "WeblogInfo_20" # 可以更改
df_missing = train_master.target[train_master[attrib].isnull()].value_counts()
df_exist = train_master.target[train_master[attrib].notnull()].value_counts()
# print(df_missing[0], df_missing[1], df_exist[0], df_exist[1])

del df_missing, df_exist

# %% 杂项

# 对于城市数据，统一去除“市”后缀
# 对于通信商数据，统一去除空格
train_master["UserInfo_8"] = train_master["UserInfo_8"].str.replace("市", "")
train_master["UserInfo_9"] = train_master["UserInfo_9"].str.replace(" ", "")
# 放开以检查效果：
# train_master["UserInfo_9"].sort_values().unique()

train_master['Idx'] = train_master['Idx'].astype(np.int32)

def encodingstr(s):
    regex = re.compile(r'.+市')
    if regex.search(s):
        s = s[:-1]
        return s
    else:
        return s


train_master['UserInfo_8'] = train_master['UserInfo_8'].apply(encodingstr)

# %% 去除用处不大的数字列

# feature_std = train_master.std().sort_values(ascending=True)
# feature_std.head(20) # 打印的结果为标准差很小的列，为去除依据
train_master.drop(columns=["WeblogInfo_49", "WeblogInfo_10", "WeblogInfo_44"], axis=1, inplace=True)

# %%
# 用众数填充缺失值
categoric_cols = ['UserInfo_1', 'UserInfo_2', 'UserInfo_3', 'UserInfo_4', 'UserInfo_5',
                  'UserInfo_6', 'UserInfo_7', 'UserInfo_8', 'UserInfo_9', 'UserInfo_11',
                  'UserInfo_12', 'UserInfo_13', 'UserInfo_14', 'UserInfo_15', 'UserInfo_16',
                  'UserInfo_17', 'UserInfo_19', 'UserInfo_20', 'UserInfo_21', 'UserInfo_22',
                  'UserInfo_23', 'UserInfo_24', 'Education_Info1', 'Education_Info2',
                  'Education_Info3', 'Education_Info4', 'Education_Info5', 'Education_Info6',
                  'Education_Info7', 'Education_Info8', 'WeblogInfo_19', 'WeblogInfo_20',
                  'WeblogInfo_21', 'SocialNetwork_1', 'SocialNetwork_2', 'SocialNetwork_7',
                  'SocialNetwork_12']
for col in categoric_cols:
    mode_cols = train_master[col].mode()[0]
    train_master.loc[(train_master[col].isnull(), col)] = mode_cols

# 用均值填充缺失值
# numeric_cols = []
# for col in train_master.columns:
#     if col not in categoric_cols and col != "Idx" and col != "target" and col != 'ListingInfo':
#         mean_cols = train_master[col].mean()
#         train_master.loc[(train_master[col].isnull(), col)] = mean_cols

# 消除PerformanceWarning
train_master=train_master.copy()

# %% 借款成交时间处理

grouped_date_1 = train_master[train_master.target == 1.0]['target'] \
    .groupby(train_master['ListingInfo']).count()
grouped_date_1.sort_values(ascending=False)
grouped_date_0 = train_master[train_master.target == 0.0]['target'] \
    .groupby(train_master['ListingInfo']).count()
grouped_date_0.sort_values(ascending=False)
plt.figure()
plt.title(u'date')
grouped_date_1.plot(color='r')
grouped_date_0.plot(color='b')
plt.show()


# %%  借款日期离散化
# 把月、日单独拎出来放

train_master['month'] = pd.DatetimeIndex(train_master.ListingInfo).month
train_master['day'] = pd.DatetimeIndex(train_master.ListingInfo).day
train_master.drop(['ListingInfo'], axis=1, inplace=True)

train_master['target'] = train_master['target'].astype(str)


# %% 将处理好的数据导出到工作目录

train_master.to_csv(path_or_buf="./data/train/Master_Training_Modified.csv")


