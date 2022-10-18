#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 17:20:38 2022

@author: xzytql
"""

# %% import modules

# import numpy as np
import pandas as pd

# import datetime as dt
# from collections import defaultdict

# %% read original file
df1 = pd.read_csv("data/train/LogInfo_Training_Set.csv")

df1["Listinginfo1"] = pd.to_datetime(df1["Listinginfo1"], format="%Y-%m-%d")
df1["LogInfo3"] = pd.to_datetime(df1["LogInfo3"], format="%Y-%m-%d")
df1["delta_days"] = (df1["Listinginfo1"] - df1["LogInfo3"]).astype('timedelta64[D]').astype(int)

# %% 进一步的数据处理

grp = df1.groupby("Idx")

# FIXME: 这里的处理数据方法是按照Idx区分之后统计登录次数和“值的个数”（去除重复），
# 有什么更好的方法吗？
# <sad>

f1 = lambda x: len(x.unique())  # LogInfo1的特异值数，以下类推
f2 = lambda x: len(x.unique())
f3 = lambda x: len(x.unique())

df2 = grp.agg({"LogInfo1": f1, "LogInfo2": f2, "delta_days": f3})

# %% 保存数据
df2.to_csv('data/train/loginfo_df.csv', index=True, encoding='utf-8')
