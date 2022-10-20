#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 17:20:38 2022

@author: xzytql
This File can be used by all models
"""

# %% import modules

# import numpy as np
import pandas as pd

# import datetime as dt
# from collections import defaultdict

"""
- 1.ListingInfo：借款成交时间
- 2.LogInfo1：操作代码
- 3.LogInfo2：操作类别
- 4.LogInfo3：登陆时间
- 5.idx：每一笔借款的unique key
"""


def log_info_process(input_file_path, output_file_path):
    # %% read original file
    df1 = pd.read_csv(input_file_path)

    df1["Listinginfo1"] = pd.to_datetime(df1["Listinginfo1"], format="%Y-%m-%d")
    df1["LogInfo3"] = pd.to_datetime(df1["LogInfo3"], format="%Y-%m-%d")
    # df1["delta_days"] = (df1["Listinginfo1"] - df1["LogInfo3"]).astype('timedelta64[D]').astype(int)
    # %% 进一步的数据处理

    grp = df1.groupby("Idx", as_index=False)

    # FIXME: 这里的处理数据方法是按照Idx区分之后统计登录次数和“值的个数”（去除重复），
    # 有什么更好的方法吗？
    # <sad>
    def f1(x):
        return len(x.unique())

    df2 = grp.agg({"LogInfo1": f1, "LogInfo2": f1}).rename({"LogInfo1": "kind_of_L1", "LogInfo2": "kind_of_L2"}, axis=1)
    df2 = pd.merge(df2, grp.agg({"LogInfo1": 'count'}).rename({"LogInfo1": "num_of_logins"}, axis=1))
    df2 = pd.merge(df2, grp.agg({"Listinginfo1": 'min', "LogInfo3": 'min'}).rename(
        {"Listinginfo1": "earliest_trans", "LogInfo3": "earliest_log"}, axis=1))
    df2 = pd.merge(df2, grp.agg({"LogInfo3": 'max'}).rename({"LogInfo3": "latest_log"}, axis=1))
    df2['between_early'] = abs(df2['earliest_trans'] - df2['earliest_log']).astype('timedelta64[D]').astype(int)
    df2['between_late'] = abs(df2['latest_log'] - df2['earliest_trans']).astype('timedelta64[D]').astype(int)
    df2 = df2.drop(['earliest_trans', 'earliest_log', 'latest_log'], axis=1)

    # %% 保存数据
    df2.to_csv(output_file_path, index=False, encoding='utf-8')
    return df2


if __name__ == "__main__":
    log_info_process(r"./data/train/LogInfo_Training_Set.csv", r"./data/train/loginfo_df.csv")
    log_info_process(r"./data/test/LogInfo_Test_Set.csv", r"./data/test/test_loginfo_df.csv")