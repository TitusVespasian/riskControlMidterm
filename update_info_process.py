#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 10:58:56 2022

@author: xzytql, zlwtql
"""

from collections import defaultdict
import datetime as dt
import pandas as pd


# %% 文件处理I
# 在该部分，做一些较为初级的处理
# 目前是将UserupdateInfo1一列全部转为小写字母

train_userupdateinfo = pd.read_csv(
    r"./data/train/Userupdate_Info_Training_Set.csv", encoding="gb18030")

train_userupdateinfo['UserupdateInfo1'] = train_userupdateinfo['UserupdateInfo1'].apply(
    lambda x: x.lower())

train_userupdateinfo.to_csv('data/train/userupdate_Modified.csv',
                            index=False, encoding='utf-8')

# %%  userupdateinfo表

# defaultdict的作用是在于，
# 当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值

userupdate_info_number = defaultdict(list)  # 用户信息更新的次数
userupdate_info_category = defaultdict(set)  # 用户信息更新的种类数
userupdate_info_times = defaultdict(list)  # 用户分几次更新了
userupdate_info_date = defaultdict(list)  # 用户借款成交与信息更新时间跨度

userupdate_info_number_ = defaultdict(int)  # 用户信息更新的次数
userupdate_info_category_ = defaultdict(int)  # 用户信息更新的种类数
userupdate_info_times_ = defaultdict(int)  # 用户分几次更新了
userupdate_info_date_ = defaultdict(int)  # 用户借款成交与信息更新时间跨度


# %% 文件处理II

with open('data/train/userupdate_Modified.csv', 'r') as f:
    f.readline()  # 舍弃第一行
    for line in f.readlines():
        # 这里的每一条line都是一条上传记录
        cols = line.strip().split(",")  # cols 是list结果
        userupdate_info_date[cols[0]].append(cols[1])  # 按照ID对ListingInfo1区分
        userupdate_info_number[cols[0]].append(cols[2])  # 按照ID对于UserupdateInfo1区分，不允许重复
        userupdate_info_category[cols[0]].add(cols[2])  # 同上，但允许重复
        userupdate_info_times[cols[0]].append(cols[3])  # 按照ID对UserupdateInfo2区分
    # print(u'提取信息完成')


# 对于每一个ID，
for key in userupdate_info_date.keys():

    # 上传的总次数
    userupdate_info_times_[key] = len(set(userupdate_info_times[key]))

    # 时间差值
    delta_date = dt.datetime.strptime(userupdate_info_date[key][0], '%Y/%m/%d') \
        - dt.datetime.strptime(list(set(userupdate_info_times[key]))[0], '%Y/%m/%d')

    # 时间差值绝对值化
    userupdate_info_date_[key] = abs(delta_date.days)
    userupdate_info_number_[key] = len(userupdate_info_number[key])
    userupdate_info_category_[key] = len(userupdate_info_category[key])

# 节省内存空间
del userupdate_info_date, userupdate_info_number, userupdate_info_category, userupdate_info_times
# print('信息处理完成')


# %% 建立一个DataFrame

Idx_ = list(userupdate_info_date_.keys())
numbers_ = list(userupdate_info_number_.values())
categorys_ = list(userupdate_info_category_.values())
times_ = list(userupdate_info_times_.values())
dates_ = list(userupdate_info_date_.values())
userupdate_df = pd.DataFrame({'Idx': Idx_, 'numbers': numbers_,
                             'categorys': categorys_, 'times': times_, 'dates': dates_})

userupdate_df = userupdate_df.sort_values("Idx")

userupdate_df.to_csv('data/train/userupdate_df.csv', index=False, encoding='utf-8')
