#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 21:22:13 2022

@author: xzytql,zlwtql

This File can be used by all models
"""
import re

# 引入模块
import numpy as np
import pandas as pd


# 对于城市数据，统一去除“市或省”后缀
def encodingstr(s, appendix):
    regex = re.compile(r'.+' + appendix)
    if regex.search(s):
        s = s[:-1]
        return s
    else:
        return s


def clean_and_process_master(input_file_path, output_file_path,if_test=False):
    # %% 对数据进行读入

    # 双引号前面的r表示不转义
    # train_loginfo = pd.read_csv(r"./data/train/LogInfo_Training_Set.csv", encoding=("gb18030"))
    # train_master = pd.read_csv(r"./data/train/Master_Training_Set.csv", encoding="gb18030")
    train_master = pd.read_csv(input_file_path, encoding="gb18030")
    # 加载微软雅黑中文字体
    # 去掉缺失比例接近百分之百的字段
    train_master.drop(['WeblogInfo_1', 'WeblogInfo_3'], axis=1, inplace=True)

    # 处理UserInfo_12缺失: res->需要有/无分类
    train_master.loc[(train_master.UserInfo_12.isnull(), 'UserInfo_12')] = 2.0

    # 处理UserInfo_11缺失: res->需要有/无分类
    train_master.loc[(train_master.UserInfo_11.isnull(), 'UserInfo_11')] = 2.0

    # 处理UserInfo_13缺失: res->需要有/无分类
    train_master.loc[(train_master.UserInfo_13.isnull(), 'UserInfo_13')] = 2.0

    # 处理WeblogInfo_20缺失: res->需要有/无分类
    train_master.loc[(train_master.WeblogInfo_20.isnull(), 'WeblogInfo_20')] = u'不详'

    # 处理WeblogInfo_19缺失: res->需要有/无分类
    train_master.loc[(train_master.WeblogInfo_19.isnull(), 'WeblogInfo_19')] = u'不详'

    # 处理WeblogInfo_21缺失: res->需要有/无分类
    train_master.loc[(train_master.WeblogInfo_21.isnull(), 'WeblogInfo_21')] = '0'

    # 数据预处理和特征工程 TODO:可以删或不删观察结果
    # 如果选择以0填充，下述部分就维持现状，如果选择中位数/众数填充，就把下述的部分注释掉
    train_master.loc[(train_master.UserInfo_2.isnull(), 'UserInfo_2')] = '0'
    train_master.loc[(train_master.UserInfo_4.isnull(), 'UserInfo_4')] = '0'
    train_master.loc[(train_master.UserInfo_8.isnull(), 'UserInfo_8')] = '0'
    train_master.loc[(train_master.UserInfo_9.isnull(), 'UserInfo_9')] = '0'
    train_master.loc[(train_master.UserInfo_20.isnull(), 'UserInfo_20')] = '0'
    train_master.loc[(train_master.UserInfo_7.isnull(), 'UserInfo_7')] = '0'
    train_master.loc[(train_master.UserInfo_19.isnull(), 'UserInfo_19')] = '0'

    # 用众数填充缺失值
    categoric_cols = ['UserInfo_1', 'UserInfo_2', 'UserInfo_3', 'UserInfo_4', 'UserInfo_5', 'UserInfo_6', 'UserInfo_7',
                      'UserInfo_8', 'UserInfo_9', 'UserInfo_11', 'UserInfo_12', 'UserInfo_13', 'UserInfo_14',
                      'UserInfo_15',
                      'UserInfo_16', 'UserInfo_17', 'UserInfo_19', 'UserInfo_20', 'UserInfo_21', 'UserInfo_22',
                      'UserInfo_23', 'UserInfo_24', 'Education_Info1', 'Education_Info2', 'Education_Info3',
                      'Education_Info4', 'Education_Info5', 'Education_Info6', 'Education_Info7', 'Education_Info8',
                      'WeblogInfo_19', 'WeblogInfo_20', 'WeblogInfo_21', 'SocialNetwork_1', 'SocialNetwork_2',
                      'SocialNetwork_7', 'SocialNetwork_12']
    for col in categoric_cols:
        mode_cols = train_master[col].mode()[0]
        train_master.loc[(train_master[col].isnull(), col)] = mode_cols

    # 用均值填充缺失值
    # numeric_cols = []
    for col in train_master.columns:
        if col not in categoric_cols and col != u'Idx' and col != u'target' and col != 'ListingInfo':
            mean_cols = train_master[col].mean()
            train_master.loc[(train_master[col].isnull(), col)] = mean_cols

    #y_train = train_master['target'].values

    # 剔除标准差几乎为零的特征项 TODO:删除了几乎为0的两个 DONE
    train_master.drop(['WeblogInfo_10','WeblogInfo_49'], axis=1, inplace=True)

    # numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] 0.01*mean
    # new_df = train_master.select_dtypes(include=numerics).loc[:, train_master.quantile(0.25) < train_master.std()]
    # train_master = pd.concat([new_df, train_master.select_dtypes(exclude=numerics)], axis=1)

    # feature_std = train_master.std().sort_values(ascending=True)
    # %% 杂项
    # 对于文本，统一去除空格
    # TODO: ipynb 改动 多加了几列 DONE
    for col in ['UserInfo_2', 'UserInfo_4', 'UserInfo_7', 'UserInfo_8', 'UserInfo_9', 'UserInfo_19', 'UserInfo_20',
                'UserInfo_22'
        , 'UserInfo_24', 'UserInfo_20', 'Education_Info2', 'Education_Info3', 'Education_Info4', 'Education_Info6'
        , 'Education_Info7', 'Education_Info8', 'WeblogInfo_19', 'WeblogInfo_20', 'WeblogInfo_21']:
        train_master[col] = train_master[col].apply(lambda x: x.strip())
    # 放开以检查效果：
    # train_master["UserInfo_9"].sort_values().unique()

    train_master['Idx'] = train_master['Idx'].astype(np.int32)

    # TODO: ipynb 改动 多加了几列包括删去‘省’和‘市’ 'UserInfo_7','UserInfo_19'因为是直辖市 DONE
    for col in ['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20', 'UserInfo_24', 'UserInfo_7', 'UserInfo_19']:
        train_master[col] = train_master[col].apply(lambda x: encodingstr(x, '市'))
    for col in ['UserInfo_7', 'UserInfo_19']:
        train_master[col] = train_master[col].apply(lambda x: encodingstr(x, '省'))

    # TODO: 本文档还未包含one-hot 编码化 这里和ipynb不一致，特征工程文件中包含了 DONE

    # %% 借款成交时间处理

    # notes:说明target=1违约的就比较稳定

    # %%  借款日期离散化
    # 把月、日单独拎出来放

    train_master['month'] = pd.DatetimeIndex(train_master.ListingInfo).month
    train_master['day'] = pd.DatetimeIndex(train_master.ListingInfo).day
    train_master.drop(['ListingInfo'], axis=1, inplace=True)

    if if_test==False:
        train_master['target'] = train_master['target'].astype(str)

    # %% 将处理好的数据导出到工作目录

    # train_master.to_csv(path_or_buf="./data/train/Master_Training_Cleaned.csv", index=False)
    train_master.to_csv(path_or_buf=output_file_path, index=False)


if __name__ == "__main__":
    # clean_and_process_master(r"./data/train/Master_Training_Set.csv", r"./data/train/Master_Training_Cleaned.csv")
    clean_and_process_master(r"./data/test/Master_Test_Set.csv",r"./data/test/Master_Test_Cleaned.csv",True)