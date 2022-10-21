import re

import lightgbm as lgb
import pandas as pd
from sklearn import preprocessing

import city_and_province_process
import merge_data

best_pa = {'n_estimators': 1000, 'max_depth': 7, 'num_leaves': 15, 'max_bin': 65, 'min_data_in_leaf': 11,
           'bagging_fraction': 0.6, 'bagging_freq': 0, 'feature_fraction': 0.8, 'lambda_l1': 0.001, 'lambda_l2': 0.001,
           'min_split_gain': 0.0}

model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.005,
                           max_depth=best_pa['max_depth'],
                           num_leaves=best_pa['num_leaves'], max_bin=best_pa['max_bin'],
                           min_data_in_leaf=best_pa['min_data_in_leaf'],
                           bagging_fraction=best_pa['bagging_fraction'], bagging_freq=best_pa['bagging_freq'],
                           feature_fraction=best_pa['feature_fraction'],
                           lambda_l1=best_pa['lambda_l1'], lambda_l2=best_pa['lambda_l2'],
                           min_split_gain=best_pa['min_split_gain'],
                           n_estimators=1000, is_unbalance=True)


def temp():
    train_master = pd.read_csv('./data/train/Master_Training_Cleaned.csv')
    train_master = city_and_province_process.province_selection(train_master)
    train_master = city_and_province_process.city_process(train_master)
    update_info_pd = pd.read_csv('data/train/userupdate_df.csv')
    log_info_pd = pd.read_csv('data/train/loginfo_df.csv')

    test_master = pd.read_csv('./data/test/Master_Test_Cleaned.csv')
    test_master = city_and_province_process.province_selection(test_master)
    test_master = city_and_province_process.city_process(test_master)
    test_update_info_pd = pd.read_csv('data/test/test_userupdate_df.csv')
    test_log_info_pd = pd.read_csv('./data/test/test_loginfo_df.csv')

    train_all = merge_data.merge_data(train_master, update_info_pd, log_info_pd)
    test_all = merge_data.merge_data(test_master, test_update_info_pd, test_log_info_pd, if_all=True)

    # 拼接，同时处理特征
    X = train_all.append(test_all)

    # X = test_all.drop(['target'], axis=1)  # .to_numpy()
    # y = train_master.target  # .to_numpy()
    # new_dict = {key:i for (i,key) in enumerate(X.columns)}
    categorical_ordered = ['UserInfo_1', 'UserInfo_3', 'UserInfo_5', 'UserInfo_6', 'UserInfo_14', 'UserInfo_15',
                           'UserInfo_16', 'SocialNetwork_1', 'SocialNetwork_2', 'SocialNetwork_7', 'SocialNetwork_12']
    categorical_cols = ['UserInfo_2_level', 'UserInfo_4_level', 'UserInfo_6',
                        # 'UserInfo_7',
                        'UserInfo_8_level',
                        'UserInfo_9',
                        'UserInfo_11', 'UserInfo_12', 'UserInfo_13', 'UserInfo_17',
                        # 'UserInfo_19',
                        'UserInfo_20_level',
                        'UserInfo_21', 'UserInfo_22', 'UserInfo_23', 'UserInfo_24', 'Education_Info1',
                        'Education_Info2',
                        'Education_Info3', 'Education_Info4', 'Education_Info5', 'Education_Info6', 'Education_Info7',
                        'Education_Info8',
                        'WeblogInfo_19', 'WeblogInfo_20', 'WeblogInfo_21', 'city_sim']

    for icol in categorical_ordered:
        X[icol].astype('category').cat.as_ordered()

    for icol in categorical_cols:
        X[icol].astype('category')

    new_dict = {key: re.sub('[^A-Za-z0-9]+', '', key) + str(i) if key != "target" else "target" for (i, key) in
                enumerate(X.columns)}
    X.rename(columns=new_dict, inplace=True)

    list_fea_str = [
        'UserInfo920', 'UserInfo2231', 'UserInfo2332', 'UserInfo2433', 'EducationInfo235', 'EducationInfo336',
        'EducationInfo437', 'EducationInfo639', 'EducationInfo740', 'EducationInfo841', 'WeblogInfo1942',
        'WeblogInfo2043', 'WeblogInfo2144']
    dict_fea_encode = {}
    for icol in list_fea_str:
        dict_fea_encode[icol] = preprocessing.LabelEncoder()
        X[icol] = dict_fea_encode[icol].fit_transform(X[icol].astype(str))  # 将提示的包含错误数据类型这一列进行转换

    # 分开
    train_all = X[X["target"].notnull()]
    test_all = X[X["target"].isnull()]

    ID = test_all['Idx0'].rename('Idx')
    train_all = train_all.drop(['Idx0'], axis=1)
    test_all = test_all.drop(['Idx0'], axis=1)

    model.fit(train_all.drop(['target'], axis=1), train_all['target'])
    import joblib
    joblib.dump(model, 'dota_model.pkl')
    clf = joblib.load('dota_model.pkl')

    test_res = clf.predict_proba(test_all.drop(['target'],axis=1))[:, 1]
    df = pd.DataFrame(data={"target": test_res})
    df = df.join(ID)
    df.to_csv("result.csv", index=False)
    print(df.shape)


if __name__ == "__main__":
    Ytemp = temp()
