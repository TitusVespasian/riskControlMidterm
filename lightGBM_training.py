import re

import lightgbm as lgb
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
# from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import city_and_province_process
import merge_data

train_master = pd.read_csv('./data/train/Master_Training_Cleaned.csv')
update_info_pd = pd.read_csv('data/train/userupdate_df.csv')
log_info_pd = pd.read_csv('data/train/loginfo_df.csv')
train_master = city_and_province_process.city_process(train_master)

train_all = merge_data.merge_data(train_master, update_info_pd, log_info_pd)

X = train_master.drop(['target'], axis=1)  # .to_numpy()
y = train_master.target  # .to_numpy()
# new_dict = {key:i for (i,key) in enumerate(X.columns)}
categorical_ordered = ['UserInfo_1', 'UserInfo_3', 'UserInfo_5', 'UserInfo_6', 'UserInfo_14', 'UserInfo_15',
                       'UserInfo_16', 'SocialNetwork_1', 'SocialNetwork_2', 'SocialNetwork_7', 'SocialNetwork_12']
categorical_cols = ['UserInfo_2_level', 'UserInfo_4_level', 'UserInfo_6', 'UserInfo_7', 'UserInfo_8_level',
                    'UserInfo_9',
                    'UserInfo_11', 'UserInfo_12', 'UserInfo_13', 'UserInfo_17', 'UserInfo_19', 'UserInfo_20_level',
                    'UserInfo_21', 'UserInfo_22', 'UserInfo_23', 'UserInfo_24', 'Education_Info1', 'Education_Info2',
                    'Education_Info3', 'Education_Info4', 'Education_Info5', 'Education_Info6', 'Education_Info7',
                    'Education_Info8',
                    'WeblogInfo_19', 'WeblogInfo_20', 'WeblogInfo_21', 'city_sim']

for icol in categorical_ordered:
    X[icol].astype('category').cat.as_ordered()

for icol in categorical_cols:
    X[icol].astype('category')

new_dict = {key: re.sub('[^A-Za-z0-9]+', '', key) + str(i) for (i, key) in enumerate(X.columns)}
X.rename(columns=new_dict, inplace=True)

list_fea_str = ['UserInfo720', 'UserInfo921', 'UserInfo1931', 'UserInfo2233', 'UserInfo2334', 'UserInfo2435',
                'EducationInfo237', 'EducationInfo338', 'EducationInfo439', 'EducationInfo641', 'EducationInfo742',
                'EducationInfo843', 'WeblogInfo1944', 'WeblogInfo2045', 'WeblogInfo2146']
dict_fea_encode = {}
for icol in list_fea_str:
    dict_fea_encode[icol] = preprocessing.LabelEncoder()
    X[icol] = dict_fea_encode[icol].fit_transform(X[icol].astype(str))  # 将提示的包含错误数据类型这一列进行转换

X = X.drop(['Idx0'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, test_size=0.2, stratify=y)
# params = {
#     # 'n_estimators':56,
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': 'auc',
#     'nthread': 4,
#     'learning_rate': 0.1,
#     'num_leaves': 30,
#     'max_depth': 5,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'is_unbalance': 'true'
# }
#
# # %% 第一步：学习率和迭代次数
# data_train = lgb.Dataset(X_train, y_train)
# cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',
#                     early_stopping_rounds=50, seed=0)
# print('best n_estimators:', len(cv_results['auc-mean']))
# print('best cv score:', pd.Series(cv_results['auc-mean']).max())
# # best n_estimators: 73
# # best cv score: 0.729953228991693
# best_pa = {'n_estimators': len(cv_results['auc-mean'])}
#
# # %% 确定max_depth和num_leaves
# params_test1 = {'max_depth': range(3, 8, 1), 'num_leaves': range(5, 100, 5)}
#
# gsearch1 = GridSearchCV(
#     estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
#                                  n_estimators=best_pa['n_estimators'], max_depth=6, bagging_fraction=0.8,
#                                  feature_fraction=0.8,
#                                  is_unbalance=True),
#     param_grid=params_test1, scoring='roc_auc', cv=5, n_jobs=-1)
# gsearch1.fit(X_train, y_train)
# print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
# # {'max_depth': 5, 'num_leaves': 10} 0.7334410002865965
# best_pa.update(gsearch1.best_params_.items())
#
# #
# # %% 第三步：确定min_data_in_leaf和max_bin in
# params_test2 = {'max_bin': range(5, 256, 10), 'min_data_in_leaf': range(1, 102, 10)}
#
# gsearch2 = GridSearchCV(
#     estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
#                                  n_estimators=best_pa['n_estimators'], max_depth=best_pa['max_depth'],
#                                  num_leaves=best_pa['num_leaves'], bagging_fraction=0.8,
#                                  feature_fraction=0.8, is_unbalance=True),
#     param_grid=params_test2, scoring='roc_auc', cv=5, n_jobs=-1)
# gsearch2.fit(X_train, y_train)
# print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_)
# best_pa.update(gsearch2.best_params_)
# # {'max_bin': 75, 'min_data_in_leaf': 101} 0.7401419009586869
#
# # %% 第四步：确定feature_fraction、bagging_fraction、bagging_freq
# params_test3 = {'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
#                 'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
#                 'bagging_freq': range(0, 81, 10)
#                 }
#
# gsearch3 = GridSearchCV(
#     estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
#                                  n_estimators=best_pa['n_estimators'], max_depth=best_pa['max_depth'],
#                                  num_leaves=best_pa['num_leaves'], max_bin=best_pa['max_bin'],
#                                  min_data_in_leaf=best_pa['min_data_in_leaf'],
#                                  is_unbalance=True),
#     param_grid=params_test3, scoring='roc_auc', cv=5, n_jobs=-1)
# gsearch3.fit(X_train, y_train)
# print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)
# # {'bagging_fraction': 0.6, 'bagging_freq': 0, 'feature_fraction': 0.8} 0.7401419009586869
# best_pa.update(gsearch3.best_params_)
#
# # # %% 第五步：确定lambda_l1和lambda_l2
# params_test4 = {'lambda_l1': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
#                 'lambda_l2': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
#                 }
#
# gsearch4 = GridSearchCV(
#     estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
#                                  n_estimators=best_pa['n_estimators'], max_depth=best_pa['max_depth'],
#                                  num_leaves=best_pa['num_leaves'], max_bin=best_pa['max_bin'],
#                                  min_data_in_leaf=best_pa['min_data_in_leaf'],
#                                  bagging_fraction=best_pa['bagging_fraction'], bagging_freq=best_pa['bagging_freq'],
#                                  feature_fraction=best_pa['feature_fraction'], is_unbalance=True),
#     param_grid=params_test4, scoring='roc_auc', cv=5, n_jobs=-1)
# gsearch4.fit(X_train, y_train)
# print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)
# best_pa.update(gsearch4.best_params_)
# # {'lambda_l1': 1e-05, 'lambda_l2': 0.001} 0.7401421591287665
# #
# # %% 第六步：确定 min_split_gain
# params_test5 = {'min_split_gain': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
#
# gsearch5 = GridSearchCV(
#     estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
#                                  n_estimators=best_pa['n_estimators'], max_depth=best_pa['max_depth'],
#                                  num_leaves=best_pa['num_leaves'], max_bin=best_pa['max_bin'],
#                                  min_data_in_leaf=best_pa['min_data_in_leaf'],
#                                  bagging_fraction=best_pa['bagging_fraction'], bagging_freq=best_pa['bagging_freq'],
#                                  feature_fraction=best_pa['feature_fraction'],
#                                  lambda_l1=best_pa['lambda_l1'], lambda_l2=best_pa['lambda_l2'], is_unbalance=True),
#     param_grid=params_test5, scoring='roc_auc', cv=5, n_jobs=-1)
# gsearch5.fit(X_train, y_train)
# print(gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_)
# best_pa.update(gsearch5.best_params_)
# # {'min_split_gain': 0.2} 0.7403077983762778
#
#
# print(best_pa)
best_pa= {'n_estimators': 73, 'max_depth': 7, 'num_leaves': 15, 'max_bin': 65, 'min_data_in_leaf': 11, 'bagging_fraction': 0.6, 'bagging_freq': 0, 'feature_fraction': 0.8, 'lambda_l1': 0.001, 'lambda_l2': 0.001, 'min_split_gain': 0.0}
# acc: 0.7623333333333333
# auc: 0.6814672478483803
# %% 第七步：降低学习率，增加迭代次数，验证模型
from sklearn import metrics

model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.01,
                           n_estimators=1000, max_depth=best_pa['max_depth'],
                           num_leaves=best_pa['num_leaves'], max_bin=best_pa['max_bin'],
                           min_data_in_leaf=best_pa['min_data_in_leaf'],
                           bagging_fraction=best_pa['bagging_fraction'], bagging_freq=best_pa['bagging_freq'],
                           feature_fraction=best_pa['feature_fraction'],
                           lambda_l1=best_pa['lambda_l1'], lambda_l2=best_pa['lambda_l2'],
                           min_split_gain=best_pa['min_split_gain'], is_unbalance=True)
model.fit(X_train, y_train)
import joblib
joblib.dump(model, 'dota_model.pkl')
clf = joblib.load('dota_model.pkl')
#
# # 模型预测
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
y_pre = clf.predict_proba(X_test)
print(y_pre)
y_pre=clf.predict(X_test)
print(y_pre)
print("acc:", metrics.accuracy_score(y_test, y_pre))
print("auc:", metrics.roc_auc_score(y_test, y_pre))

## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(y_test,y_pre)
print('The confusion matrix result:\n',confusion_matrix_result)

def temp():
    train_master = pd.read_csv('./data/train/Master_Training_Cleaned.csv')
    update_info_pd = pd.read_csv('data/train/userupdate_df.csv')
    log_info_pd = pd.read_csv('data/train/loginfo_df.csv')

    test_master=pd.read_csv('./data/train/Master_Training_Cleaned.csv')
    test_master=city_and_province_process.city_process(test_master)
    test_update_info_pd = pd.read_csv('data/train/userupdate_df.csv')

    train_master = city_and_province_process.city_process(train_master)

    train_all = merge_data.merge_data(train_master, update_info_pd, log_info_pd)

    X = train_master.drop(['target'], axis=1)  # .to_numpy()
    y = train_master.target  # .to_numpy()
    # new_dict = {key:i for (i,key) in enumerate(X.columns)}
    categorical_ordered = ['UserInfo_1', 'UserInfo_3', 'UserInfo_5', 'UserInfo_6', 'UserInfo_14', 'UserInfo_15',
                           'UserInfo_16', 'SocialNetwork_1', 'SocialNetwork_2', 'SocialNetwork_7', 'SocialNetwork_12']
    categorical_cols = ['UserInfo_2_level', 'UserInfo_4_level', 'UserInfo_6', 'UserInfo_7', 'UserInfo_8_level',
                        'UserInfo_9',
                        'UserInfo_11', 'UserInfo_12', 'UserInfo_13', 'UserInfo_17', 'UserInfo_19', 'UserInfo_20_level',
                        'UserInfo_21', 'UserInfo_22', 'UserInfo_23', 'UserInfo_24', 'Education_Info1',
                        'Education_Info2',
                        'Education_Info3', 'Education_Info4', 'Education_Info5', 'Education_Info6', 'Education_Info7',
                        'Education_Info8',
                        'WeblogInfo_19', 'WeblogInfo_20', 'WeblogInfo_21', 'city_sim']

    for icol in categorical_ordered:
        X[icol].astype('category').cat.as_ordered()

    for icol in categorical_cols:
        X[icol].astype('category')

    new_dict = {key: re.sub('[^A-Za-z0-9]+', '', key) + str(i) for (i, key) in enumerate(X.columns)}
    X.rename(columns=new_dict, inplace=True)

    list_fea_str = ['UserInfo720', 'UserInfo921', 'UserInfo1931', 'UserInfo2233', 'UserInfo2334', 'UserInfo2435',
                    'EducationInfo237', 'EducationInfo338', 'EducationInfo439', 'EducationInfo641', 'EducationInfo742',
                    'EducationInfo843', 'WeblogInfo1944', 'WeblogInfo2045', 'WeblogInfo2146']
    dict_fea_encode = {}
    for icol in list_fea_str:
        dict_fea_encode[icol] = preprocessing.LabelEncoder()
        X[icol] = dict_fea_encode[icol].fit_transform(X[icol].astype(str))  # 将提示的包含错误数据类型这一列进行转换

    X = X.drop(['Idx0'], axis=1)