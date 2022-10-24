import joblib
from sklearn import metrics
import lightgbm as lgb
import pandas as pd
# from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler

from imblearn.over_sampling import SMOTE, ADASYN

canceData = pd.read_csv('./train_all.csv', encoding='utf-8')  # training data
# canceData = pd.read_csv('./train_test_all.csv', encoding='utf-8')# training if target=Nan, test if target!=nan
X = canceData.drop(['target'], axis=1).to_numpy()
y = canceData.target.to_numpy()


ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = SMOTE().fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, random_state=0, test_size=0.2)

params = {
    # 'n_estimators':56,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'learning_rate': 0.001,
    'num_leaves': 18,
    'max_depth': 12,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'is_unbalance': 'true'
}

# %% 第一步：学习率和迭代次数
# data_train = lgb.Dataset(X_train, y_train)
# cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',
#                     early_stopping_rounds=50, seed=0)
# print('best n_estimators:', len(cv_results['auc-mean']))
# print('best cv score:', pd.Series(cv_results['auc-mean']).max())
# best_pa = {'n_estimators': 320}
# best_pa = {'n_estimators': len(cv_results['auc-mean'])}

# best n_estimators: 73
# best cv score: 0.729953228991693

# best n_estimators: 102
# best cv score: 0.725927144946139

# best n_estimators: 123
# best cv score: 0.7437586174184376

# best n_estimators: 147
# best cv score: 0.7458067831572681

# best n_estimators: 996
# best cv score: 0.7438864135888321

# best n_estimators: 480
# best cv score: 0.747441553599157

# best n_estimators: 320
# best cv score: 0.7460144686350494

# best n_estimators: 272
# best cv score: 0.7467946717590775



# %% 确定max_depth和num_leaves

# params_test1 = {'max_depth': range(12, 20, 1), 'num_leaves': range(16, 24, 1)}
# gsearch1 = GridSearchCV(
#     estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=params['learning_rate'],
#                                  n_estimators=best_pa['n_estimators'], bagging_fraction=0.8,
#                                  feature_fraction=0.8,
#                                  is_unbalance=True),
#     param_grid=params_test1, scoring='roc_auc', cv=5, n_jobs=-1)
# gsearch1.fit(X_train, y_train)
# print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
best_pa = {'n_estimators': 278, 'max_depth': 12, 'num_leaves': 18}


# {'max_depth': 5, 'num_leaves': 10} 0.7334410002865965
# {'max_depth': 4, 'num_leaves': 8} 0.7525711222807675
# {'max_depth': 3, 'num_leaves': 5} 0.7444022125723817
# {'max_depth': 8, 'num_leaves': 12} 0.7503923790639597
# {'max_depth': 9, 'num_leaves': 18} 0.7507011974408743
# best_pa.update(gsearch1.best_params_.items())


# %% 第三步：确定min_data_in_leaf和max_bin in
# params_test2 = {'max_bin': range(80, 90, 1), 'min_data_in_leaf': range(55, 65, 1)}

# gsearch2 = GridSearchCV(
#     estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=params['learning_rate'],
#                                  n_estimators=best_pa['n_estimators'], max_depth=best_pa['max_depth'],
#                                  num_leaves=best_pa['num_leaves'], bagging_fraction=0.8,
#                                  feature_fraction=0.8, is_unbalance=True),
#     param_grid=params_test2, scoring='roc_auc', cv=5, n_jobs=-1)
# gsearch2.fit(X_train, y_train)
# print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_)
# best_pa.update(gsearch2.best_params_)
best_pa = {'n_estimators': 278, 'max_depth': 12, 'num_leaves': 18,
           'max_bin': 86, 'min_data_in_leaf': 63
           }

# {'max_bin': 75, 'min_data_in_leaf': 101} 0.7401419009586869
# {'max_bin': 85, 'min_data_in_leaf': 81} 0.7489440909327147
# {'max_bin': 85, 'min_data_in_leaf': 60} 0.7506061852399537
# {'max_bin': 86, 'min_data_in_leaf': 63} 0.7548011506725587


# %% 第四步：确定feature_fraction、bagging_fraction、bagging_freq

# params_test3 = {'feature_fraction': [0.32, 0.33, 0.34, 0.35],
#                 'bagging_fraction': [0.808, 0.809, 0.81, 0.812, 0.813]
#                 }
# gsearch3 = GridSearchCV(
#     estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=params['learning_rate'],
#                                  n_estimators=best_pa['n_estimators'], max_depth=best_pa['max_depth'],
#                                  num_leaves=best_pa['num_leaves'], max_bin=best_pa['max_bin'],
#                                  min_data_in_leaf=best_pa['min_data_in_leaf'], bagging_freq=5,
#                                  is_unbalance=True),
#     param_grid=params_test3, scoring='roc_auc', cv=5, n_jobs=-1)
# gsearch3.fit(X_train, y_train)
# print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)

# {'bagging_fraction': 0.6, 'bagging_freq': 0, 'feature_fraction': 0.8} 0.7401419009586869
# {'bagging_fraction': 0.6, 'bagging_freq': 0, 'feature_fraction': 0.8} 0.7506061852399537
# {'bagging_fraction': 0.81, 'feature_fraction': 0.35} 0.7525001563693136
# {'bagging_fraction': 0.81, 'feature_fraction': 0.34} 0.7568269213071707
# best_pa.update(gsearch3.best_params_)
best_pa = {'n_estimators': 278, 'max_depth': 12, 'num_leaves': 18,
           'max_bin': 86, 'min_data_in_leaf': 63,
           'bagging_fraction': 0.81, 'feature_fraction': 0.34, 'bagging_freq': 5,
           }


# %% 第五步：确定lambda_l1和lambda_l2

# params_test4 = {'lambda_l1': [0.305, 0.306, 0.307, 0.308, 0.309, 0.4],
#                 'lambda_l2': [0.201, 0.202, 0.203]
#                 }

# gsearch4 = GridSearchCV(
#     estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=params['learning_rate'],
#                                  n_estimators=best_pa['n_estimators'], max_depth=best_pa['max_depth'],
#                                  num_leaves=best_pa['num_leaves'], max_bin=best_pa['max_bin'],
#                                  min_data_in_leaf=best_pa['min_data_in_leaf'],
#                                  bagging_fraction=best_pa['bagging_fraction'], bagging_freq=best_pa['bagging_freq'],
#                                  feature_fraction=best_pa['feature_fraction'], is_unbalance=True),
#     param_grid=params_test4, scoring='roc_auc', cv=5, n_jobs=-1)
# gsearch4.fit(X_train, y_train)
# print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)

# best_pa.update(gsearch4.best_params_)
# {'lambda_l1': 1e-05, 'lambda_l2': 0.001} 0.7401421591287665
# {'lambda_l1': 0.31, 'lambda_l2': 0.168} 0.754483040384203
# {'lambda_l1': 0.302, 'lambda_l2': 0.168} 0.7577108937882853
# {'lambda_l1': 0.306, 'lambda_l2': 0.202} 0.7585348163558537
best_pa = {'n_estimators': 278, 'max_depth': 12, 'num_leaves': 18,
           'max_bin': 86, 'min_data_in_leaf': 63,
           'bagging_fraction': 0.81, 'feature_fraction': 0.34, 'bagging_freq': 5,
           'lambda_l1': 0.306, 'lambda_l2': 0.202
           }


# # %% 第六步：确定 min_split_gain
# params_test5 = {'min_split_gain': [0.0, 0.0001, 0.001, 0.01, 0.02]}

# gsearch5 = GridSearchCV(
#     estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=params['learning_rate'],
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
# {'min_split_gain': 0.0} 0.754483040384203

# best_pa = {'n_estimators': 278, 'max_depth': 12, 'num_leaves': 18,
#            'max_bin': 86, 'min_data_in_leaf': 63,
#            'bagging_fraction': 0.81, 'feature_fraction': 0.34, 'bagging_freq': 5,
#            'lambda_l1': 0.306, 'lambda_l2': 0.202, 'min_split_gain': 0.0
#            }


# print(best_pa)
# best_pa = {'n_estimators': 147, 'max_depth': 3,'learning_rate':0.03,
#            'num_leaves': 5, 'max_bin': 85, 'min_data_in_leaf': 60,
#            'bagging_fraction': 0.81, 'feature_fraction': 0.35, 'bagging_freq': 5,
#            'lambda_l1': 0.31, 'lambda_l2': 0.168, 'min_split_gain': 0.0
#            }
# best_pa = {'n_estimators': 278, 'max_depth': 12, 'num_leaves': 18,
#            'max_bin': 86, 'min_data_in_leaf': 63,
#            'bagging_fraction': 0.81, 'feature_fraction': 0.34, 'bagging_freq': 3,
#            'lambda_l1': 0.306, 'lambda_l2': 0.202, 'min_split_gain': 0.0
#            }
best_pa = {'n_estimators': 278, 'max_depth': 12, 'num_leaves': 31,
           'max_bin': 86, 'min_data_in_leaf': 63,
           'bagging_fraction': 0.81, 'feature_fraction': 0.34, 'bagging_freq': 5,
           'lambda_l1': 0, 'lambda_l2': 0, 'min_split_gain': 0.0
           }
# best_pa = {'n_estimators': 65, 'max_depth': 5, 'num_leaves': 15, 'max_bin': 65, 'min_data_in_leaf': 21, 'bagging_fraction': 0.6,
#            'bagging_freq': 0, 'feature_fraction': 0.8, 'lambda_l1': 1.0, 'lambda_l2': 0.1, 'min_split_gain': 0.0}
# %% 第七步：降低学习率，增加迭代次数，验证模型

model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.001,
                           n_estimators=15000, max_depth=best_pa['max_depth'],
                           num_leaves=best_pa['num_leaves'], max_bin=best_pa['max_bin'],
                           min_data_in_leaf=best_pa['min_data_in_leaf'],
                           bagging_fraction=best_pa['bagging_fraction'], bagging_freq=best_pa['bagging_freq'],
                           feature_fraction=best_pa['feature_fraction'],
                           lambda_l1=best_pa['lambda_l1'], lambda_l2=best_pa['lambda_l2'],
                           min_split_gain=best_pa['min_split_gain'], is_unbalance=False)
model.fit(X_train, y_train)

joblib.dump(model, 'dota_model.pkl')
clf = joblib.load('dota_model.pkl')
# auc: 0.7536763114513698
# # 模型预测
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
y_pre = clf.predict_proba(X_test)[:, 1]
# print("acc:", metrics.accuracy_score(y_test, y_pre))
print("auc:", metrics.roc_auc_score(y_test, y_pre))
# auc: 0.7536763114513698
all_all = pd.read_csv("./train_test_all.csv")
train = all_all[all_all['target'].notnull()]
X_train_all = train.drop(['target', "Idx"], axis=1)
y_train_all = train["target"]
test = all_all[all_all['target'].isnull()]
ID = test["Idx"]
ID.reset_index(drop=True, inplace=True)
X_test = test.drop(['target', "Idx"], axis=1)
model.fit(X_train_all, y_train_all)

y_hat = model.predict_proba(X_test)[:, 1]
df = pd.DataFrame(data={"target": y_hat})
df = df.join(ID)
df.to_csv("result.csv", index=False)
