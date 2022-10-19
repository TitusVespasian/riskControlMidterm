import pandas as pd
import lightgbm as lgb
# from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split
from sklearn.model_selection import GridSearchCV
import re

canceData = pd.read_csv('data/train/train_all.csv',encoding='utf-8')
X = canceData.drop(['target'],axis=1).to_numpy()
y = canceData.target.to_numpy()
#new_dict = {key:i for (i,key) in enumerate(X.columns)}
#new_dict = {key:re.sub('[^A-Za-z0-9]+', '', key)+str(i) for (i,key) in enumerate(X.columns)}
#canceData.rename(columns=new_dict, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
params = {
    #'n_estimators':97,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'learning_rate': 0.1,
    'num_leaves': 30,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'is_unbalance': 'true'
}

# %% 第一步：学习率和迭代次数
data_train = lgb.Dataset(X_train, y_train)
cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',
                    early_stopping_rounds=50, seed=0)
print('best n_estimators:', len(cv_results['auc-mean']))# 70
print('best cv score:', pd.Series(cv_results['auc-mean']).max())

# %% 确定max_depth和num_leaves

params_test1 = {'max_depth': range(3, 8, 1), 'num_leaves': range(5, 100, 5)}

gsearch1 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
                                 n_estimators=70, max_depth=6, bagging_fraction=0.8, feature_fraction=0.8),
    param_grid=params_test1, scoring='roc_auc', cv=5, n_jobs=-1)
gsearch1.fit(X_train, y_train)
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
# {'max_depth': 4, 'num_leaves': 10} 0.7388414689545019
# {'max_depth': 5, 'num_leaves': 20} 0.7355383853913817


# %% 第三步：确定min_data_in_leaf和max_bin in
params_test2 = {'max_bin': range(5, 256, 10), 'min_data_in_leaf': range(1, 102, 10)}

gsearch2 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
                                 n_estimators=97, max_depth=4, num_leaves=10, bagging_fraction=0.8,
                                 feature_fraction=0.8),
    param_grid=params_test2, scoring='roc_auc', cv=5, n_jobs=-1)
gsearch2.fit(X_train, y_train)
print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_)
# {'max_bin': 165, 'min_data_in_leaf': 71} 0.7419681796813903

# %% 第四步：确定feature_fraction、bagging_fraction、bagging_freq
params_test3 = {'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
                'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
                'bagging_freq': range(0, 81, 10)
                }

gsearch3 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
                                 n_estimators=97, max_depth=4, num_leaves=10, max_bin=165, min_data_in_leaf=71),
    param_grid=params_test3, scoring='roc_auc', cv=5, n_jobs=-1)
gsearch3.fit(X_train, y_train)
print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)
# {'bagging_fraction': 0.6, 'bagging_freq': 0, 'feature_fraction': 0.8} 0.7419681796813903

# %% 第五步：确定lambda_l1和lambda_l2
params_test4 = {'lambda_l1': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                'lambda_l2': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
                }

gsearch4 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
                                 n_estimators=97, max_depth=4, num_leaves=10, max_bin=165, min_data_in_leaf=71,
                                 bagging_fraction=0.6, bagging_freq=0, feature_fraction=0.8),
    param_grid=params_test4, scoring='roc_auc', cv=5, n_jobs=-1)
gsearch4.fit(X_train, y_train)
print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)
# n_estimators=188 {'lambda_l1': 0.001, 'lambda_l2': 0.7} 0.7448869900162023
# n_estimators=97  {'lambda_l1': 0.001, 'lambda_l2': 1e-05} 0.7419695884937678

# %% 第六步：确定 min_split_gain
params_test5 = {'min_split_gain': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

gsearch5 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
                                 n_estimators=97, max_depth=4, num_leaves=10, max_bin=165, min_data_in_leaf=71,
                                 bagging_fraction=0.6, bagging_freq=0, feature_fraction=0.8,
                                 lambda_l1=0.001, lambda_l2=1e-05),
    param_grid=params_test5, scoring='roc_auc', cv=5, n_jobs=-1)
gsearch5.fit(X_train, y_train)
print(gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_)
# {'min_split_gain': 0.0} 0.7419695884937678


#{'num_leaves': 10, 'max_depth': 4, 'max_bin': 65, 'min_data_in_leaf': 101, 'feature_fraction': 0.6, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'lambda_l1': 0.001, 'lambda_l2': 0.001, 'min_split_gain': 0.0}
# %% 第七步：降低学习率，增加迭代次数，验证模型
from sklearn import metrics
model=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.01,
                                 n_estimators=70, max_depth=4, num_leaves=10, max_bin=165, min_data_in_leaf=101,
                                 bagging_fraction=0.6, bagging_freq=5, feature_fraction=0.8,
                                 lambda_l1=0.001, lambda_l2=0.001,min_split_gain=0,is_unbalance=True)
model.fit(X_train,y_train)
y_pre=model.predict(X_test)
print("acc:",metrics.accuracy_score(y_test,y_pre))
print("auc:",metrics.roc_auc_score(y_test,y_pre))




