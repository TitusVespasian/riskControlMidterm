import pandas as pd
import lightgbm as lgb
# from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split
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
print('best n_estimators:', len(cv_results['auc-mean']))
print('best cv score:', pd.Series(cv_results['auc-mean']).max())

# %% 确定max_depth和num_leaves
from sklearn.model_selection import GridSearchCV

params_test1 = {'max_depth': range(3, 8, 1), 'num_leaves': range(5, 100, 5)}

gsearch1 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
                                 n_estimators=97, max_depth=6, bagging_fraction=0.8, feature_fraction=0.8),
    param_grid=params_test1, scoring='roc_auc', cv=5, n_jobs=-1)
gsearch1.fit(X_train, y_train)
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
# {'max_depth': 4, 'num_leaves': 10} 0.7388414689545019
