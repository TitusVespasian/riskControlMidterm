# blending ensemble for classification using hard voting
import lightgbm as lgb
import pandas as pd
from numpy import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier


# get the dataset
def get_dataset():
    train_all = pd.read_csv("data/train/train_all.csv")
    X = train_all.drop(['target'], axis=1)
    y = train_all['target']
    return X, y


def get_model_param(adjusted_weight):
    lgb_best_pa = {'n_estimators': 65, 'max_depth': 5, 'num_leaves': 15, 'max_bin': 65, 'min_data_in_leaf': 21,
                   'bagging_fraction': 0.6, 'bagging_freq': 0, 'feature_fraction': 0.8, 'lambda_l1': 1.0,
                   'lambda_l2': 0.1,
                   'min_split_gain': 0.0}
    lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.01,
                                   max_depth=lgb_best_pa['max_depth'],
                                   num_leaves=lgb_best_pa['num_leaves'], max_bin=lgb_best_pa['max_bin'],
                                   min_data_in_leaf=lgb_best_pa['min_data_in_leaf'],
                                   bagging_fraction=lgb_best_pa['bagging_fraction'],
                                   bagging_freq=lgb_best_pa['bagging_freq'],
                                   feature_fraction=lgb_best_pa['feature_fraction'],
                                   lambda_l1=lgb_best_pa['lambda_l1'], lambda_l2=lgb_best_pa['lambda_l2'],
                                   min_split_gain=lgb_best_pa['min_split_gain'],
                                   n_estimators=1000, is_unbalance=True)

    xgb_model = XGBClassifier(
        learning_rate=0.1,
        n_estimators=70,
        max_depth=3,
        min_child_weight=7,
        gamma=0,
        # subsample=,
        nthread=-1,
        scale_pos_weight=adjusted_weight,
        # 如果可以，在远端服务器运行，把下面的注释放开以获得显卡支持
        # tree_method='gpu_hist',
        random_state=1,
        use_label_encoder=False,
    )
    rf_model = RandomForestClassifier(class_weight='balanced')
    return lgb_model, xgb_model, rf_model


# get a list of base models
def get_models(lgb_model, xgb_model, rf_model):
    _models = list()
    _models.append(('lgb', lgb_model))
    _models.append(('rf', rf_model))
    _models.append(('xgb', xgb_model))
    return _models


# fit the blending ensemble
def fit_ensemble(models, X_train, X_val, y_train, y_val):
    # fit all models on the training set and predict on hold out set
    meta_X = list()
    for name, model in models:
        # fit in training set
        model.fit(X_train, y_train)
        # predict on hold out set
        yhat = model.predict_proba(X_val)[:,1]
        # reshape predictions into a matrix with one column
        yhat = yhat.reshape(len(yhat), 1)
        # store predictions as input for blending
        meta_X.append(yhat)
    # create 2d array from predictions, each set is an input feature
    meta_X = hstack(meta_X)
    # define blending model
    blender = LogisticRegression(penalty='l1',class_weight="balanced", solver='saga')
    # fit on predictions from base models
    blender.fit(meta_X, y_val)
    return blender


# make a prediction with the blending ensemble
def predict_ensemble(models, blender, X_test):
    # make predictions with base models
    meta_X = list()
    for name, model in models:
        # predict with base model
        yhat = model.predict_proba(X_test)[:, 1]
        # reshape predictions into a matrix with one column
        yhat = yhat.reshape(len(yhat), 1)
        # store prediction
        meta_X.append(yhat)
    # create 2d array from predictions, each set is an input feature
    meta_X = hstack(meta_X)
    # predict
    return blender.predict_proba(meta_X)[:, 1]


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# define dataset
X, y = get_dataset()
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
# split training set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.4, random_state=1,
                                                  stratify=y_train_full)
# summarize data split
print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))
# get model params
negative_num = y.value_counts()[0]
positive_num = y.value_counts()[1]
adjusted_weight = round(negative_num / positive_num, 2)
lgb_model_, xgb_model, rf_model = get_model_param(adjusted_weight)
# create the base models
models = get_models(lgb_model_, xgb_model, rf_model)
# train the blending ensemble
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
# make predictions on test set
yhat = predict_ensemble(models, blender, X_test)
# evaluate predictions
score = roc_auc_score(y_test, yhat)
print('Blending Auc: %.3f' % (score * 100))

# all_all = pd.read_csv("data/all/all_set.csv")
# X = all_all.drop(['target'], axis=1)
# y = all_all['target']
# X_train_all=X['target'].notnull()
