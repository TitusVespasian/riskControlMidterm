# compare ensemble to each baseline classifier
import lightgbm as lgb
import pandas as pd
from matplotlib import pyplot
from numpy import mean
from numpy import std
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
import pickle
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

    rf_model=RandomForestClassifier(class_weight='balanced')

    return lgb_model,xgb_model,rf_model

# get a stacking ensemble of models
def get_stacking(lgb_model,xgb_model,rf_model):

    # define the base models
    level0 = list()
    level0.append(('rf', rf_model))
    level0.append(('lgb', lgb_model))
    level0.append(('xgb', xgb_model))

    # define meta learner model
    level1 = LogisticRegression()
    # define the stacking ensemble
    emodel = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return emodel


# get a list of models to evaluate
def get_models(lgb_model,xgb_model,rf_model):
    models = dict()
    models['lgb'] = lgb_model
    models['rf'] = rf_model
    models['xgb'] = xgb_model
    models['stacking'] = get_stacking(lgb_model,rf_model,xgb_model)
    return models


# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1, error_score='raise')
    return scores

if __name__=="__main__":
    # define dataset
    X, y = get_dataset()
    negative_num = y.value_counts()[0]
    positive_num = y.value_counts()[1]
    adjusted_weight = round(negative_num / positive_num, 2)
    lgb_model_,xgb_model,rf_model=get_model_param(adjusted_weight)

    # get the models to evaluate
    models = get_models(lgb_model_,xgb_model,rf_model)
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        outfile = open("./saved_model/"+name+"_model.pickle", "wb")
        pickle.dump(model, outfile)
        outfile.close()
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    # plot model performance for comparison
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()
