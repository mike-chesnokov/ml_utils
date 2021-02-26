# Methods for working with LightGBM
# - "lgb_cv_regression" - cross-validation for regression task with metric RMSE
# - "lgb_cv_multiclass" - cross-validation for multiclass classification task with metric accuracy
# - "lgb_cv_binary_class" - cross-validation for binary classification task with metric AUC

import gc

import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold

from utils.metrics_utils import rmse, rmse_lgb


kf = KFold(n_splits=5, random_state=7, shuffle=False)
lgb_regression_params = {
    'boosting': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'boost_from_average': False,
    'is_training_metric': False,
    'seed': 77,
    'learning_rate': 0.01,
    #'max_depth': 5,
    'min_data_in_leaf': 200,
    #'min_gain_to_split': 0.001,
    #'min_sum_hessian_in_leaf': 120.,
    'save_binary': True,
    'num_leaves': 31,
    #'max_bin': 32,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 10,
    'lambda_l2': 1.,
    'lambda_l1': 1.,
    'num_threads': 8,
    'verbosity': -1
    }


def lgb_cv_regression(X_train,
                      y_train,
                      X_test,
                      lgb_params,
                      cv_schema,
                      num_trees=1000,
                      early_stopping=200,
                      verbose_eval=None,
                      print_progress=False):
    """
    Calculate LightGBM cross-validation by cv_schema for regression task (metric - RMSE)
    
    params:
        X_train: 2d array, train data
        y_train: 1d array, train target
        X_test: 2d array, test data, if X_test=None not to compute mean of test predictions from every fold
        lgb_params: params for LightGBM
        cv_schema: cross-validation schema
        num_trees: num_boosting_rounds
        early_stopping: early stopping rounds
        verbose_eval: print progress of LightGBM training
        print_progress: print progress of cross-validation
    return:
        mean test predictions from every fold (if X_test != None)
        RMSE out of fold
        RMSE mean over folds
        RMSE std over folds
        mean of best LightGBM iteration over folds
    """

    lgb_preds = []
    rmses = []
    best_iterations = []

    y_valid_preds = np.zeros(y_train.values.shape)
    
    for fold_count, (train_index, valid_index) in enumerate(cv_schema.split(X_train, y_train)):
        if print_progress:
            print('*******************************************')
            print('Starting fold {}'.format(fold_count))

        X_train_, X_valid_ = X_train.values[train_index], X_train.values[valid_index]
        y_train_, y_valid_ = y_train.values[train_index], y_train.values[valid_index]

        lgb_train = lgb.Dataset(X_train_, y_train_) 
        lgb_valid = lgb.Dataset(X_valid_, y_valid_, reference=lgb_train)

        lgb_model = lgb.train(lgb_params, lgb_train, num_trees,  
                              valid_sets=[lgb_valid],
                              valid_names=['valid'],
                              feval=rmse_lgb,
                              verbose_eval=verbose_eval, 
                              early_stopping_rounds=early_stopping,
                              feature_name=list(X_train.columns))

        del X_train_, y_train_, lgb_train, lgb_valid
        gc.collect()
        
        if print_progress:
            print('Fold trained...')

        # make predictions on valid dataset
        y_valid_pred = lgb_model.predict(X_valid_,  num_iteration=lgb_model.best_iteration)

        best_iterations.append(lgb_model.best_iteration)
        rmse_current = rmse(y_valid_, y_valid_pred)
        rmses.append(rmse_current)
        if print_progress:
            print('RMSE  valid',  round(rmse_current, 6))

        # make predictions on test dataset
        if X_test is not None:
            lgb_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
            lgb_preds.append(lgb_pred)
        y_valid_preds[valid_index] = y_valid_pred

        del X_valid_, y_valid_
        gc.collect()

    if print_progress:
        print('\nRMSE mean: ', round(np.mean(rmses), 6), ', std: ', round(np.std(rmses), 6))
        print('RMSE OOF valid: ', round(rmse(y_train, y_valid_preds), 6))
        print('AVG num best iteration: ', np.mean(best_iterations))
        print('LGBM model trained...')
    
    if X_test is not None:
        return np.mean(lgb_preds, axis=0), rmse(y_train, y_valid_preds), \
               np.mean(rmses), np.std(rmses), np.mean(best_iterations)
    else:
        return rmse(y_train, y_valid_preds), np.mean(rmses), np.std(rmses), np.mean(best_iterations)


lgb_multiclass_params = {
    'boosting': 'gbdt',
    'objective': 'multiclass',
    'num_class': 4,
    'metric': ['multi_logloss', 'multi_error'],
    'is_training_metric': False,
    'seed': 77,
    'learning_rate': 0.15,
    'min_data_in_leaf': 500,
    'save_binary': True,
    'num_leaves': 31,
    'max_bin': 32,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'lambda_l2': 1.,
    'lambda_l1': 1.,
    'num_threads': 10,
    'verbosity': -1
}


def lgb_cv_multiclass(X_train,
                      y_train,
                      X_test,
                      lgb_params,
                      cv_schema,
                      num_trees=120,
                      early_stopping=20,
                      verbose_eval=20,
                      print_progress=True):
    """
    Calculate LightGBM cross-validation by cv_schema for multiclass classification (metric accuracy)

    params:
        X_train: 2d array, train data (sparse)
        y_train: 1d array, train target
        X_test: 2d array, test data, if X_test=None not to compute mean of test predictions from every fold
        lgb_params: params for LightGBM
        cv_schema: cross-validation schema
        num_trees: num_boosting_rounds
        early_stopping: early stopping rounds
        verbose_eval: print progress of LightGBM training
        print_progress: print progress of cross-validation
    return:
        mean test predictions from every fold (if X_test != None)
        Accuracy out of fold
        Accuracy mean over folds
        Accuracy std over folds
        mean of best LightGBM iteration over folds
    """
    accuraces = []
    best_iterations = []
    test_preds = []
    y_valid_preds = np.zeros(y_train.shape)

    for fold_count, (train_index, valid_index) in enumerate(cv_schema.split(X_train, y_train)):
        if print_progress:
            print('*******************************************')
            print('Starting fold {}'.format(fold_count))

        X_train_, X_valid_ = X_train[train_index, :], X_train[valid_index, :]
        y_train_, y_valid_ = y_train[train_index], y_train[valid_index]

        lgb_train = lgb.Dataset(X_train_, y_train_)
        lgb_valid = lgb.Dataset(X_valid_, y_valid_, reference=lgb_train)

        if print_progress:
            print(X_train_.shape, X_valid_.shape)
            print(y_train_.shape, y_valid_.shape)
            print('datasets created')

        lgb_model = lgb.train(lgb_params, lgb_train, num_trees,
                              valid_sets=[lgb_valid],
                              valid_names=['valid'],
                              verbose_eval=verbose_eval,
                              early_stopping_rounds=early_stopping)

        del X_train_, y_train_, lgb_train, lgb_valid
        gc.collect()
        if print_progress:
            print('Fold trained...')

        # make predictions on valid dataset
        y_valid_pred = lgb_model.predict(X_valid_, num_iteration=lgb_model.best_iteration)
        y_valid_pred = y_valid_pred.argmax(axis=1)
        y_valid_preds[valid_index] = y_valid_pred

        # make predictions on test dataset
        if X_test is not None:
            current_test_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
            test_preds.append(current_test_pred)

        best_iterations.append(lgb_model.best_iteration)
        acc_current = accuracy_score(y_valid_, y_valid_pred)
        accuraces.append(acc_current)
        if print_progress:
            print('Accuracy  valid', round(acc_current, 6))

        del X_valid_, y_valid_
        gc.collect()

    if print_progress:
        print('\nAccuracy mean: ', round(np.mean(accuraces), 6), ', std: ', round(np.std(accuraces), 6))
        print('Accuracy OOF valid: ', round(accuracy_score(y_train, y_valid_preds), 6))
        print('AVG num best iteration: ', np.mean(best_iterations))

    if X_test is not None:
        return sum(test_preds) / len(test_preds), accuracy_score(y_train, y_valid_preds), \
               np.mean(accuraces), np.std(accuraces), np.mean(best_iterations)
    else:
        return accuracy_score(y_train, y_valid_preds), np.mean(accuraces), np.std(accuraces), np.mean(best_iterations)


lgb_params = {
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'is_training_metric': False,
    'seed': 77,
    'learning_rate': 0.01,
    'min_data_in_leaf':100,
    'save_binary': True,
    'num_leaves': 31,
    'max_bin': 32,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'lambda_l2': 1.,
    'lambda_l1': 1.,
    'num_threads': 10,
    'verbosity':-1
}


def lgb_cv_binary_class(X_train,
                        y_train,
                        X_test,
                        lgb_params,
                        cv_schema,
                        num_trees=120,
                        early_stopping=20,
                        verbose_eval=20,
                        print_progress=True):
    """
    Calculate LightGBM cross-validation by cv_schema for binary classification (metric auc)

    params:
        X_train: 2d array, train data (sparse)
        y_train: 1d array, train target
        X_test: 2d array, test data, if X_test=None not to compute mean of test predictions from every fold
        lgb_params: params for LightGBM
        cv_schema: cross-validation schema
        num_trees: num_boosting_rounds
        early_stopping: early stopping rounds
        verbose_eval: print progress of LightGBM training
        print_progress: print progress of cross-validation
    return:
        mean test predictions from every fold (if X_test != None)
        AUC out of fold
        AUC mean over folds
        AUC std over folds
        mean of best LightGBM iteration over folds
    """
    aucs = []
    best_iterations = []
    test_preds = []
    y_valid_preds = np.zeros(y_train.shape)

    for fold_count, (train_index, valid_index) in enumerate(cv_schema.split(X_train, y_train)):
        if print_progress:
            print('*******************************************')
            print('Starting fold {}'.format(fold_count))

        X_train_, X_valid_ = X_train.values[train_index], X_train.values[valid_index]
        y_train_, y_valid_ = y_train.values[train_index], y_train.values[valid_index]

        lgb_train = lgb.Dataset(X_train_, y_train_)
        lgb_valid = lgb.Dataset(X_valid_, y_valid_, reference=lgb_train)

        if print_progress:
            print(X_train_.shape, X_valid_.shape)
            print(y_train_.shape, y_valid_.shape)
            print('datasets created')

        lgb_model = lgb.train(lgb_params, lgb_train, num_trees,
                              valid_sets=[lgb_valid],
                              valid_names=['valid'],
                              verbose_eval=verbose_eval,
                              early_stopping_rounds=early_stopping)

        del X_train_, y_train_, lgb_train, lgb_valid
        gc.collect()
        if print_progress:
            print('Fold trained...')

        # make predictions on valid dataset
        y_valid_pred = lgb_model.predict(X_valid_, num_iteration=lgb_model.best_iteration)
        y_valid_preds[valid_index] = y_valid_pred

        # make predictions on test dataset
        if X_test is not None:
            current_test_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
            test_preds.append(current_test_pred)

        best_iterations.append(lgb_model.best_iteration)
        auc_current = roc_auc_score(y_valid_, y_valid_pred)
        aucs.append(auc_current)
        if print_progress:
            print('AUC  valid', round(auc_current, 6))

        del X_valid_, y_valid_
        gc.collect()

    if print_progress:
        print('\nAUC mean: ', round(np.mean(aucs), 6), ', std: ', round(np.std(aucs), 6))
        print('AUC OOF valid: ', round(roc_auc_score(y_train, y_valid_preds), 6))
        print('AVG num best iteration: ', np.mean(best_iterations))

    if X_test is not None:
        return sum(test_preds) / len(test_preds), roc_auc_score(y_train, y_valid_preds), \
               np.mean(aucs), np.std(aucs), np.mean(best_iterations)
    else:
        return y_valid_preds, roc_auc_score(y_train, y_valid_preds), \
               np.mean(aucs), np.std(aucs), np.mean(best_iterations)
