# Functions for working with LightGBM
import gc

import numpy as np
import pandas as pd
import lightgbm as lgb

from metrics_utils import rmse, rmse_lgb


def lgb_cv(X_train, 
           y_train,
           X_test,
           lgb_params,
           cv_schema,
           num_trees=1000,
           early_stopping=200,
           verbose_eval=None,
           print_progress=False):
    """
    Calculate LightGBM cross-validation by cv_schema
    
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
    # make predictions on test dataset
    if X_test:
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
        lgb_valid = lgb.Dataset(X_valid_, y_valid_, reference = lgb_train)

        lgb_model = lgb.train(lgb_params, lgb_train, num_trees,  
                              valid_sets = [lgb_valid], 
                              valid_names=['valid'],
                              feval = rmse_lgb, 
                              verbose_eval=verbose_eval, 
                              early_stopping_rounds=early_stopping,
                              feature_name=list(X_train.columns))

        del X_train_, y_train_, lgb_train, lgb_valid
        gc.collect()
        
        if print_progress:
            print('Fold trained...')

        # make predictions on test and valid datasets
        if X_test:
            lgb_pred = lgb_model.predict(X_test,  num_iteration = lgb_model.best_iteration)
        y_valid_pred = lgb_model.predict(X_valid_,  num_iteration = lgb_model.best_iteration)

        best_iterations.append(lgb_model.best_iteration)
        rmse_current = rmse(y_valid_, y_valid_pred)
        rmses.append(rmse_current)
        if print_progress:
            print('RMSE  valid',  round(rmse_current, 6))

        if X_test:
            lgb_preds.append(lgb_pred)
        y_valid_preds[valid_index] = y_valid_pred

        del X_valid_, y_valid_
        gc.collect()

    if print_progress:
        print('\nRMSE mean: ', round(np.mean(rmses), 6), ', std: ', round(np.std(rmses), 6))
        print('RMSE OOF valid: ', round(rmse(y_train, y_valid_preds), 6))
        print('AVG num best iteration: ', np.mean(best_iterations))
        print('LGBM model trained...')
    
    if X_test:
        return np.mean(lgb_preds, axis=0), rmse(y_train, y_valid_preds), np.mean(rmses), np.std(rmses), np.mean(best_iterations)
    else:
        return rmse(y_train, y_valid_preds), np.mean(rmses), np.std(rmses), np.mean(best_iterations)
    