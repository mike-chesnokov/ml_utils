# Functions for metrics calculation
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE) for 2 arrays
    
    params:
        y_true: array, true values
        y_pred: array, predicted values
    return:
        RMSE
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmse_lgb(preds, dtrain):
    """
    Root Mean Squared Error (RMSE) for LightGBM data
    
    params:
        preds:array, predicted values
        dtrain: lgb.Dataset, true values
    return:
        RMSE
    """
    actuals = np.array(dtrain.get_label()) 
    return 'rmse', rmse(actuals, preds), False
