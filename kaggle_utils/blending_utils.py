# Functions for blending predictions
import numpy as np
import pandas as pd


def make_blending(list_of_csvs, 
                  blending_name,
                  id_col,
                  target_col,
                  target_transform = 'nothing',
                  agg_function = 'mean'):
    """
    Blending several CSVs, save to new, 'blending_name' file. 
    CSV contains 2 columns: 'id_col', 'target_col'
    
    params:
        list_of_csvs: list of CSV files
        blending_name: str, name of new file where to save blended CSVs
        id_col: str, name of id column in CSV files
        target_col: str, name of target column in CSV files 
        target_transform: str, transformation of target before aggregation
            possible values: 'nothing', 'rank' (classification), 'expm1' (regression)
        agg_function: aggregation function over id_field, possible values: 'mean', 'median' 
    return:
        None
    """
    #first file processing
    result = pd.read_csv(list_of_csvs[0])
    
    # target transformation
    print(target_transform)
    if target_transform == 'rank':
        result[target_col] = result[target_col].rank(method = 'average')/result.shape[0]
    
    elif target_transform == 'expm1':
        result[target_col] = result[target_col].apply(np.expm1)
    
    elif target_transform == 'nothing':
        pass
    
    #other files processing
    for sub in list_of_csvs[1:]:
        # append new sub to the end of previous sub dataframe
        temp_sub = pd.read_csv(sub)
        
        # target transformation
        if target_transform == 'rank':
            temp_sub[target_col] = temp_sub[target_col].rank(method = 'average')/temp_sub.shape[0]
        
        elif target_transform == 'expm1':
            temp_sub[target_col] = temp_sub[target_col].apply(np.expm1)
        
        elif target_transform == 'nothing':
            pass
            
        result = result.append(temp_sub)
    
    # aggregate over id_field
    result = result.groupby([id_col], as_index=False, sort=False).agg({target_col:agg_function})
    print(agg_function)       
    
    # post aggreagte for target_transform == expm1
    if target_transform == 'expm1':
        result[target_col] = result[target_col].apply(np.log1p)
    
    result.to_csv(blending_name, index=False, header=True, sep=';')
