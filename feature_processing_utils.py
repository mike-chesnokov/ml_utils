# functions for data processing
import numpy as np
import pandas as pd


def get_2month_group(month):
    """
    Get 2 month group from 1 to 6 (december and january in one group)
    """
    if month == 12:
        return 1
    else:
        return int(month/2.) + 1
    

def feature_preprocessing(df_):
    """
    Simple handle of features and create other features
    
    Returns:
        df: handled dataframe
    """
    df = df_.copy()
    
    # datetime features
    df['date'] = pd.to_datetime(df['date'].astype(str))
    df['2month_gr'] = df['date'].dt.month.apply(lambda x: get_2month_group(x))
    df['session_month_year'] = df['date'].dt.month.astype(str) + '_' + df['date'].dt.year.astype(str)
    df['session_week_year'] = df['date'].dt.weekofyear.astype(str) + '_' + df['date'].dt.year.astype(str)
    
    df['session_dow'] = df['date'].dt.dayofweek
    df['session_woy'] = df['date'].dt.weekofyear
    df['session_doy'] = df['date'].dt.dayofyear
    
    df['session_year'] = df['date'].dt.year   
    df['session_month'] = df['date'].dt.month
    df['session_day'] = df['date'].dt.day
    
    df['visitStartTime'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['visitStartTime_hour'] = df['visitStartTime'].dt.hour
    
    return df


def clean_cat_features(df_, features_to_handle, features_nans):
    """
    Clean features_to_handle from df (pandas dataframe):
    join small categories, fill nans in string features where possible
    
    """
    df = df_.copy()
    for feature in features_to_handle:
        #print(features_to_handle[feature])
        df[feature + '_handled'] = df[feature].fillna(features_nans[feature]).str.lower()
        df[feature + '_handled_cut'] = 'other'
        for regexp in features_to_handle[feature]:
            inds = df[df[feature + '_handled'].str.contains(regexp, case=False, regex=True)].index
            df.loc[inds, feature + '_handled_cut'] = features_to_handle[feature][regexp]
    
        df = df.drop([feature, feature + '_handled'], axis=1)

    return df


def cut_unique_values(array, threshold, default_value = 'other'):
    '''
    Method cut number of unique values in numpy array
    with frequency lower than threshold - change values to default_value
    Params:
        array: numpy array
        threshold: int
        default_value: str
    Returns:
        object_map: dictionary with changed or original values
    '''
    uniques, counts = np.unique(array, return_counts=True)
    unique_cnt = dict(zip(uniques, counts))
    
    object_to_cut= []
    object_to_keep = []
    # if frequency lower than threshold - cut this unique value
    for obj in unique_cnt:
        if unique_cnt[obj] < threshold:
            object_to_cut.append(obj)
        else:
            object_to_keep.append(obj)
            
    # make dictionary for all unique values
    object_map = dict(zip(object_to_cut, [default_value]*len(object_to_cut)))
    object_map.update(dict(zip(object_to_keep, object_to_keep)))
    
    return object_map


def replace_low_frequency_categories(df_, cut_thresholds):
    '''
    Replace low frequency categories to other
    
    Returns:
        df: pandas dataframe with replaced categories
    '''
    df = df_.copy()
    
    for feature in cut_thresholds:
        map_dict = cut_unique_values(df[feature].values, cut_thresholds[feature])
        df[feature + '_handled_cut'] = df[feature].map(map_dict)#.apply(lambda x: map_dict(x), axis=1)
        df = df.drop([feature], axis=1)
        
    return df


def factorize_cat_features(df_, cats_to_factorize=None):
    """
    Handle dataframe features, label encoding of categoricals
    Returns:
        df: handled dataframe
        cat_indexers: unique values of categoricals from pd.factorize
    """
    df = df_.copy()
    
    # categorical features label encoding
    cat_indexers = {}
    
    for feature in cats_to_factorize:
        df[feature], indexer = pd.factorize(df[feature])
        cat_indexers[feature] = indexer
        
    return df, cat_indexers


def make_dummy_features(df_, features_to_dummies=None):
    """
    Handle categorical features to dummies
    """
    df = df_.copy()
    for feature in features_to_dummies:
        df_dummies = pd.get_dummies(df[feature], prefix=feature)
        df = pd.concat([df, df_dummies], axis=1)
        df = df.drop([feature], axis=1)
        
    return df


def make_user_aggregates(df, aggregates):
    """
    Makes user aggregates over features
    """
    # groupby user id
    df_agg = df.groupby(['fullVisitorId'], as_index=False).agg(aggregates)
    # rename multiindex columns with flat column names
    column_names = zip(df_agg.columns.get_level_values(0), df_agg.columns.get_level_values(1))
    df_agg.columns = [lvl1 +'_'+ lvl2 if lvl2 != '' else lvl1 for lvl1, lvl2 in column_names]
    
    return df_agg


def make_aggregates(df, groupby_cols, aggregates):
    """
    Makes aggregates by groupby_cols over features aggregates
    """
    # groupby user id
    df_agg = df.groupby(groupby_cols, as_index=False).agg(aggregates)
    # rename multiindex columns with flat column names
    column_names = zip(df_agg.columns.get_level_values(0), df_agg.columns.get_level_values(1))
    df_agg.columns = [lvl1 +'_'+ lvl2 if lvl2 != '' else lvl1 for lvl1, lvl2 in column_names]
    
    return df_agg


def get_string_of_categorical_sequences(df_, feature):
    """
    Get strings of categorical sequences, 
    df_ should be sorted by time
    
    return:
        user_strings: defaultdict, user as key, 
                    string of categorical sequences as value
    """
    df = df_.copy()
    user_strings = defaultdict(str)
    
    for row in df[['fullVisitorId', feature]].values:
        user_strings[row[0]] += row[1] + ' '

    for user in user_strings:
        user_strings[user] = user_strings[user].strip()
        
    return user_strings

def make_sequence_features(df_, categorical_features):
    """
    Make tfidf features from categorical featrue
    params:
        df_: pandas dataframe, sorted by time (session level)
        feature: str, categorical feature
    return:
        df_agg: pandas dataframe with new sequence feature (user level)
    """
    df = df_.copy()
    # switch to user level
    df_agg = pd.DataFrame(df['fullVisitorId'].unique(), columns=['fullVisitorId'])

    for feature in categorical_features:
        df[feature] = df[feature].astype(str)
        # get strings of categorical sequences
        user_strings = get_string_of_categorical_sequences(df, feature)
        # make feature with sequence
        df_agg[feature + '_seq'] = df_agg['fullVisitorId'].apply(lambda x: user_strings[x])
    
    return df_agg
