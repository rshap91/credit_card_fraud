# -- coding utf8 --
# Title: Cleaning Function Helpers
# Author: Rick Shapiro
"""
Functions to 
    - Remove (optionally non-fraudulent) outliers
    - Normalize Data
    - Deskew features
""";


import numpy as np
import pandas as pd
from scipy import stats


def is_outlier(ser, S=3, std=True):
    """
    Return boolean series for whether or not
    point is greater than S std from the mean
    or S times the iqr (if std = False) from the median
    """
    
    if std:
        return ((ser-ser.mean())/ser.std()).abs() > S
    else:
        iqr = ser.quantile(0.75) - ser.quantile(0.25)
        return ser.abs() > (ser.median() + S*iqr)


def remove_outliers(df, S=3, std=True, subset_rows=None, subset_cols=None):
    '''
    Function to transform dataframe and return frame with all outliers removed.
    
    df: DataFrame; Data containing outliers to be removed
    S: INT; Parameter used for identifying outliers. Larger S is more conservative, smaller is more liberal.
    std: BOOL; If True, all data_points in a column more than S * STD from the mean (in either direction) 
                        are considered outliers.
                If False, all data_points in a column greater S * IQR from the median (in either direction)
                        are considered outliers.
    subset_cols: LIST; Only remove outliers from passed columns. If None, remove outliers from all numeric columns.
    subset_rows: BOOLEAN MASK; passed to df.loc[subset_rows,subset_cols]
    sort_values: STRING or LIST of STRINGS; Column name(s) to sort resulting dataframe on.
    
    '''
    if isinstance(subset_cols, type(None)):
        subset_cols = df.select_dtypes('number').columns
    if isinstance(subset_rows, type(None)):
        subset_rows = df.index
    
    included = df.loc[subset_rows, subset_cols]
    
    outliers = included.apply(lambda ser: is_outlier(ser, S, std))
    drop = outliers[outliers.any(1)].index
    
    return df.drop(drop).reset_index(drop=True)



def scale_data(df, scaler, subset_cols=None, fit=True):
    """
    Scales columns of a DataFrame using scaler. Optionally fit the scaler as well as transform.
    df: DATAFRAME; data with columns to scale.
    scaler: Instantiated (and optionally already fit) instance of sklearn scaler or similar
    
    """
    if isinstance(subset_cols, type(None)):
        subset_cols = df.select_dtypes('number').columns
    
    cdf = df.copy()
    
    if fit:
        cdf.loc[:,subset_cols] = scaler.fit_transform(cdf.loc[:,subset_cols])
    else:
        cdf.loc[:,subset_cols] = scaler.transform(cdf.loc[:,subset_cols])
    return cdf
    
    
def deskew_df(df, subset_cols=None, threshold=None, topn=None, method=np.sqrt):
    """
    Deskews columns that meet either threshold or topn criteria for skewness using function `method`.
    
    df: DATAFRAME; data with columns to be deskewed.
    subset_cols: LIST; columns to deskew
    threshold: FLOAT; deskew columns with abs(skew) > threshold. Ignored if subset_cols is specified
    topn: INTEGER; deskew topn most skewed columns. Ignored if threshold or subset_cols is specified.
        NOTE: if none of subset_cols, threshold, topn are specified, all columns are deskewed.
    method: function to transform data. Defaults to boxcox.
    """
    
    if subset_cols:
        include = subset_cols
    elif threshold:
        include = df.columns[df.skew().abs()>threshold]
    elif topn:
        include = df.skew().abs().sort_values(ascending=False).index[:topn].tolist()
    else:
        include = df.columns.tolist()
    assert (df[include] >= 0).all().all(), 'Data must be positive.' 
    exclude = df.columns[~df.columns.isin(include)].tolist()

    return df.apply(lambda ser: ser.map(method) if ser.name in include else ser)
    