import pandas as pd
import numpy as np
import pandas_flavor as pf


@pf.register_dataframe_method
def remove_multicollinearity_by_coefficient_threshold(df, method = 'spearman', coefficient_threshold = 0.7):
    """
    Uses the correlation between features and a specified threshold to 
    identify and remove collinear/multicollinear features.

    Args:
        df ([pandas.DataFrame]): 
            A dataframe that includes all the features that are being considered 
            for modeling without the target.
        method (str, optional): 
            spearman: Spearman rank correlation
            pearson : Pearson correlation coefficient
            kendall : Kendall Tau correlation coefficient
            Defaults to 'spearman'as it is less prone to false alarms and 
            tests if a monotonic relationship exists, however it is more 
            computationally expensive as it is non-parametric.
        correlation_threshold (float, optional): 
            Defaults to 0.7 as per the threshold established by Dormann, C. F., 
            J. Elith, S. Bacher, et al. 2013. Collinearity: a review of methods 
            to deal with it and a simulation study evaluating their performance. 
            Ecography 36:27–46.

    Returns:
        A Pandas Dataframe that has removed all collinear/multicollinear features from 
        the feature space based upon the correlation coefficient threshold. 
    """
    correlated_features = []
    correlation_matrix  = df.corr(method = method)

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > coefficient_threshold:
                colname = correlation_matrix.columns[i]
                correlated_features.append(colname)

    # Create Final DataFrame Without Correlated Features
    df = df.drop(correlated_features, axis=1)

    return df
