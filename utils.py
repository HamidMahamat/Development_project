import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
1 - enlever NA values pour chaque colonne (arguments, le nom de la colonne, p, )
-> compter le nombre de catégories
-> si le nombre de Nan <p*min(N autres catégories) avec p petit, alors remplacer par catégorie makoritaire
-> sinon mettre une nouvelle catégorie 
"""

def deal_with_NA_values(df, column, r, r_float):
    """This function deals with NA values:
    - if the number of NA values is high (according to ratio r) create a new category "unknown"
    - else: replace by the majoritary category

    Args:
        df (_DataFrame_): data
        column (_string_): name of the column
        r (_float_): ratio between the number of NA values and the number of values in minority category
        r_float (_float_) : ratio between number of categories and length of the column. In order to consider if the column is categorical or not
    Returns:
        df (_pd_dataframe_): dataframe containing the column with dealed Nan problem
    """

    categories_counts = df[column].value_counts(dropna=True)
    Nan_counts = df[column].size - sum(categories_counts)
    
    if df[column].dtype == 'float' : 
        # Verify if its categorical or take any float values
        
        # Categorical float value
        if categories_counts.size/df[column].size <= r_float : 
            if Nan_counts/df[column].size <= r : # Don't create a new class
                return df[column].fillna(categories_counts.idxmax())
            else :
                return df[column].fillna(np.mean(df[column]))
            
        else :
            # When it's not categorical compute the mean 
            return df[column].fillna(np.mean(df[column]))

    else :
        # We consider as categorical all column with other non float type
        if Nan_counts/df[column].size <= r : # Don't create a new class
            return df[column].fillna(categories_counts.idxmax())
        else :
            return df[column].fillna("unknown")
            
        
        
    
