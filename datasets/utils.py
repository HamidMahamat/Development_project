import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
1 - enlever NA values pour chaque colonne (arguments, le nom de la colonne, p, )
-> compter le nombre de catégories
-> si le nombre de Nan <p*min(N autres catégories) avec p petit, alors remplacer par catégorie makoritaire
-> sinon mettre une nouvelle catégorie 
"""

def deal_with_NA_values(df, column, r):
    """This function deals with NA values:
    - if the number of NA values is high (according to ratio r) create a new category "unknown"
    - else: replace by the majoritary category

    Args:
        df (_DataFrame_): data
        column (_string_): name of the column
        r (_float_): ratio between the number of NA values and the number of values in minority category

    Returns:
        _type_: _description_
    """

    categories_counts = df[column].value_counts(dropna=True)
    
    return categories_counts
