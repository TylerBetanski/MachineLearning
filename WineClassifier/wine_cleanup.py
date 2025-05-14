import numpy as np
import pandas as pd

def eval_quality(quality):
    if 3 <= quality <= 4:
         return 0
    elif 4 < quality <= 7:
        return 1
    else:
        return 2

# https://www.kaggle.com/datasets/taweilo/wine-quality-dataset-balanced-classification
# 20,000 records
# Label:
#   quality:                The quality of the wine, rated from 3 to 9, with higher values indicating better quality
#                           For the purpose of this model, this will be simplified into 'low' (3 & 4), 'med' (5-7), 'high' (8 & 9)
#
# Features:
#   fixed_acidity:          The amount of fixed acids in the wine, which is typically a combination of tartaric, malic, and citric acids
#   volatile_acidity:       The amount of volatile acids in the wine, primarily acetic acid
#   citric_acid:            The amount of citric acid in the wine, contributing to the overall acidity
#   residual_sugar:         The amount of sugar remaining after fermentation
#   chlorides:              The amount of chlorides in the wine, which can indicate the presence of salt
#   free_sulfur_dioxide:    The amount of free sulfur dioxide in the wine, used as a preservative
#   total_sulfur_dioxide:   The total amount of sulfur dioxide, including bound and free forms
#   density:                The density of the wine, related to alcohol and sugar content
#   pH:                     The pH level of the wine, indicating its acidity
#   sulphates:              The amount of sulphates in the wine, contributing to its taste and preservation
#   alcohol: 	            The alcohol content of the wine in percentage
#
df = pd.read_csv("wine_data.csv")
# float_cols = df.select_dtypes("float64").columns
# df[float_cols] = df[float_cols] - np.average(df[float_cols], axis=0)
# df[float_cols] = df[float_cols] / np.std(df[float_cols], axis=0)
df['quality'] = df['quality'].apply(eval_quality)
df.to_csv("wine_cleaned.csv")