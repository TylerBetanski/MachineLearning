import numpy as np
import pandas as pd

# https://www.kaggle.com/datasets/currie32/crimes-in-chicago/data
# Limit the data to roughly the Chicago Area (since the dataset includes outliers outside of Chicago)
df = pd.read_csv('Chicago_Crimes_2012_to_2017.csv').query("41.6 <= Latitude <= 42.0 and -88.0 <= Longitude <= -87.5")
df['PrimaryType'] = df['Primary Type']
df = df[['Longitude', 'Latitude', 'PrimaryType']]
df.to_csv('crimes_cleaned')
print(df.describe())