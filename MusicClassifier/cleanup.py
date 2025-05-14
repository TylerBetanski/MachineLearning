import pandas as pd
import numpy as np

df = pd.read_csv("spotify_dataset.csv")
df["track_genre"] = df["track_genre"].astype("string")
df = df.query("track_genre == 'country' or track_genre == 'hip-hop'")
df['genre'] = df['track_genre'].apply(lambda x: {"country": 0, "hip-hop": 1}[x]).astype("int64")
df = df.dropna()

df = df[['danceability', 'energy', 'speechiness',
        'acousticness', 'valence', 'tempo', 'genre']]

float_cols = df.select_dtypes("float64").columns
df[float_cols] = df[float_cols] - np.average(df[float_cols], axis=0)
df[float_cols] = df[float_cols] / np.std(df[float_cols], axis=0)

df.to_csv('music_cleaned.csv')
