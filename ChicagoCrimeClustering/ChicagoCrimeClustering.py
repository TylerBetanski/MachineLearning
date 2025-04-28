import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Chicago Map from:
# https://maps.co/gis/

# Chicago Police Precincts:
# https://chicagopd.maps.arcgis.com/apps/instant/nearby/index.html?appid=11a23d43d62b4f929dd0ec0f8c013506

# Dataset:
# https://www.kaggle.com/datasets/currie32/crimes-in-chicago/data
df = pd.read_csv('crimes_cleaned')

# Tracking:
# THEFT, ROBBERY, BURGLARY
# BATTERY, ASSAULT, HOMICIDE, INTIMIDATION
# NARCOTICS, OTHER NARCOTIC VIOLATION
# STALKING, CRIM SEXUAL ASSAULT, SEX OFFENSE, KIDNAPPING, OBSCENITY, HUMAN TRAFFICKING
# WEAPONS VIOLATION, CONCEALED CARRY LICENSE VIOLATION

X_larceny = df.query("PrimaryType.str.contains('ROBBERY') or "
                     "PrimaryType.str.contains('BURGLARY') or "
                     "PrimaryType.str.contains('THEFT')").iloc[:, 1:3].to_numpy()
X_violent_crimes = df.query("PrimaryType.str.contains('BATTERY') or "
                     "PrimaryType.str.contains('ASSAULT') or "
                     "PrimaryType.str.contains('HOMICIDE') or "
                     "PrimaryType.str.contains('INTIMIDATION')").iloc[:, 1:3].to_numpy()
X_drug_crimes = df.query("PrimaryType.str.contains('NARCOTICS') or "
                     "PrimaryType.str.contains('OTHER NARCOTIC VIOLATION')").iloc[:, 1:3].to_numpy()
X_sexual_crimes = df.query("PrimaryType.str.contains('STALKING') or "
                     "PrimaryType.str.contains('CRIM SEXUAL ASSAULT') or "
                     "PrimaryType.str.contains('SEX OFFENSE') or "
                     "PrimaryType.str.contains('KIDNAPPING') or "
                     "PrimaryType.str.contains('OBSCENITY') or "
                     "PrimaryType.str.contains('HUMAN TRAFFICKING')").iloc[:, 1:3].to_numpy()
X_weapons = df.query("PrimaryType.str.contains('WEAPONS VIOLATION') or "
                     "PrimaryType.str.contains('CONCEALED CARRY LICENSE VIOLATION')").iloc[:, 1:3].to_numpy()

# DRAW CRIME BY TYPE
index = 0
names = ['LARCENY', 'VIOLENT CRIMES', 'NARCOTICS', 'SEX CRIMES', 'WEAPONS VIOLATIONS']
for X_type in [X_larceny, X_violent_crimes, X_drug_crimes, X_sexual_crimes, X_weapons]:
    plt.style.use('dark_background')
    fig, ax = plt.subplots()

    # Set the limits of the plot to bound them to the extends of the city.
    ax.set_xlim([-88.0, -87.5])
    ax.set_ylim([41.6, 42.0])

    # K-Means clustering to find the 8 areas with the most offenses of each type.
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(X_type)

    # Draw lines to mark the boundaries of clusters with a Voronoi diagram.
    vor = Voronoi(kmeans.cluster_centers_)
    voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_colors='black', line_width=1.0, ax=ax)

    # Draw a "heatmap" with crimes of this type over the map of Chicago.
    plt.title(names[index])
    index += 1
    plt.hexbin(X_type[:, 0], X_type[:, 1], gridsize=100, alpha=0.4)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='r')
    img = plt.imread("chicago2.png")
    ax.imshow(img, extent=[-88.0, -87.5, 41.6, 42.0])
    plt.show()


# DRAW CALCULATED PRECINCTS
# Chicago has 22 police precincts, so hopefully
# clustering with 22 clusters should produce a similar map.
X = df.iloc[:, 1:3].to_numpy()
kmeans = KMeans(n_clusters=22)
kmeans.fit(X)

plt.style.use('dark_background')
fig, ax = plt.subplots()
ax.set_xlim([-88.0, -87.5])
ax.set_ylim([41.6, 42.0])

vor = Voronoi(kmeans.cluster_centers_)
voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_colors='black', line_width=1.0, ax=ax)

# Draw our calculated precincts over the map of Chicago.
# Use a hexbin to show a "heatmap" of crimes.
plt.title("Precincts")
plt.hexbin(X[:, 0], X[:, 1], gridsize=100, alpha=0.4)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='r')
img = plt.imread("chicago2.png")
ax.imshow(img, extent=[-88.0, -87.5, 41.6, 42.0], alpha=0.5)
plt.show()


