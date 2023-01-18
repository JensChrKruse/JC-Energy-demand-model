# Libraries
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely as sh
from scipy.spatial import cKDTree

# Custom libraries
import Setup as setup

# Find the nearest neighbour to gdB in gdA
def ckd_nearest(gdA, gdB):
    nA = np.array(list(gdA.geometry.centroid.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.centroid.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf

def points_in_geo(area):
    # Import geometry
    gdf_NUTS = gpd.read_file(setup.NUTS_file)
    gdf_area = gdf_NUTS.loc[gdf_NUTS['NUTS_ID'] == area]

    # Import sampling grid points and make a geo dataframe
    df_points = pd.read_excel(setup.grid_file)
    gdf_points = gpd.GeoDataFrame(df_points, crs='epsg:4326', geometry=gpd.points_from_xy(df_points.Longitude, df_points.Latitude))

    # Find points within geometry
    gdf_intersect_points = gpd.sjoin(gdf_points, gdf_area, how='inner')

    df = pd.DataFrame(gdf_intersect_points[['Name', 'Latitude', 'Longitude']])
    df = df.set_index('Name')

    
    # Population data
    # Import population data 
    pup = gpd.read_file(setup.population_file)
    pup = pup.to_crs('epsg:4326')

    # Select population point samples inside area
    pup_area = gpd.sjoin(pup, gdf_area, how='inner')

    # Associate all population samples to nearest sampling point
    gdf_pup_points = ckd_nearest(pup_area.to_crs('epsg:7416'), gdf_intersect_points.to_crs('epsg:7416'))

    # Sum all population samples for each sampling point
    gdf_pup_points_sum = gdf_pup_points.groupby('Name').sum()

    # Put population weight into dataframe
    df['Pup_weight'] = gdf_pup_points_sum['TOT_P_2018'] / gdf_pup_points_sum['TOT_P_2018'].sum()
    

    return df

# Determine the number of clusters for a given area based on its latitudinal span
def n_clusters(area):
    # Import geometry
    gdf_NUTS = gpd.read_file(setup.NUTS_file)
    gdf_area = gdf_NUTS.loc[gdf_NUTS['NUTS_ID'] == area]

    # Span of country meased in degrees latitude
    latitude_range = float(gdf_area.bounds['maxy'] - gdf_area.bounds['miny'])

    # Number of clusters
    n = int(max(1, round(latitude_range*setup.clusters_per_degree_lat)))

    return n

def plot_clusters(m):
    # Import geometry
    gdf_NUTS = gpd.read_file(setup.NUTS_file)
    gdf_area = gdf_NUTS.loc[gdf_NUTS['NUTS_ID'] == m.area]

    # Import sampling grid points and make a geo dataframe
    gdf_points = gpd.GeoDataFrame(m.geo_points, crs='epsg:4326', geometry=gpd.points_from_xy(m.geo_points.Longitude, m.geo_points.Latitude))

    fig, ax = plt.subplots(1,1, figsize=(8,8))
    gdf_area.plot(ax=ax)
    gdf_points.plot(column='Temp_cluster', ax=ax, cmap='BuGn')
    ax.set_title(m.area + ' Clusters')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid(alpha=0.7)
    filename = 'Model validation/' + m.area + '/' + ' ' + m.area + ' Cluster map.png'
    plt.savefig(filename, dpi=300, transparent=False)
    plt.show()


# Import population geodata
#pup = gpd.read_file('Geodata/Eurostat/JRC_GRID_2018/JRC_POPULATION_2018.shp')

# Select country
#pup_DK = pup.query("CNTR_ID=='DK'")
#pup_DK.to_file("Geodata/Eurostat/JRC_GRID_2018/JRC_POPULATION_2018_DK.shp")




