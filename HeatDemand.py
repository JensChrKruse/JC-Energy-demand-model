import numpy as np
import pandas as pd

# Custom libraries
import Setup as setup

def heat_demand(m, DF):

    # Variables
    HD_yearly = 1 # Yearly heat demand
    
    # Find columns with temperature clusters
    idx = DF.columns[DF.columns.str.startswith('Temp_cluster')]

    # Select temperature cluster columns and resample to daily mean temperatures
    df_temp = DF.loc[m.t_start:m.t_end, idx].resample('D').mean()

    # Set of years
    years = df_temp.index.year.unique()

    # Create an array to return to
    mean_demand = np.array([])
    yearly_deg_days = np.zeros(len(years))

    # Determine heat demand profile for each year
    for y in range(len(years)):
        temp_year = np.array(df_temp.loc[str(years[y])])

        T_thr = (temp_year < setup.T_set).astype(int) # Binary time series indicating if temperature is above threshold
        # Degree hour: the difference between the desired indoor temperature and the mean temperature in a given hour
        degDay = np.multiply((setup.T_desired-temp_year), T_thr)

        # Number of degree days per year
        deg_day_sum = np.sum(degDay, axis=0)

        # Find hourly demand as the scaled combination of the demand dependent and independent of ambient temperature
        HD_daily = HD_yearly * degDay / deg_day_sum * setup.TDD + HD_yearly / len(temp_year) * setup.TID

        # Population weights for each cluster
        weights = m.geo_points.groupby('Temp_cluster').sum()['Pup_weight'].values

        # Population weighted mean across clusters
        mean_demand = np.concatenate([mean_demand, np.sum(HD_daily * weights, axis=1)])
        yearly_deg_days[y] = np.sum(deg_day_sum * weights)
   
    # Make dataframe
    df_profile_daily = pd.DataFrame({'Heat demand': mean_demand}, index=df_temp.index)
    df_profile_hourly = df_profile_daily.resample('H').ffill() / 24
    df_deg_days = pd.DataFrame({'Number of degree days': np.round(yearly_deg_days,0)}, index=years)


    return df_profile_hourly, df_deg_days