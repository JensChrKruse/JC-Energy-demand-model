# Libraries
import numpy as np
import pandas as pd
import holidays
from suntimes import SunTimes

# Custom libraries
import Setup as setup
import ReadData as read
import Statistics as stats

# Holidays
# Create a time series of national holidays
def create_holiday_list(country, year):
    # Instantiate holiday object for country
    holidays_obj = holidays.country_holidays(country, years=year)
    holidays_obj.observed = False
    holiday_list = []

    for date, holiday_name in sorted(holidays_obj.items()): 
        holiday_list.append(holiday_name)
    
    # Exclude from holiday list
    df_custom_holidays = pd.read_excel(setup.custom_holidays, sheet_name='Exclude')
    excluded_holidays = df_custom_holidays.loc[(df_custom_holidays['Country'] == 'All') | (df_custom_holidays['Country'] == country), 'Holiday'].to_list()
    holiday_list = list(filter(lambda a: a not in excluded_holidays, holiday_list))

    return holiday_list

def holiday(m, time_range):
    # Instantiate holiday object for country
    if m.t_start != m.t_end:
        year_range = range(int(m.t_start), int(m.t_end))
    else:
        year_range = int(m.t_start)

    holidays_obj = holidays.country_holidays(m.country, years=year_range)
    holidays_obj.observed = False

    # Create dataframe to store holidays
    df_holidays = pd.DataFrame(index=time_range)
    df_holidays['Holiday'] = np.full(len(time_range), np.nan)

    # Get date of holiday based on names in list and insert into holidays dataframe
    for i in m.holiday_list:
        dates = holidays_obj.get_named(i)
        for j in dates:
            df_holidays.loc[(df_holidays.index.year == j.year) & (df_holidays.index.month == j.month) & (df_holidays.index.day == j.day)] = i
        
    # Custom holidays
    df_custom_holidays = pd.read_excel(setup.custom_holidays, sheet_name='Include')

    # Select universal and region specific holidays
    df_custom_holidays_region = df_custom_holidays.loc[(df_custom_holidays['Country'] == 'All') | (df_custom_holidays['Country'] == m.country)]

    # Put custom holidays into holidays dataframe
    for i in range(0, len(df_custom_holidays_region)):
        df_holidays.loc[(df_holidays.index.month == df_custom_holidays_region['Month'].iloc[i]) & (df_holidays.index.day == df_custom_holidays_region['Day'].iloc[i]), 'Holiday'] = df_custom_holidays_region['Holiday'].iloc[i]

    # Encode holidays as numerical categories for the random forest model
    if m.type[0] == 'RF':
        df_holidays['Holiday_cat'] = df_holidays['Holiday'].astype('category')
        cat_columns = df_holidays.select_dtypes(['category']).columns
        df_holidays['Holiday_cat'] = df_holidays[cat_columns].apply(lambda x: x.cat.codes)
        df_holidays = df_holidays.drop('Holiday', axis=1)


    return df_holidays

# Daylength
def daylength(m, time_range):
    daylength = np.zeros(len(time_range))

    longitude, latitude = m.geo_points['Longitude'].mean(), m.geo_points['Latitude'].mean()
    sun_obj = SunTimes(longitude, latitude, altitude=0)

    for t in range(0,len(time_range)):
        try:
            delta_sec = sun_obj.durationdelta(time_range[t])
            delta_hours = round(delta_sec.total_seconds()/3600, 2)
            daylength[t] = delta_hours
        except Exception: # Special case if daylength cannot be calculated due to polar position
            if delta_sec == 'Not calculable : PD':
                delta_hours = 24
                daylength[t] = delta_hours
            elif delta_sec == 'Not calculable : PN':
                delta_hours = 0
                daylength[t] = delta_hours
            pass
    
    df_daylength = pd.DataFrame({'Daylength': daylength}, index = time_range)


    return df_daylength


def request_exog_inputs(m):
    # Create dataframe to store exogenous variables
    df = pd.DataFrame()
    df['Time'] = pd.Series(pd.date_range(start=pd.Timestamp(f'{m.t_start}-01-01 00:00') - pd.DateOffset(hours=setup.t_offset), end=f'{m.t_end}-12-31 23:00', freq='H', tz='UTC'))
    df = df.set_index('Time')
    df['Const'] = 1
    
    # Create time variables
    df['Day'] = df.index.weekday
    df.loc[df['Day'] <= 4, 'Day'] = 1
    df.loc[df['Day'] == 5, 'Day'] = 2
    df.loc[df['Day'] == 6, 'Day'] = 3
    df['Hour'] = df.index.hour
    
    
    # Holidays
    df_holidays = holiday(m, df.index)
    df = df.join(df_holidays) 
    # Unassign the weekday of all holidays
    df.loc[df['Holiday'] == df['Holiday'].astype(str), 'Day'] = 0

    # Daylength
    df_daylength = daylength(m, df.index)
    #df_daylength_cluster = stats.mean_clustering(m, df_daylength, 'Daylength_cluster')
    df = df.join(df_daylength)
    
    '''
    # Irradiance
    df_irradiance = read.irradiance(m)
    df_irradiance_cluster = stats.mean_clustering(m, df_irradiance, 'Irr_cluster')
    df = df.join(df_irradiance_cluster)
    '''

    # Temperature
    df_temp = read.temperature(m)
    df_temp_cluster = stats.mean_clustering(m, df_temp, 'Temp_cluster')
    df = df.join(df_temp_cluster)
    # Backfill missing temperatures in hindcast if requested period exceeds dataset (due to setup.t_offset)
    idx = df.columns[df.columns.str.startswith('Temp_cluster')]
    df[idx] = df[idx].fillna(method='bfill')
    
    
    # Non-linearities for least squares models
    if m.type[0] == 'LSQ':
        
        # Piece wise function
        df_piecewise_temp1 = stats.piecewise(df, 'Temp', setup.T_breakpoint, 'BP_0')
        df = df.join(df_piecewise_temp1)

        # One-hot encoding of categorical variables
        df = pd.get_dummies(df, columns=['Day', 'Hour', 'Holiday'], prefix=['Day', 'Hour', 'Holiday'])
        
        # Interactions
        dummy_days = ['Day_', 'Daylength', 'Holiday']
        for i in range(0, len(dummy_days)):
            df = df.join(stats.dummy_interaction(df, dummy_days[i], 'Hour_'))
        

        
    # Allocate space for autoregressive terms in dynamic models
    if m.type[1] == 'Dynamic':
        df['Demand_Lag1'] = np.zeros(len(df.index))
        df['Demand_Lag2'] = np.zeros(len(df.index))

    return df



