# Libraries
import pandas as pd
import rioxarray
import xarray as xr
import numpy as np

# Custom libraries
import Setup as setup

# Import electricity demand data
def demand(m):
    file = 'demand_' + m.area + '.csv'
    df = pd.read_csv(file)
    df = df.set_index('Time') # Set time as index
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', utc=True) # Specify time structure
    
    # Select timeslice
    df_slice = df.loc[m.t_start:m.t_end]

    return df_slice

# Import electricity demand data
def demand_hindcast(region, type):
    file = setup.outputdatafolder + '/' + region + ' Hindcast_' + type + '.csv'
    df = pd.read_csv(file)
    df = df.set_index('Time') # Set time as index
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', utc=True) # Specify time structure

    return df

# Import electricity demand data
def degdays_hindcast(area):
    file = setup.outputdatafolder + '/' + area + ' deg_days.csv'
    df = pd.read_csv(file)
    df = df.set_index('Time') # Set time as index
    df.index = df.index.astype(str)

    return df

# Import ENTSOE hydro hindcast data
def hydro_hindcast_ENTSOE(area, type):
    dict = {
    'Reservoir': [53, 'Week'],
    'Pump storage - Open Loop': [53, 'Week'],
    'Run of River': [366, 'Day'],
    'Pondage': [366, 'Day']
    }
    
    file = 'Geodata/ENTSOE/PEMMDB_' + area + '_Hydro Inflow_2024.xlsx'
    df = pd.read_excel(file, sheet_name=type, skiprows=12, nrows=dict[type][0], usecols='Q:BA', index_col=0, header=0)
    df.index.names = [dict[type][1]]
    df.columns = np.arange(1982,2018).astype(str)
    df = df.fillna(0)

    capacity = pd.read_excel(file, sheet_name=type, skiprows=4, nrows=1, usecols='C').fillna(0).values[0][0]

    # Only output if both inflow profile and capacity is available
    if (df.sum().sum() == 0) or (capacity == 0):
        df[:] = 0
        capacity = 0

    return df, capacity

def IT_hydro_hindcast_ENTSOE(type):
    df_ITCA, capacity_ITCA = hydro_hindcast_ENTSOE('ITCA', type)
    df_ITCN, capacity_ITCN = hydro_hindcast_ENTSOE('ITCN', type)
    df_ITCS, capacity_ITCS = hydro_hindcast_ENTSOE('ITCS', type)
    df_ITN1, capacity_ITN1 = hydro_hindcast_ENTSOE('ITN1', type)
    df_ITS1, capacity_ITS1 = hydro_hindcast_ENTSOE('ITS1', type)
    df_ITSA, capacity_ITSA = hydro_hindcast_ENTSOE('ITSA', type)
    df_ITSI, capacity_ITSI = hydro_hindcast_ENTSOE('ITSI', type)

    dfs = [df_ITCA, df_ITCN, df_ITCS, df_ITN1, df_ITS1, df_ITSA, df_ITSI]

    df = pd.concat(dfs).sum(level=0)
    capacity = capacity_ITCA + capacity_ITCN + capacity_ITCS + capacity_ITN1 + capacity_ITS1 + capacity_ITSA + capacity_ITSI

    df_res, cap_res = IT_hydro_hindcast_ENTSOE('Reservoir')
    df_ph, cap_ph = IT_hydro_hindcast_ENTSOE('Pump storage - Open Loop')
    df_RR, cap_RR = IT_hydro_hindcast_ENTSOE('Run of River')

    df_RR.to_excel('IT00_RR.xlsx')
    df_res.to_excel('IT00_res.xlsx')
    df_ph.to_excel('IT00_ph.xlsx')

    return df, capacity



# Import Copernicus hydro hindcast data
def hydro_hindcast_Copernicus(type):
    file = 'Geodata/Copernicus/Copernicus_Hydro_' + type + '.csv'
    df = pd.read_csv(file, skiprows=61, index_col=0)
    df.index.names = ['Day']
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', utc=True)
    df.columns = df.columns.astype(str)
    df = df.fillna(0)

    return df

def production_ENTSOE(region):
    file = 'ENTSOE energy production/Actual Generation per Production Type_201601010000-201701010000_' + region + '.csv'
    df = pd.read_csv(file, index_col=1)
    df = df.drop(['Area'], axis=1)
    df = df.replace(to_replace='n/e', value=0)

    df_sum = df.sum() / 10**6 / (len(df) / 8785)

    return df_sum

def day_ahead_prices_ENTSOE(region):
    file = 'ENTSOE day-ahead prices/Day-ahead Prices_201601010000-201701010000_' + region + '.csv'
    df = pd.read_csv(file, index_col=0)
    df = df.iloc[:,0]
    df = df.replace(to_replace='n/e', value=0)

    df_mean = df.mean()

    return df_mean


# Import Copernicus power hindcast data
def power_hindcast_Copernicus():
    file = 'Geodata/Copernicus/Copernicus_Power_Demand.csv'
    df = pd.read_csv(file, skiprows=53, index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d', utc=True)

    return df


def filter_demand_area():
    area = 'LU'
    #df = pd.read_excel('OPSD_time_series.xlsx', sheet_name='60min')

    df_filtered = df.filter(regex=area, axis=1)
    df_filtered.columns = df_filtered.iloc[1]

    df_area = pd.DataFrame()
    df_area['Time'] = df.iloc[6:,0]
    df_area['Demand'] = df_filtered.iloc[6:,7]
    df_area = df_area.set_index('Time') # Set time as index
    df_area.index = pd.to_datetime(df_area.index, format='%Y-%m-%d %H:%M:%S', utc=True) # Specify time structure

    # Export CSV
    filename = 'demand_' + area + '.csv'
    df_area.to_csv(filename)

    return df_area

# Import CorRES data
def corres(filename):
    df = pd.read_csv(setup.corres_folder + filename, sep=",")
    df = df.set_index('Time') # Set time as index
    df.index = pd.to_datetime(df.index, format='%d-%b-%Y %H:%M:%S', utc=True) # Specify time structure
    df = df.replace(to_replace=np.NAN, value=0)
    return df


def weather(infile, points):
    # Import
    file = 'Geodata/ERA5/Weather_100km_Europe/' + infile
    ds = rioxarray.open_rasterio(file)

    # From ordinal time to datetime
    timeoffset = np.datetime64('1979-01-01 00:00:00') - np.datetime64('1970-01-01 00:00:00')
    datenum = timeoffset.astype('int64') + ds.y.values*60*60
    time = np.array(datenum, dtype='datetime64[s]').astype(object)

    # Select temperatures
    Temp = ds.T2.sel(band=1).values - 273.15
    Temp_names = [str(x) for x in points]

    # Select wind speed
    #WS = ds.WS.sel(x=geo_points, band=1).values
    #WS_names = ['WS_' + str(x) for x in ds.x.sel(x=geo_points).values.astype(int)]
    
    # Create dataframe
    df = pd.DataFrame(Temp, columns=Temp_names)
    #df = pd.DataFrame(WS, columns=WS_names)

    # Set index 
    df['Time'] = time
    df = df.set_index('Time') # Set time as index
    df.index = pd.to_datetime(df.index, format='%d-%b-%Y %H:%M:%S', utc=True) # Specify time structure
    df = df.sort_index()
    
    #outfile = 'temperature_100km_' + str(num) + '.csv'
    
    #df.to_csv(outfile)
    return df

def request_weather():
    infile = ['ds_meso_000.nc', 'ds_meso_001.nc', 'ds_meso_002.nc', 'ds_meso_003.nc', 'ds_meso_004.nc', 'ds_meso_005.nc']
    points = [range(0,87), range(87,173), range(173,259), range(0+259, 84+259), range(84+259, 167+259), range(167+259, 250+259)]
    df = weather(infile[0], points[0])
    for i in range(1, len(infile)):
        print(i)
        df = df.join(weather(infile[i], points[i]))

    df.to_csv('temperature_all.csv')

def request_irr():
    # Irradiance
    file = 'Geodata/ERA5/Irradiance_All_1982-2020/GHI_' + str(1) + '.csv'
    df = pd.read_csv(file)
    df = df.set_index('time') # Set time as index
    
    for i in range(2,7):
        print(i)
        file = 'Geodata/ERA5/Irradiance_All_1982-2020/GHI_' + str(i) + '.csv'
        df2 = pd.read_csv(file)
        df2 = df2.set_index('time') # Set time as index
        df = df.join(df2)
    
    df = df.rename(columns={"time":"Time"})
    df = df.set_index('time') # Set time as index
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', utc=True)
    df.to_csv('GHI_all.csv')

def irradiance(m):
    # Irradiance
    file = 'Geodata/ERA5/Irradiance_All_1982-2020/GHI_all.csv'
    df = pd.read_csv(file)
    df = df.rename(columns={"time":"Time"})
    df = df.set_index('Time') # Set time as index
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', utc=True)

    # Select columns with the specified area and rows with specified timeslice
    col = m.geo_points.index.astype(str)
    df_slice = df[col].loc[m.t_start:m.t_end]
    df_slice = df_slice.add_prefix('Irr_')

    df_slice = df_slice.resample('24H').sum()
    df_slice = df_slice.resample('H').ffill()
    '''
    # Population weighted mean
    weights = m.geo_points['Pup_weight'].values
    for i in range(0,len(weights)):
        df_slice.iloc[:,i] = df_slice.iloc[:,i] * weights[i]
    df_slice_mean = pd.DataFrame({'Irr': df_slice.sum(axis=1)}, index=df_slice.index)
    '''

    #df_slice1 = df_slice.rolling(6, center=True).mean()
    #df_slice2 = pd.DataFrame({'Irr': df_slice.mean(axis=1)}, index=df_slice.index)

    return df_slice

def temperature(m):
    file = setup.temperature_file
    df = pd.read_csv(file)
    df = df.set_index('Time') # Set time as index
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', utc=True)

    col = m.geo_points.index.astype(str)
    start = pd.Timestamp(f'{m.t_start}-01-01 00:00', tz='UTC') - pd.DateOffset(hours=setup.t_offset)
    end = pd.Timestamp(f'{m.t_end}-12-31 23:00', tz='UTC')
    df_slice = df[col].loc[start:end]
    df_slice = df_slice.add_prefix('Temp_')

    return df_slice


def windspeed(m):
    file = 'windspeed_data_' + m.area + '.csv'
    df = pd.read_csv(file)
    df = df.rename(columns={"time":"Time"})
    df = df.set_index('Time') # Set time as index
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', utc=True)

    #col = m.geo_points.index.astype(str)
    df_slice = df.loc[m.t_start:m.t_end]

    return df_slice

def precip(m):
    year = [2015,2016,2017,2018,2019]
    df = pd.read_csv('Precipitation_DK/hele-landet-' + str(year[0]) +'.csv', sep=';', index_col=0, decimal=',')
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', utc=True)
    for i in range(1,len(year)):
        df_new = pd.read_csv('Precipitation_DK/hele-landet-' + str(year[i]) +'.csv', sep=';', index_col=0, decimal=',')
        df_new.index = pd.to_datetime(df_new.index, format='%Y-%m-%d %H:%M:%S', utc=True)
        df = pd.concat([df, df_new])

    df_upsample = df['Nedbør'].resample('H').ffill()
    df_slice = df_upsample.loc[m.t_start:m.t_end]

    return df_slice
