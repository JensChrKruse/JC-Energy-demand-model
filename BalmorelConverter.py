# Libraries
import numpy as np
import pandas as pd
import os

# Custom libraries
import Setup as setup
import ReadData as read

# Working directory
os.chdir(setup.datafolder)

# Balmorel time sets and geography
Bal_TS = pd.read_excel('Balmorel_time_sets.xlsx')
Bal_years = pd.read_excel('Bal_years.xlsx', index_col=0)
Bal_years.index = Bal_years.index.astype(str)
model_geo = pd.read_excel('Model geography.xlsx', sheet_name=['Demand_regions', 'Hydro_areas', 'Solar_areas', 'Onshore_areas', 'Offshore_areas', 'Onshore_areas_SP277_HH100_RGB', 'Offshore_areas_SP370_HH155'])
Bal_regions = model_geo['Demand_regions']['Balmorel region code'].values
Bal_solar_areas = model_geo['Solar_areas']['Balmorel area'].values
Bal_hydro_areas = model_geo['Hydro_areas']['Balmorel area'].values

# Wind type
wind_type = 'Future'

if wind_type == 'Existing':
    onshore_dict = model_geo['Onshore_areas'][['Balmorel area', 'CorRES area 1', 'CorRES area 2', 'CorRES area 3', 'CorRES area 4']].set_index('Balmorel area').T.to_dict('list')
    offshore_dict = model_geo['Offshore_areas'][['Balmorel area', 'CorRES area 1']].set_index('Balmorel area').T.to_dict('list')
    Bal_onshore_areas = model_geo['Onshore_areas']['Balmorel area'].values
    Bal_offshore_areas = model_geo['Offshore_areas']['Balmorel area'].values
    Bal_wind_areas = np.concatenate([Bal_onshore_areas, Bal_offshore_areas])
elif wind_type == 'Future':
    onshore_dict = model_geo['Onshore_areas_SP277_HH100_RGB'][['Balmorel area', 'CorRES area 1', 'CorRES area 2', 'CorRES area 3', 'CorRES area 4']].set_index('Balmorel area').T.to_dict('list')
    offshore_dict = model_geo['Offshore_areas_SP370_HH155'][['Balmorel area', 'CorRES area 1', 'CorRES area 2']].set_index('Balmorel area').T.to_dict('list')
    Bal_onshore_areas = model_geo['Onshore_areas_SP277_HH100_RGB']['Balmorel area'].values
    Bal_offshore_areas = model_geo['Offshore_areas_SP370_HH155']['Balmorel area'].values
    Bal_wind_areas = np.concatenate([Bal_onshore_areas, Bal_offshore_areas])

# Make dictionaries
demand_dict = model_geo['Demand_regions'][['Balmorel region code', 'ENTSOE region code']].set_index('Balmorel region code').T.to_dict('list')
solar_dict = model_geo['Solar_areas'][['Balmorel area', 'CorRES area 1', 'CorRES area 2', 'CorRES area 3', 'CorRES area 4']].set_index('Balmorel area').T.to_dict('list')
hydro_dict = model_geo['Hydro_areas'][['Balmorel area', 'ENTSOE area']].set_index('Balmorel area').T.to_dict('list')

# Multicolumn indices
CL_years = Bal_TS['CL_years'].dropna().astype(int).astype(str).values # Climate years

# Multirow indices
S = Bal_TS['S'].dropna().values
T = Bal_TS['T'].dropna().values

# Set shapes
n_Y = len(CL_years) # Number of climate years
n_T = len(T) # Number of time steps per week
n_S = len(S) # Number of seasons
n_R = len(Bal_regions) # Number of regions
n_A_solar = len(Bal_solar_areas) # Number of onshore areas
n_A_onshore = len(Bal_onshore_areas) # Number of onshore areas
n_A_offshore = len(Bal_offshore_areas) # Number of onshore areas
n_A_hydro = len(Bal_hydro_areas) # Number of hydro areas

# Create excel file with start and end timestamps of Balmorel years
def create_Bal_years(DF):
    # Empty dataframe
    df = pd.DataFrame({'t_start': np.zeros(len(CL_years)), 't_end': np.zeros(len(CL_years))}, index=CL_years)

    for y in CL_years:
        # Find index of first monday of the year
        idx_offset = np.where(np.logical_and(DF.index.weekday == 0, DF.index.year == int(y)))[0][0]

        # If the year is the last in the data set, the Balmorel year is started 1 week before to ensure enough data points are available
        if y == CL_years[-1]:
            idx_offset = idx_offset - 168

        df.loc[y,'t_start'] = DF.index[idx_offset].strftime(format='%Y-%m-%d %H:%M:%S')
        df.loc[y,'t_end'] = DF.index[idx_offset+(n_S*n_T)-1].strftime(format='%Y-%m-%d %H:%M:%S')

    # Export
    df.to_excel('Bal_years.xlsx')

# Slice away entries such that dataframe starts on a monday and has the length of a Balmorel year using the Bal_years excel file
def Bal_year_slice(DF, y):
    # Slice year
    df_slice = DF.loc[Bal_years.loc[y,'t_start']:Bal_years.loc[y,'t_end']]

    return df_slice

#---------------------------------- Classic electricity demand ---------------------------------- #

# Generate electricity demand profiles
def create_DE_VAR_T_CLASSIC_CLY():
    # Initialize dataframe
    mulcol = pd.MultiIndex.from_product([CL_years, Bal_regions], names=["Year", "Region"])
    mulrow = pd.MultiIndex.from_product([S, T], names=["Season", "Time"])
    df = pd.DataFrame(np.zeros((n_S*n_T, n_Y*n_R)), columns=mulcol, index=mulrow)

    # For all areas
    for r in Bal_regions:
        # Read in hindcast file
        df_r_classic = read.demand_hindcast(demand_dict[r][0], 'classic')

        # For all climate years
        for y in CL_years:
            df[y, r] = (Bal_year_slice(df_r_classic, y) / df_r_classic.max()).values # Normalized consumption

    df.replace(to_replace=0, value='EPS').transpose().to_excel('DE_VAR_T_CLASSIC_CLY.xlsx')

    return df


# Generate climate year electricity demand level factors
def create_DE_CLASSIC_CLY_FACTOR():
    # Initialize dataframe
    mulcol = pd.MultiIndex.from_product([Bal_regions], names=["Region"])
    mulrow = pd.MultiIndex.from_product([CL_years], names=["Year"])
    df = pd.DataFrame(np.zeros((n_Y, n_R)), columns=mulcol, index=mulrow)

    # For all areas
    for r in Bal_regions:
        # Read in hindcast file
        df_r_classic = read.demand_hindcast(demand_dict[r][0], 'classic')

        # For all climate years
        for y in CL_years:
            df.loc[y, r] = df_r_classic.loc[y].sum().values

    # Calculate climate year correction factor (deviation from long term mean for each region)
    df_CL_factor = df / df.mean()

    df_CL_factor.replace(to_replace=0, value='EPS').to_excel('DE_CLASSIC_CLY_FACTOR.xlsx')

    return df_CL_factor


#---------------------------------- Heat demand ---------------------------------- #

# Generate heat demand profiles
def create_DH_VAR_T_CLY():
    # Initialize dataframe
    mulcol = pd.MultiIndex.from_product([CL_years, Bal_regions], names=["Year", "Region"])
    mulrow = pd.MultiIndex.from_product([S, T], names=["Season", "Time"])
    df = pd.DataFrame(np.zeros((n_S*n_T, n_Y*n_R)), columns=mulcol, index=mulrow)

    # For all areas
    for r in Bal_regions:
        # Read in hindcast file
        df_r_heat = read.demand_hindcast(demand_dict[r][0], 'heat')

        # For all climate years
        for y in CL_years:
            df[y, r] = Bal_year_slice(df_r_heat, y).values
    
    df.replace(to_replace=0, value='EPS').transpose().to_excel('DH_VAR_T_CLY.xlsx')

    return df


# Generate climate year heat demand level factors
def create_DH_CLY_FACTOR():
    # Initialize dataframe
    mulcol = pd.MultiIndex.from_product([Bal_regions], names=["Region"])
    mulrow = pd.MultiIndex.from_product([CL_years], names=["Year"])
    df = pd.DataFrame(np.zeros((n_Y, n_R)), columns=mulcol, index=mulrow)

    # For all areas
    for r in Bal_regions:
        # Read in hindcast file
        df_r = read.degdays_hindcast(demand_dict[r][0])

        # For all climate years
        # Calculate climate year correction factor (deviation from long term mean)
        for y in CL_years:
            df.loc[y, r] = (df_r.loc[y] / df_r.mean()).values
    
    # The climate correction factors only applies to the fraction of the heat demand that is temperature dependent
    df_factor = df * setup.TDD + setup.TID

    df_factor.replace(to_replace=0, value='EPS').to_excel('DH_CLY_factor.xlsx')

    return df_factor


#---------------------------------- Solar PV production ---------------------------------- #

# Generate solar pv generation profiles
def create_SOLE_VAR_T_CLY():
    df_solarpv = read.corres('Solar PV/solarPV.csv')
    
    # Initialize dataframe
    mulcol = pd.MultiIndex.from_product([CL_years, Bal_solar_areas], names=["Year", "Area"])
    mulrow = pd.MultiIndex.from_product([S, T], names=["Season", "Time"])
    df = pd.DataFrame(np.zeros((n_S*n_T, n_Y*n_A_solar)), columns=mulcol, index=mulrow)

    # For all onshore areas
    for a in Bal_solar_areas:
        CorRES_areas = [x for x in solar_dict[a] if pd.notnull(x)] # Find valid (no nans) CorRES areas based on Balmorel area
        df_solarpv_area = df_solarpv[CorRES_areas].mean(axis=1)
        # For all climate years
        for y in CL_years:
            df[y, a] = Bal_year_slice(df_solarpv_area, y).values
    
    df.replace(to_replace=0, value='EPS').transpose().to_excel('SOLE_VAR_T_CLY.xlsx')

    return df

# Generate solar pv full load hours
def create_SOLE_FLH_CLY():
    df_solarpv = read.corres('Solar PV/solarPV.csv')

    # Initialize dataframe
    mulcol = pd.MultiIndex.from_product([Bal_solar_areas], names=["Area"])
    mulrow = pd.MultiIndex.from_product([CL_years], names=["Year"])
    df = pd.DataFrame(np.zeros((n_Y, n_A_solar)), columns=mulcol, index=mulrow)

    # For all onshore areas
    for a in Bal_solar_areas:
        CorRES_areas = [x for x in solar_dict[a] if pd.notnull(x)] # Find valid (no nans) CorRES areas based on Balmorel area
        df_solarpv_area = df_solarpv[CorRES_areas].mean(axis=1)
        
        for y in CL_years:
            df.loc[y,a] = df_solarpv_area[y].sum()

    df.round(0).astype(int).replace(to_replace=0, value='EPS').to_excel('SOLE_FLH_CLY.xlsx')

    return df


#---------------------------------- Wind production ---------------------------------- #


# Generate wind generation profiles
def create_WND_VAR_T_CLY():
    if wind_type == 'Existing':
        df_onshore = read.corres('Onshore wind Existing/Existing.csv')
        df_offshore = read.corres('Offshore wind Existing/Existing.csv')
    elif wind_type == 'Future':
        df_onshore = read.corres('Onshore wind SP277 HH150 RGB/SP277_HH150_B.csv')
        df_offshore = read.corres('Offshore wind SP370 HH155/SP370.csv')

    # Initialize dataframe
    mulcol = pd.MultiIndex.from_product([CL_years, Bal_wind_areas], names=["Year", "Area"])
    mulrow = pd.MultiIndex.from_product([S, T], names=["Season", "Time"])
    df = pd.DataFrame(np.zeros((n_S*n_T, n_Y*(n_A_onshore+n_A_offshore))), columns=mulcol, index=mulrow)

    # For all onshore areas
    for a in Bal_onshore_areas:
        CorRES_areas = [x for x in onshore_dict[a] if pd.notnull(x)] # Find valid (no nans) CorRES areas based on Balmorel area
        df_onshore_area = df_onshore[CorRES_areas].mean(axis=1)
        # For all climate years
        for y in CL_years:
            df[y, a] = Bal_year_slice(df_onshore_area, y).values
    
    # For all offshore areas
    for a in Bal_offshore_areas:
        CorRES_areas = [x for x in offshore_dict[a] if pd.notnull(x)] # Find valid (no nans) CorRES areas based on Balmorel area
        df_offshore_area = df_offshore[CorRES_areas].mean(axis=1)
        # For all climate years
        for y in CL_years:
            df[y, a] = Bal_year_slice(df_offshore_area, y).values

    # Use Eps value instead of 0 in GAMS
    df = df.replace(to_replace=0, value='EPS')

    df.transpose().to_excel('WND_VAR_T_CLY_FUTURE.xlsx')

    return df


# Generate wind full load hours
def create_WND_FLH_CLY():
    if wind_type == 'Existing':
        df_onshore = read.corres('Onshore wind Existing/Existing.csv')
        df_offshore = read.corres('Offshore wind Existing/Existing.csv')
    elif wind_type == 'Future':
        df_onshore = read.corres('Onshore wind SP277 HH150 RGB/SP277_HH150_B.csv')
        df_offshore = read.corres('Offshore wind SP370 HH155/SP370.csv')

    # Initialize dataframe
    mulcol = pd.MultiIndex.from_product([Bal_wind_areas], names=["Area"])
    mulrow = pd.MultiIndex.from_product([CL_years], names=["Year"])
    df = pd.DataFrame(np.zeros((n_Y, (n_A_onshore+n_A_offshore))), columns=mulcol, index=mulrow)

    # For all onshore areas
    for a in Bal_onshore_areas:
        CorRES_areas = [x for x in onshore_dict[a] if pd.notnull(x)] # Find valid (no nans) CorRES areas based on Balmorel area
        df_onshore_area = df_onshore[CorRES_areas].mean(axis=1)
        for y in CL_years:
            df.loc[y,a] = df_onshore_area[y].sum()
    
    # For all offshore areas
    for a in Bal_offshore_areas:
        CorRES_areas = [x for x in offshore_dict[a] if pd.notnull(x)] # Find valid (no nans) CorRES areas based on Balmorel area
        df_offshore_area = df_offshore[CorRES_areas].mean(axis=1)
        for y in CL_years:
            df.loc[y,a] = df_offshore_area[y].sum()

    df.round(0).astype(int).replace(to_replace=0, value='EPS').to_excel('WND_FLH_CLY_FUTURE.xlsx')

    return df


#---------------------------------- Reservoir hydro production ---------------------------------- #


# Generate reservoir hydro water inflow profiles
def create_WTRRSVAR_S_CLY():
    # Initialize dataframe
    mulcol = pd.MultiIndex.from_product([CL_years, Bal_hydro_areas], names=["Year", "Region"])
    mulrow = pd.MultiIndex.from_product([S], names=["Season"])
    df = pd.DataFrame(np.zeros((n_S, n_Y*n_A_hydro)), columns=mulcol, index=mulrow)

    # For all areas
    for a in Bal_hydro_areas:
        # Read in hindcast file
        df_hydro_area1, capacity1 = read.hydro_hindcast_ENTSOE(hydro_dict[a][0], 'Reservoir')
        df_hydro_area2, capacity2 = read.hydro_hindcast_ENTSOE(hydro_dict[a][0], 'Pump storage - Open Loop')

        df_hydro_area = df_hydro_area1 + df_hydro_area2
        capacity = capacity1 + capacity2

        # For all climate years
        for y in CL_years:
            if capacity > 0:
                df[y,a] = df_hydro_area[y].values[0:len(df[y,a])] * 10**3 / (capacity * n_T) # Normalize profile
            else:
                df[y,a] = 0
    
    # ENTSOE models all NO hydro as res, but some are ror. Scale down inflows accordingly
    for y in CL_years:
        df[y,'NO_N_A']  = df[y,'NO_N_A'] * 0.9
        df[y,'NO_M_A']  = df[y,'NO_M_A'] * 0.9
        df[y,'NO_MW_A']  = df[y,'NO_MW_A'] * 0.9
        df[y,'NO_SE_A']  = df[y,'NO_SE_A'] * 0.9
        df[y,'NO_SW_A']  = df[y,'NO_SW_A'] * 0.9
        df[y,'SE_N1_A']  = df[y,'SE_N1_A'] * 0.9
        df[y,'SE_N2_A']  = df[y,'SE_N2_A'] * 0.9
        df[y,'SE_M_A']  = df[y,'SE_M_A'] * 0.9
        df[y,'SE_S_A']  = df[y,'SE_S_A'] * 0.9
        df[y,'FR_A']  = df[y,'FR_A'] * 1.2

    # Remove areas with no profiles
    df = df.loc[:, (df != 0).any(axis=0)]

    df.replace(to_replace=0, value='EPS').transpose().to_excel('WTRRSVAR_S_CLY.xlsx')

    return df

# Generate reservoir hydro full load hours
def create_WTRRSFLH_CLY():
    # Initialize dataframe
    mulcol = pd.MultiIndex.from_product([Bal_hydro_areas], names=["Area"])
    mulrow = pd.MultiIndex.from_product([CL_years], names=["Year"])
    df = pd.DataFrame(np.zeros((n_Y, n_A_hydro)), columns=mulcol, index=mulrow)

    # For all areas
    for a in Bal_hydro_areas:
        # Read in hindcast file
        df_hydro_area1, capacity1 = read.hydro_hindcast_ENTSOE(hydro_dict[a][0], 'Reservoir')
        df_hydro_area2, capacity2 = read.hydro_hindcast_ENTSOE(hydro_dict[a][0], 'Pump storage - Open Loop')

        df_hydro_area = df_hydro_area1 + df_hydro_area2
        capacity = capacity1 + capacity2

        # For all climate years
        for y in CL_years:
            if capacity > 0:
                df.loc[y,a] = df_hydro_area[y].sum() * 10**3 / capacity
            else:
                df.loc[y,a] = 0

    # Calibrate to reach annual level
    for y in CL_years:
        df.loc[y,'NO_N_A']  = df.loc[y,'NO_N_A'] * 0.9
        df.loc[y,'NO_M_A']  = df.loc[y,'NO_M_A'] * 0.9
        df.loc[y,'NO_MW_A']  = df.loc[y,'NO_MW_A'] * 0.9
        df.loc[y,'NO_SE_A']  = df.loc[y,'NO_SE_A'] * 0.9
        df.loc[y,'NO_SW_A']  = df.loc[y,'NO_SW_A'] * 0.9
        df.loc[y,'SE_N1_A']  = df.loc[y,'SE_N1_A'] * 0.9
        df.loc[y,'SE_N2_A']  = df.loc[y,'SE_N2_A'] * 0.9
        df.loc[y,'SE_M_A']  = df.loc[y,'SE_M_A'] *  0.9
        df.loc[y,'SE_S_A']  = df.loc[y,'SE_S_A'] * 0.9
        df.loc[y,'FR_A']  = df.loc[y,'FR_A'] * 1.2
    

    # Remove areas with no profiles
    df = df.loc[:, (df != 0).any(axis=0)]

    df.round(0).astype(int).replace(to_replace=0, value='EPS').to_excel('WTRRSFLH_CLY.xlsx')

    return df


#---------------------------------- Run-of-river hydro production ---------------------------------- #

# Generate run of river hydro water inflow profiles
def create_WTRRRVAR_T_CLY():
    # Initialize dataframe
    mulcol = pd.MultiIndex.from_product([CL_years, Bal_hydro_areas], names=["Year", "Region"])
    mulrow = pd.MultiIndex.from_product([S, T], names=["Season", 'Time'])
    df = pd.DataFrame(np.zeros((n_S*n_T, n_Y*n_A_hydro)), columns=mulcol, index=mulrow)

    # For all areas
    for a in Bal_hydro_areas:
        # Read in hindcast file
        df_hydro_area1, capacity1 = read.hydro_hindcast_ENTSOE(hydro_dict[a][0], 'Run of River')
        df_hydro_area2, capacity2 = read.hydro_hindcast_ENTSOE(hydro_dict[a][0], 'Pondage')
        
        df_hydro_area = df_hydro_area1 + df_hydro_area2
        capacity = capacity1 + capacity2

        time_range = pd.date_range(start=f'{CL_years[0]}-01-01', end=f'{CL_years[-1]}-12-31', freq='D')
        df_years = pd.DataFrame({'Inflow': np.zeros(len(time_range))}, index=time_range)

        # For all climate years
        for y in CL_years:
            df_years.loc[y, 'Inflow'] = df_hydro_area[y].values[0:len(df_years.loc[y])]
        
        # Upsample to hourly resolution
        df_hourly = df_years.resample('H').interpolate(method='linear').fillna(method='ffill')/24

        # For all climate years
        for y in CL_years:
            if capacity > 0:
                df[y,a] = Bal_year_slice(df_hourly, y).values * 10**3 / (capacity) # Normalize profile
            else:
                df[y,a] = 0


    # Fill in remaining using reservoir inflows as an approximation
    for a in ['SE_N1_A','SE_N2_A','SE_M_A','SE_S_A','FI_A','NO_N_A','NO_M_A','NO_MW_A','NO_SE_A','NO_SW_A']:
        df_hydro_area1, capacity1 = read.hydro_hindcast_ENTSOE(hydro_dict[a][0], 'Reservoir')
        df_hydro_area2, capacity2 = read.hydro_hindcast_ENTSOE(hydro_dict[a][0], 'Pump storage - Open Loop')

        df_hydro_area = df_hydro_area1 + df_hydro_area2
        capacity = capacity1 + capacity2

        time_range = pd.date_range(start=f'1981-01-01', end=f'2017-12-31', freq='W')
        df_years = pd.DataFrame({'Inflow': np.full(len(time_range),np.nan)}, index=time_range)
        
        # For all climate years
        for y in CL_years:
            df_years.loc[y, 'Inflow'] = df_hydro_area[y].values[0:len(df_years.loc[y])]

        # Upsample to hourly resolution
        df_hourly = df_years.resample('H').interpolate(method='linear').fillna(method='bfill')/n_T

        # For all climate years
        for y in CL_years:
            if capacity > 0:
                df[y,a] = (Bal_year_slice(df_hourly, y).values * 10**3 / capacity).clip(max=1) # Normalize profile and clip to 1, such that production never exceeds capacity
            else:
                df[y,a] = 0

    # Calibrate to reach annual level
    for y in CL_years:
        df[y,'NO_N_A']  = df[y,'NO_N_A'] * 0.9
        df[y,'NO_M_A']  = df[y,'NO_M_A'] * 0.9
        df[y,'NO_MW_A']  = df[y,'NO_MW_A'] * 0.9
        df[y,'NO_SE_A']  = df[y,'NO_SE_A'] * 0.9
        df[y,'NO_SW_A']  = df[y,'NO_SW_A'] * 0.9
        df[y,'SE_N1_A']  = df[y,'SE_N1_A'] * 0.9
        df[y,'SE_N2_A']  = df[y,'SE_N2_A'] * 0.9
        df[y,'SE_M_A']  = df[y,'SE_M_A'] * 0.9
        df[y,'SE_S_A']  = df[y,'SE_S_A'] * 0.9
        df[y,'FR_A']  = df[y,'FR_A'] * 1.2

    # Remove areas with no profiles
    df = df.loc[:, (df != 0).any(axis=0)]
    
    # Export
    df.replace(to_replace=0, value='EPS').transpose().to_excel('WTRRRVAR_T_CLY.xlsx')

    return df

# Generate run of river hydro full load hours
def create_WTRRRFLH_CLY():
    # Initialize dataframe
    mulcol = pd.MultiIndex.from_product([Bal_hydro_areas], names=["Region"])
    mulrow = pd.MultiIndex.from_product([CL_years], names=["Year"])
    df = pd.DataFrame(np.zeros((n_Y, n_A_hydro)), columns=mulcol, index=mulrow)

    # Fill in using hindcast data from ENTSOE for all areas
    for a in Bal_hydro_areas:
        df_hydro_area1, capacity1 = read.hydro_hindcast_ENTSOE(hydro_dict[a][0], 'Run of River')
        df_hydro_area2, capacity2 = read.hydro_hindcast_ENTSOE(hydro_dict[a][0], 'Pondage')
        
        df_hydro_area = df_hydro_area1 + df_hydro_area2
        capacity = capacity1 + capacity2

        # For all climate years
        for y in CL_years:
            if capacity > 0:
                df.loc[y,a] = df_hydro_area[y].sum() * 10**3 / capacity
            else:
                df.loc[y,a] = 0

    # Fill in remaining using reservoir inflows as an approximation
    for a in ['SE_N1_A','SE_N2_A','SE_M_A','SE_S_A','FI_A','NO_N_A','NO_M_A','NO_MW_A','NO_SE_A','NO_SW_A']:
        df_hydro_area1, capacity1 = read.hydro_hindcast_ENTSOE(hydro_dict[a][0], 'Reservoir')
        df_hydro_area2, capacity2 = read.hydro_hindcast_ENTSOE(hydro_dict[a][0], 'Pump storage - Open Loop')

        df_hydro_area = df_hydro_area1 + df_hydro_area2
        capacity = capacity1 + capacity2

        for y in CL_years:
            if capacity > 0:
                df.loc[y,a] = (df_hydro_area[y] * 10**3 / (capacity * n_T)).clip(upper=1).sum() * n_T
            else:
                df.loc[y,a] = 0

    # Calibrate to reach annual level
    for y in CL_years:
        df.loc[y,'NO_N_A']  = df.loc[y,'NO_N_A'] * 0.9
        df.loc[y,'NO_M_A']  = df.loc[y,'NO_M_A'] * 0.9
        df.loc[y,'NO_MW_A']  = df.loc[y,'NO_MW_A'] * 0.9
        df.loc[y,'NO_SE_A']  = df.loc[y,'NO_SE_A'] * 0.9
        df.loc[y,'NO_SW_A']  = df.loc[y,'NO_SW_A'] * 0.9
        df.loc[y,'SE_N1_A']  = df.loc[y,'SE_N1_A'] * 0.9
        df.loc[y,'SE_N2_A']  = df.loc[y,'SE_N2_A'] * 0.9
        df.loc[y,'SE_M_A']  = df.loc[y,'SE_M_A'] * 0.9
        df.loc[y,'SE_S_A']  = df.loc[y,'SE_S_A'] * 0.9
        df.loc[y,'FR_A']  = df.loc[y,'FR_A'] * 1.2

    # Remove areas with no profiles
    df = df.loc[:, (df != 0).any(axis=0)]
    
    # Export
    df.round(0).astype(int).replace(to_replace=0, value='EPS').to_excel('WTRRRFLH_CLY.xlsx')

    return df



#---------------------------------- Run ---------------------------------- #

# Create dataset
def run():
    
    create_DE_VAR_T_CLASSIC_CLY()
    create_DE_CLASSIC_CLY_FACTOR()

    create_DH_VAR_T_CLY()
    create_DH_CLY_FACTOR()

    create_SOLE_VAR_T_CLY()
    create_SOLE_FLH_CLY()

    create_WND_VAR_T_CLY()
    create_WND_FLH_CLY()

    create_WTRRSVAR_S_CLY()
    create_WTRRSFLH_CLY()

    create_WTRRRVAR_T_CLY()
    create_WTRRRFLH_CLY()

    
