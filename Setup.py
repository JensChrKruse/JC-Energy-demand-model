# Model settings
model_type = ['LSQ', 'Dynamic']

## Variables
TDD = 0.75 # Fraction of heat demand dependent on ambient temperature
TID = 0.25 # Fraction of heat demand independent of ambient temperature
T_set = 15 # Temperature threshold below which heating is turned on
T_desired = 18 # Desired indoor temperature
T_breakpoint = 15 # Breakpoint for piece wise temperature function
t_offset = 24 # Number of extra hours to include before simulation period to stabilize autoregressive effects
clusters_per_degree_lat = 0.4 # Number of climate clusters per degree latitude


# Model year ranges
t_start_cluster = '2015'
t_end_cluster = '2019'

t_start_train = '2015'
t_end_train = '2018'

t_start_test = '2019'
t_end_test = '2019'

t_start_hindcast = '1982'
t_end_hindcast = '2019'

## Filepaths
datafolder = '/Users/JensChristian/Google Drev/Skole/DTU/Thesis/Modelling/Data' # Datafolder
outputdatafolder = 'Hindcast output'
corres_folder = 'Geodata/CorRES/'
custom_holidays = 'custom_holidays.xlsx'
temperature_file = 'temperature_100km.csv'
NUTS_file = 'Geodata/Boundaries/Eurostats_NUTS/NUTS_selected_regions_w_bidding_zones.shp'
grid_file = 'Geodata/Grid/100kmgrid.xlsx'
population_file = 'Geodata/Eurostat/JRC_GRID_2018/JRC_POPULATION_2018_25km.shp'