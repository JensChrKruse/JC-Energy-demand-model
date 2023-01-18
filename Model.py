# -*- coding: utf-8 -*-

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os

# Custom libraries
import Setup as setup
import ReadData as read
import ExogInputs as exoginputs
import Statistics as stats
import SpatialAnalysis as spatial
import LeastSquaresModel as LSQ
import HeatDemand as HD
import Plotter as plotter

# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})

# Working directory
os.chdir(setup.datafolder)

# Model object
class M:
  def __init__(self, type, t_start, t_end, area, country):
    self.type = type
    self.t_start = t_start
    self.t_end = t_end
    self.area = area
    self.country = country
    self.geo_points = spatial.points_in_geo(area)
    self.holiday_list = exoginputs.create_holiday_list(country, int(t_start))
    self.n_clusters = spatial.n_clusters(area)

def run_model(region, country):
  ## ----------------------------- Initialize model ----------------------------- ##
  m_cluster = M(setup.model_type, setup.t_start_cluster, setup.t_end_cluster, region, country)

  # Cluster data points based on training set and update geo_points in cluster model
  stats.cluster(m_cluster, read.temperature(m_cluster), 'Temp_cluster')
  #stats.cluster(m_cluster, read.windspeed(m_cluster), 'WS_cluster')
  #irr_cluster_labels = stats.cluster(m_cluster, read.irradiance(m_cluster), 'Irr_cluster')

  # Instantiate models
  m_train = copy.deepcopy(m_cluster)
  m_train.t_start, m_train.t_end = setup.t_start_train, setup.t_end_train
  m_test = copy.deepcopy(m_cluster)
  m_test.t_start, m_test.t_end = setup.t_start_test, setup.t_end_test
  m_hindcast = copy.deepcopy(m_cluster)
  m_hindcast.t_start, m_hindcast.t_end = setup.t_start_hindcast, setup.t_end_hindcast

  # Create training and test sets
  df_train_features, df_train_labels, df_train, df_test_features, df_test_labels, df_test = stats.train_test_set(m_train, m_test)
  feature_list = list(df_train_features.columns)
  test_feature_list = list(df_test_features.columns)

  ## ----------------------------- Fit and simulate ----------------------------- ##

  # Fit model
  coef, resid_train = LSQ.fit(df_train, feature_list)

  # Simulate scenarios
  init_cond = int(df_train_labels.mean())
  df_sim_train = LSQ.simulate(m_train, df_train_features, coef, init_cond)
  df_sim_test = LSQ.simulate(m_test, df_test_features, coef, init_cond)
  df_sim_train_1step = LSQ.simulate_1step(df_train, coef)
  df_sim_test_1step = LSQ.simulate_1step(df_test, coef)
  df_sim_test_mc = LSQ.simulate_monte_carlo(m_test, df_test_features, coef, init_cond, 1000, resid_train)

  # Long term hindcasting
  df_hindcast_features = exoginputs.request_exog_inputs(m_hindcast)
  df_sim_hindcast = LSQ.simulate(m_hindcast, df_hindcast_features, coef, init_cond)

  plotter.hindcast(m_hindcast, df_sim_hindcast, df_hindcast_features)
  #plotter.validate_hindcast(m_hindcast, df_sim_hindcast, 1998, 1998)


  ## ----------------------------- Heat demand ----------------------------- ##
  # Simulate with constant temperature threshold
  df_sim_test_classic = LSQ.simulate_classic(m_test, df_test_features, feature_list, coef, init_cond)
  df_sim_hindcast_classic = LSQ.simulate_classic(m_hindcast, df_hindcast_features, feature_list, coef, init_cond)

  # Degree day calculation
  df_heat_demand, df_deg_days = HD.heat_demand(m_hindcast, df_hindcast_features)

  plotter.temp_threshold(m_test, df_sim_test, df_sim_test_classic)


  ## ----------------------------- Verify model ----------------------------- ##
  # Mean absolute deviation
  errors = abs((df_sim_test['Demand'] - df_test['Demand'])/df_test['Demand']*100)
  print('Mean Absolute Percentage Error:', round(np.mean(errors), 2), '%')
  pd.DataFrame({'MAPE': [round(np.mean(errors), 2)]}).to_csv('Model validation/' + m_train.area + '/' + m_train.type[0] + ' MAPE.csv')

  # Model coefficients
  df_coef = pd.DataFrame({'Coefficient': coef}, index=feature_list)
  df_coef.to_csv('Model validation/' + m_train.area + '/' + m_train.type[0] + ' Model coefficients.csv')

  # Residuals
  df_resid_train = pd.DataFrame({'Resid': resid_train}, index = df_train.index)
  df_resid_train.to_csv('Model validation/' + m_train.area + '/' + m_train.type[0] + ' Train residuals.csv')
  resid_test = df_test['Demand'] - df_sim_test_1step['Demand']
  plotter.resid(m_train, resid_train, resid_test.values)

  # Autocorrelation in residuals
  plotter.autocor(m_train, resid_train, 'Training residuals')

  # Clusters
  spatial.plot_clusters(m_cluster)

  # Scatterplots
  plotter.scatterplot(m_train, df_train_features, df_train_labels, df_sim_train_1step, df_test_features, df_test_labels, df_sim_test_1step, '1-step ahead simulation')
  plotter.scatterplot(m_train, df_train_features, df_train_labels, df_sim_train, df_test_features, df_test_labels, df_sim_test, 'Multi-step ahead simulation')


  # Time series
  plotter.MC_time_series(m_test, int(m_test.t_start), df_test, df_sim_test_mc)
  #plotter.time_series(m_test, int(m_test.t_start), df_test, df_sim_test)
  #plotter.time_series(m_train, int(m_train.t_start), df_train, df_sim_train)

  # Load duration curve
  #plotter.LDC(m_test, df_test, df_sim_test)

  ## ----------------------------- Export ----------------------------- ##
  # Export hindcast 
  df_sim_hindcast.to_csv(setup.outputdatafolder + '/' + m_train.area + ' Hindcast_all_demands.csv')
  df_sim_hindcast_classic.to_csv(setup.outputdatafolder + '/' + m_train.area + ' Hindcast_classic.csv')
  df_heat_demand.to_csv(setup.outputdatafolder + '/' + m_train.area + ' Hindcast_heat.csv')
  df_deg_days.to_csv(setup.outputdatafolder + '/' + m_train.area + ' deg_days.csv')


def main():
  # Select model geography
  geo = pd.read_excel('Model geography.xlsx', sheet_name='Demand_regions')
  countries = geo['Country code'].values
  regions = geo['ENTSOE region code'].values

  # Run model for each region
  for i in range(0, len(regions)):
    print('################### Running region ' + regions[i] + ' ###################')
    run_model(regions[i], countries[i])

main()
