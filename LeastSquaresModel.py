# Libraries
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
import copy

# Custom libraries
import Setup as setup
import Statistics as stats

def fit(df, feature_list):

    A = np.array(df.drop('Demand', axis=1))
    b = np.array(df['Demand'])

    temp_idx = np.where(np.char.startswith(feature_list, 'Temp') == True)[0]
    daylength_idx = np.where(np.char.startswith(feature_list, 'Daylength') == True)[0]

    # Constrain temperatures and daylength
    lb = np.full(len(feature_list), -np.inf)
    ub = np.full(len(feature_list), np.inf)
    ub[temp_idx] = 0
    ub[daylength_idx] = 0

    # Fit model to data
    model_fit = lsq_linear(A, b, bounds=(lb, ub), max_iter=10000, tol=1e-10, lsq_solver='lsmr', lsmr_tol='auto', verbose=1)

    # Model results
    coef = model_fit.x
    resid = A.dot(coef) - b

    return coef, resid

# Simulate a year using the Dynamic Regression model
def simulate(m, df_features, coef, init_cond):
  # Simulate year
  scenario = np.zeros(len(df_features))

  exog_features = np.array(df_features)

  # Initial time step
  exog_features[0,-1] = init_cond
  exog_features[0,-2] = init_cond
  scenario[0] = np.dot(coef, exog_features[0,:])
  exog_features[1,-1] = init_cond
  exog_features[1,-2] = scenario[0]

  # Intermediate time steps
  for t in range(1,len(scenario)-1):
      scenario[t] = np.dot(coef, exog_features[t,:])
      exog_features[t+1,-1] = scenario[t-1]
      exog_features[t+1,-2] = scenario[t]

  # Final time step
  scenario[-1] = np.dot(coef, exog_features[-1,:])

  df = pd.DataFrame({'Demand': scenario}, index=df_features.index)

  # Cut away run-in period
  df_slice = df.loc[m.t_start:m.t_end]

  return df_slice

# Simulate a year using the Static Regression model
def simulate_1step(df, coef):
    exog_features = np.array(df.drop('Demand', axis=1))

    # Simulate year
    scenario = np.zeros(len(df))

    # Intermediate time steps
    for t in range(0,len(scenario)):
        scenario[t] = np.dot(coef, exog_features[t,:])

    df = pd.DataFrame({'Demand': scenario}, index=df.index)

    return df

# Simulate the classic demand given a lower temperature threshold (setup.T_set)
def simulate_classic(m, df_features, feature_list, coef, init_cond):

  # Simulate with actual temperature profile
  df_sim = simulate(m, df_features, coef, init_cond)
  
  # Change temperatures to constant lower value
  df_exog_features = copy.deepcopy(df_features)
  temp_idx = np.where(np.char.startswith(feature_list, 'Temp') == True)[0]

  # Adjust all temperature columns to never go lower than the temperature threshold
  for i in temp_idx:
    df_exog_features.iloc[:,i] = np.maximum(setup.T_set, df_exog_features.iloc[:,i])

  # Simulate with constant temperature
  df_sim_const_temp = simulate(m, df_exog_features, coef, init_cond)

  # Long term demands (total and temperature independent)
  demand_tot = df_sim['Demand'].sum()
  demand_const_temp = df_sim_const_temp['Demand'].sum()
  
  # Desired splitting factors
  splitting_factors = pd.read_excel('DE_level_comparison.xlsx', sheet_name='Desired splitting factors', index_col='ENTSOE_region')
  x_split_desired = splitting_factors.loc[m.area, 'Classic']

  # Determine splitting factor to achieve the desired split between classic and heat related electricity
  x_split = (x_split_desired * demand_tot - demand_tot) / (demand_const_temp - demand_tot)
  x_split = min(x_split, 1) # Ensure that classic electricity cannot make up less than the temperature independent share

  # Define the classic electricity demand as the split between temperature independent demand and total demand
  df_sim_classic = x_split * df_sim_const_temp + (1-x_split) * df_sim

  return df_sim_classic


def simulate_monte_carlo(m, df_features, coef, init_cond, N, training_resid):
  # Simulate year
  scenario = np.zeros((len(df_features), N))

  exog_features = np.array(df_features)

  # Fit t-distribution
  t_params = stats.fit_t(training_resid)

  for n in range(0,N):
    # Sample error terms
    e = stats.sample_t(t_params, len(df_features))

    # Initial time step
    exog_features[0,-1] = init_cond
    exog_features[0,-2] = init_cond
    scenario[0,n] = np.dot(coef, exog_features[0,:]) + e[0]
    exog_features[1,-1] = init_cond
    exog_features[1,-2] = scenario[0,n]

    # Intermediate time steps
    for t in range(1,len(scenario)-1):
        scenario[t,n] = np.dot(coef, exog_features[t,:]) + e[t]
        exog_features[t+1,-1] = scenario[t-1,n]
        exog_features[t+1,-2] = scenario[t,n]

    # Final time step
    scenario[-1,n] = np.dot(coef, exog_features[-1,:]) + e[-1]

  scenario_mean = np.mean(scenario, axis=1)
  scenario_lb = np.percentile(scenario, 2.5, axis=1)
  scenario_ub = np.percentile(scenario, 97.5, axis=1)

  # Create dataframe
  df = pd.DataFrame({'Demand_mean': scenario_mean, 'Demand_lb': scenario_lb, 'Demand_ub': scenario_ub}, index=df_features.index)

  # Cut away run-in period
  df_slice = df.loc[m.t_start:m.t_end]

  return df_slice