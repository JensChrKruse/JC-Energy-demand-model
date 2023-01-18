# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from scipy.stats import norm
from scipy.stats import t
import datetime

# Custom libraries
import Setup as setup
import ExogInputs as exoginputs
import ReadData as read
import Statistics as stats


# Plot residual histogram and distributions
def resid(m, resid_train, resid_test):
    x = np.arange(np.percentile(resid_train, 2.5), np.percentile(resid_train, 97.5), 0.1)

    # Fit t-distribution
    t_params_train = stats.fit_t(resid_train)
    t_pdf_train = t.pdf(x, df=t_params_train[0], loc=t_params_train[1], scale=t_params_train[2])
    t_params_test = stats.fit_t(resid_test)
    t_pdf_test = t.pdf(x, df=t_params_test[0], loc=t_params_test[1], scale=t_params_test[2])

    # Plot residual histogram, normal distribution and t distribution
    fig, axs = plt.subplots(2, 1, figsize=(11,8), sharex=True, sharey=False)
    axs[0].hist(resid_train, range=[x[0], x[-1]], bins=50, density=True)
    axs[0].plot(x, norm.pdf(x, np.mean(resid_train), np.std(resid_train)))
    axs[0].plot(x, t_pdf_train)
    axs[1].hist(resid_test, range=[x[0], x[-1]], bins=50, density=True)
    axs[1].plot(x, norm.pdf(x, np.mean(resid_test), np.std(resid_test)))
    axs[1].plot(x, t_pdf_test)
    plt.grid(alpha=0.7)
    fig.suptitle('Residuals ' + m.area)
    fig.supxlabel('Residuals [MW]')

    axs[0].legend(['Normal fit', 't fit', 'Residuals'])
    name = 'Model validation/' + m.area + '/' + m.type[0] + ' Residuals ' + m.area + '.png'
    plt.savefig(name, dpi=300, transparent=False)
    plt.show()

def autocor(m, resid, title):
    #ACF
    plt.figure()
    pacf = plot_acf(resid, lags=50)
    plt.grid(alpha=0.7)
    plt.xlabel('Lag')
    filename = 'Model validation/' + m.area + '/' + m.type[0] + ' ' + m.area + ' ' + title + ' ACF.png'
    plt.savefig(filename, dpi=300, transparent=False)
    plt.show()

    # PACF
    plt.figure()
    pacf = plot_pacf(resid, lags=50)
    plt.grid(alpha=0.7)
    plt.xlabel('Lag')
    filename = 'Model validation/' + m.area + '/' + m.type[0] + ' ' + m.area + ' ' + title + ' PACF.png'
    plt.savefig(filename, dpi=300, transparent=False)
    plt.show()

def time_series(m, year, df_meas, df_sim):
    months = np.arange(1,13)
    month_name = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    for t in range(0, len(months)):
        df_meas_slice = df_meas.loc[(df_meas.index.month == months[t]) & (df_meas.index.year == year)]
        df_sim_slice = df_sim.loc[(df_sim.index.month == months[t]) & (df_sim.index.year == year)]
        plt.figure()
        plt.plot(df_meas_slice['Demand'])
        plt.plot(df_sim_slice['Demand'])
        plt.xlim([df_meas_slice.index[0], df_meas_slice.index[-1]])
        plt.ylim([min(df_meas['Demand']), max(df_meas['Demand'])])
        plt.legend(['Measured', 'Simulated'], loc='upper right')
        plt.grid(alpha=0.7)
        plt.ylabel('Power [MW]')
        plt.title(str(month_name[t]) + ' ' + m.area)
        filename = 'Model validation/' + m.area + '/' + m.type[0] + ' ' + m.area + ' ' + str(month_name[t]) + '.png'
        plt.savefig(filename, dpi=300, transparent=False)
        plt.show()

def MC_time_series(m, year, df_meas, df_sim):
    months = np.arange(1,13)
    month_name = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    for t in range(0, len(months)):
        df_sim_slice = df_sim.loc[(df_sim.index.month == months[t]) & (df_sim.index.year == year)]
        df_meas_slice = df_meas.loc[(df_meas.index.month == months[t]) & (df_meas.index.year == year)]
        plt.figure()
        plt.plot(df_sim_slice['Demand_mean'])
        plt.plot(df_meas_slice['Demand'])
        plt.fill_between(df_sim_slice.index, df_sim_slice['Demand_lb'], df_sim_slice['Demand_ub'], alpha=0.2)
        plt.xlim([df_meas_slice.index[0], df_meas_slice.index[-1]])
        plt.ylim([min(df_sim['Demand_lb']), max(df_sim['Demand_ub'])])
        plt.legend(['Simulated', 'Measured', '95% PI'], loc='upper right')
        plt.grid(alpha=0.7)
        plt.ylabel('Power [MW]')
        plt.title(str(month_name[t]) + ' ' + m.area)
        filename = 'Model validation/' + m.area + '/' + m.type[0] + ' ' + m.area + ' ' + str(month_name[t]) + '.png'
        plt.savefig(filename, dpi=300, transparent=False)
        plt.show()

# Scatterplots
def scatterplot(m, df_train_features, df_meas_train_demand, df_sim_train_demand, df_test_features, df_meas_test_demand, df_sim_test_demand, title):
    df_temp_train = stats.pop_weight_mean(m, df_train_features, 'Temp_cluster')
    df_temp_test = stats.pop_weight_mean(m, df_test_features, 'Temp_cluster')
    df_meas_train = df_temp_train.join(df_meas_train_demand).dropna()
    df_meas_test = df_temp_test.join(df_meas_test_demand).dropna()
    df_sim_train = df_temp_train.join(df_sim_train_demand).dropna()
    df_sim_test = df_temp_test.join(df_sim_test_demand).dropna()

    fig, axs = plt.subplots(2, 1, figsize=(11,8), sharex=True, sharey=False)
    axs[0].scatter(df_meas_train['Pop weight mean'], df_meas_train['Demand'], marker='o')
    axs[0].scatter(df_sim_train['Pop weight mean'], df_sim_train['Demand'], marker='*')
    axs[1].scatter(df_meas_test['Pop weight mean'], df_meas_test['Demand'], marker='o')
    axs[1].scatter(df_sim_test['Pop weight mean'], df_sim_test['Demand'], marker='*')
    fig.suptitle(title)
    fig.supxlabel('Temperature [°C]')
    fig.supylabel('Demand [MW]')
    plt.grid(alpha=0.7)
    axs[0].set_title('Training data')
    axs[1].set_title('Test data')
    axs[0].legend(['Measured', 'Simulated'], loc='upper right')
    name = 'Model validation/' + m.area + '/' + m.type[0] + ' Scatterplot ' + m.area + ' ' + title + '.png'
    plt.savefig(name, dpi=300, transparent=False)
    plt.show()

def hindcast(m, df_sim_hindcast, df_hindcast_features):
    # Resample
    df_sim_hindcast_resampled_max = df_sim_hindcast.resample('M').max()
    df_sim_hindcast_resampled_mean = df_sim_hindcast.resample('M').mean()
    df_sim_hindcast_resampled_min = df_sim_hindcast.resample('M').min()
    df_hindcast_temp_resampled_mean = stats.pop_weight_mean(m, df_hindcast_features, 'Temp_cluster').resample('M').mean()

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(11,8), sharex=True, sharey=False)
    axs[0].fill_between(df_sim_hindcast_resampled_min.index, df_sim_hindcast_resampled_min['Demand'], df_sim_hindcast_resampled_max['Demand'], alpha=0.3)
    axs[0].plot(df_sim_hindcast_resampled_mean)
    axs[1].plot(df_hindcast_temp_resampled_mean, color='crimson')
    plt.xlim([datetime.datetime(int(m.t_start), 1, 1), datetime.datetime(int(m.t_end), 12, 31)])
    plt.grid(alpha=0.7)
    axs[0].set_ylabel('Demand [MW]')
    axs[1].set_ylabel('Temperature [°C]')
    fig.suptitle(m.country + ' Hindcast')
    axs[0].legend(['Min-Max range', 'Monthly mean'])
    plt.savefig('Model validation/' + m.area + '/' + m.type[0] + ' Hindcast ' + m.area + '.png', dpi=300, transparent=False)
    plt.show()

def validate_hindcast(m, df_sim_hindcast, t_start, t_end):
    # Read ECEM for comparision
    df_ECEM = read.ECEM_demand(m)
    df_ECEM_resampled_mean = df_ECEM.resample('D').mean()

    # Resample
    df_sim_hindcast_resampled_mean = df_sim_hindcast.resample('D').mean()

    # Plot
    plt.figure()
    plt.plot(df_sim_hindcast_resampled_mean)
    plt.plot(df_ECEM_resampled_mean['Demand'])
    plt.xlim([datetime.datetime(t_start, 1, 1), datetime.datetime(t_end, 12, 31)])
    plt.grid(alpha=0.7)
    plt.ylabel('Demand [MW]')
    plt.title(m.country + ' Hindcast validation')
    plt.legend(['My model', 'Copernicus ECEM'])
    plt.savefig('Model validation/' + m.area + '/' + m.type[0] + ' Hindcast validation ' + m.area + '.png', dpi=300, transparent=False)
    plt.show()

# Load duration curve
def LDC(m, DF1, DF2):
    df1_sorted = DF1.loc[DF1.index.year == int(m.t_start)].sort_values(by=['Demand'], ascending=False)
    df2_sorted = DF2.loc[DF2.index.year == int(m.t_start)].sort_values(by=['Demand'], ascending=False)
    t = np.linspace(1,100,len(df1_sorted))
    plt.figure()
    plt.plot(t,df1_sorted['Demand'])
    plt.plot(t,df2_sorted['Demand'])
    plt.xlim([0,101])
    plt.xlabel('Duration [%]')
    plt.ylabel('Demand [MW]')
    plt.grid(alpha=0.7)
    plt.legend(['Measured', 'Simulated'], loc='upper right')
    name = 'Model validation/' + m.area + '/' + m.type[0] + ' LDC ' + m.area + '.png'
    plt.savefig(name, dpi=300, transparent=False)
    plt.show()

def temp_threshold(m, df_sim_test, df_sim_test_const_temp):
    # Plot simulation with actual and constant temperature profiles 
    plt.figure()
    plt.plot(df_sim_test['Demand'].resample('D').mean())
    plt.plot(df_sim_test_const_temp['Demand'].resample('D').mean())
    plt.legend(['Simulation: Actual temperature', 'Simulation: Constant ' + str(setup.T_set) + '°C threshold'], loc='upper center')
    plt.ylabel('Demand [MW]')
    plt.grid(alpha=0.7)
    plt.savefig('Model validation/' + m.area + '/' + 'Temp dependent demand.png', dpi=300, transparent=False)
    plt.show()

def heat_profile(m, df_sim_test_elecheat, heat_demand_hourly, df_hindcast_features, time):
    # Compare degree day and electrical heating profiles
    elec_heat = df_sim_test_elecheat/df_sim_test_elecheat.sum()
    fig, axs = plt.subplots(2, 1, figsize=(11,8), sharex=True, sharey=False)
    axs[0].plot(heat_demand_hourly.loc[time])
    axs[0].plot(elec_heat.loc[time])
    axs[1].plot(stats.pop_weight_mean(m, df_hindcast_features, 'Temp_cluster').loc[time], color='crimson')
    axs[0].set_title('Heat profiles')
    axs[1].set_title('Temperature')
    axs[1].set_ylabel('Temperature [°C]')
    axs[0].legend(['Degree day', 'Electric heat'])
    plt.savefig('Model validation/' + m.area + '/' + 'Heat_profiles_oct.png', dpi=300, transparent=False)
    plt.show()