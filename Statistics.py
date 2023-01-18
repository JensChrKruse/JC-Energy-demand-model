# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import t
# K means
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Custom libraries
import ExogInputs as exoginputs
import ReadData as read

# Calculates statistics for a dataframe of cunsumption
def corres_stats(DF):
    year_set = pd.unique(DF.index.year) # Array of unique years in the dataframe

    # Preallocate arrays for statistics
    mean = np.zeros(len(year_set))
    std = np.zeros(len(year_set))
    min = np.zeros(len(year_set))
    max = np.zeros(len(year_set))
    sum = np.zeros(len(year_set))
    count = np.zeros(len(year_set))

    # Gather statistics for all years
    for t in range(0,len(year_set)):
        df_select = DF.loc[str(year_set[t])]
        mean[t] = df_select.mean()
        std[t] = df_select.std()
        min[t] = df_select.min()
        max[t] = df_select.max()
        sum[t] = df_select.sum() / 10**6
        count[t] = df_select.count()

    yearly_stats = pd.DataFrame({'Year': year_set,
                                'Mean [MW]': mean,
                                'Min [MW]': min,
                                'Max [MW]': max,
                                'Std [MW]': std,
                                'Sum [TWh]': sum,
                                'Count [#]': count})
    yearly_stats = yearly_stats.set_index('Year') # Set time as index
    yearly_stats = yearly_stats.round(2) # Round to 0 decimals

    # Plot histogram.
    plt.figure(figsize =(10, 7))
    for t in year_set:
        plt.hist(DF.loc[str(t)], bins=50, histtype='step', density=True)
    plt.legend(year_set)
    plt.xlabel('Average power [MW]')
    plt.show()

    # Plot box plot
    yearly_series = []
    for t in range(0,len(year_set)):
        yearly_series.append(DF.loc[str(year_set[t])].values)
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.boxplot(yearly_series)
    ax.set_xticklabels(year_set)
    plt.ylabel('Power [MW]')
    plt.show()

    return yearly_stats #, df_diff

# Calculates statistics for a dataframe of cunsumption
def demand_stats(DF):
    year_set = pd.unique(DF.index.year) # Array of unique years in the dataframe

    # Preallocate arrays for statistics
    mean = np.zeros(len(year_set))
    std = np.zeros(len(year_set))
    min = np.zeros(len(year_set))
    max = np.zeros(len(year_set))
    sum = np.zeros(len(year_set))
    count = np.zeros(len(year_set))

    # Gather statistics for all years
    for t in range(0,len(year_set)):
        df_select = DF.loc[str(year_set[t])]

        mean[t] = df_select.mean()
        std[t] = df_select.std()
        min[t] = df_select.min()
        max[t] = df_select.max()
        sum[t] = df_select.sum() / 10**6
        count[t] = df_select.count()

    yearly_stats = pd.DataFrame({'Year': year_set,
                                'Mean [MW]': mean,
                                'Min [MW]': min,
                                'Max [MW]': max,
                                'Std [MW]': std,
                                'Sum [TWh]': sum,
                                'Count [#]': count})
    yearly_stats = yearly_stats.set_index('Year') # Set time as index
    yearly_stats = yearly_stats.round(0) # Round to 0 decimals
    
    # Plot histogram.
    plt.figure()
    for t in year_set:
        plt.hist(DF.loc[str(t)], bins=50, histtype='step', density=True)
    plt.legend(year_set)
    plt.xlabel('Average power [MW]')
    plt.xlim([1000,7000])
    plt.grid(alpha=0.7)
    plt.savefig('demand_hist.png', dpi=300, transparent=False)
    plt.show()
    
    # Plot box plot
    yearly_series = []
    for t in range(0,len(year_set)):
        yearly_series.append(DF.loc[str(year_set[t])].values.flatten())
    fig, ax = plt.subplots()
    ax.boxplot(yearly_series)
    ax.set_xticklabels(year_set)
    plt.ylabel('Power [MW]')
    plt.grid(alpha=0.7)
    plt.savefig('demand_boxplot.png', dpi=300, transparent=False)
    plt.show()

    
    # Determine 1st difference
    
    df_diff = DF.diff(periods=1)

    for t in year_set:
        plt.figure()
        plt.grid()
        plt.plot(df_diff.loc[str(t)])
        plt.xlabel('Time')
        plt.ylabel('Consumption ramp [MW/h]')
        plt.title(str(t))
        plt.grid(alpha=0.7)
        name = 'demand_diff_' + str(t) + '.png'
        plt.savefig(name, dpi=300, transparent=False)
        plt.show()

    return yearly_stats, df_diff

# Plots an average day for the time period provided in the dataframe
def avg_day(DF, title):
    DF_avg_day = DF.groupby(DF.index.hour).mean()
    DF_std_day = DF.groupby(DF.index.hour).std()
    plt.figure()
    #plt.fill_between(DF_avg_day.index, DF_avg_day-DF_std_day, DF_avg_day+DF_std_day, alpha=0.2)
    plt.plot(DF_avg_day)
    plt.xlabel('Hour of the day')
    plt.ylabel('Power [MW]')
    plt.title(title)
    plt.xlim([0,24])
    plt.ylim([1000,3500])
    name = 'demand_avg_day_' + title + '.png'
    #plt.savefig(name, dpi=300, transparent=False)
    plt.show()

# Plots an average week for the time period provided in the dataframe
def avg_week(DF, title):
    DF_avg_week = DF.groupby((DF.index.dayofweek) * 24 + (DF.index.hour)).mean()
    DF_std_week = DF.groupby((DF.index.dayofweek) * 24 + (DF.index.hour)).std()
    plt.figure()
    plt.fill_between(DF_avg_week.index, DF_avg_week['Demand']-DF_std_week['Demand'], DF_avg_week['Demand']+DF_std_week['Demand'], alpha=0.2)
    plt.plot(DF_avg_week['Demand'])
    plt.xlabel('Hour of the week')
    plt.ylabel('Power [MW]')
    plt.title(title)
    plt.xlim([0,168])
    #plt.ylim([2000,6000])
    plt.grid(alpha=0.7)
    name = 'demand_avg_week_' + title + '.png'
    plt.savefig(name, dpi=300, transparent=False)
    plt.show()

# Make test and training sets for model creation
def train_test_set(m_train, m_test):
    # Training set
    df_train_features = exoginputs.request_exog_inputs(m_train)
    df_train_labels = read.demand(m_train)
    df_train = df_train_features.join(df_train_labels)

    # Autoregressive terms in dynamic models
    if m_train.type[1] == 'Dynamic':
        df_train['Demand_Lag1'] = df_train['Demand'].shift(1)
        df_train['Demand_Lag2'] = df_train['Demand'].shift(2)

    df_train = df_train.dropna()

    # Test set
    df_test_features = exoginputs.request_exog_inputs(m_test)
    df_test_labels = read.demand(m_test)
    df_test = df_test_features.join(df_test_labels)


    # Autoregressive terms in dynamic models
    if m_test.type[1] == 'Dynamic':
        df_test['Demand_Lag1'] = df_test['Demand'].shift(1)
        df_test['Demand_Lag2'] = df_test['Demand'].shift(2)

    df_test = df_test.dropna()

    return df_train_features, df_train_labels, df_train, df_test_features, df_test_labels, df_test

# Fit data to t distribution and return parameters
def fit_t(x):
    t_params = t.fit(x)

    return t_params

# Sample random variables from a t distribution
def sample_t(t_params, s):
    r = t.rvs(df=t_params[0], loc=t_params[1], scale=t_params[2], size=s)

    return r

def limit_ramp(DF, thr):
    DF['Demand_diff'] = DF['Demand'].diff(periods=1)

    DF['Demand_diff'].plot()

    for t in range(1,len(DF)):
        if DF['Demand_diff'].iloc[t] >= thr:
            DF['Demand'].iloc[t] = DF['Demand'].iloc[t-1] + thr
        elif DF['Demand_diff'].iloc[t] < -thr:
            DF['Demand'].iloc[t] = DF['Demand'].iloc[t-1] - thr

    DF['Demand_diff_new'] = DF['Demand'].diff(periods=1)

    DF['Demand_diff_new'].plot()

    DF = DF.drop('Demand_diff', axis = 1)
    DF = DF.drop('Demand_diff_new', axis = 1)

    return DF

# Dummy table with interactions between dummy a and dummy b
def dummy_interaction(DF, dummy_a, dummy_b):
    a = DF.columns[DF.columns.str.startswith(dummy_a)]
    b = DF.columns[DF.columns.str.startswith(dummy_b)]

    names = [('Interact_' + col1 + '_' + col2) for col1 in a for col2 in b]
    interactions = [(DF[col1].mul(DF[col2]).values) for col1 in a for col2 in b]

    df = pd.DataFrame(np.transpose(interactions), columns=names, index=DF.index)

    return df

# Make piecewise function
def piecewise(DF, var, thr, BP):
    var_col = DF.columns[DF.columns.str.startswith(var)]

    names = [('PW_' + BP + '_' + i) for i in var_col]
    pw = [np.multiply((DF[i] - thr).values, ((DF[i] - thr) > 0).astype('int').values) for i in var_col]

    df = pd.DataFrame(np.transpose(pw), columns=names, index=DF.index)
    
    return df

def pop_weight_mean(m, DF, string):
    # Find columns with temperature clusters
    idx = DF.columns[DF.columns.str.startswith(string)]

    # Select temperature cluster columns
    df_temp = np.array(DF[idx])

    # Population weights for each cluster
    weights = m.geo_points.groupby('Temp_cluster').sum()['Pup_weight'].values

    # Population weighted mean across clusters
    mean = np.sum(df_temp * weights, axis=1)

    df = pd.DataFrame({'Pop weight mean': mean}, index=DF.index)

    return df

# Cluster columns in dataframe and return labels indicating which clusters each column belongs to
def cluster(m, DF, name):
    
    features = np.array(DF).T

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Initialize kmeans object
    kmeans = KMeans(
        init="random",
        n_clusters=m.n_clusters,
        n_init=10,
        max_iter=300,
        random_state=42
    )

    # Determine clusters
    kmeans.fit(scaled_features)

    # Labels indicating which data points belong together in clusters
    labels = kmeans.labels_[:]

    df = pd.DataFrame(labels, columns=[name], index=m.geo_points.index)

    m.geo_points = m.geo_points.join(df)

# Perform simple elementwise mean of columns given the cluster labels are given in model object
def mean_clustering(m, DF, name):

    # Data arrays
    features = np.array(DF)
    
    # Number of clusters
    n = m.n_clusters

    # Array of column indices
    col = np.arange(np.shape(features)[1])

    # Put labels and columns together
    a = np.vstack((m.geo_points[name].values, col)).T
    a = a[a[:, 0].argsort()]

    # Split into arrays with the columns that belong together
    cluster_idx = np.split(a[:,1], np.unique(a[:, 0], return_index=True)[1][1:])
    cluster_names = [name + '_' + str(x) for x in np.unique(a[:, 0])]

    # Calculate total population weight of each cluster 
    cluster_weights = m.geo_points.groupby(name).sum()['Pup_weight'].to_dict()

    # Calculate the weight of each point within each cluster
    weights = np.divide(m.geo_points['Pup_weight'].values, [cluster_weights[p] for p in m.geo_points[name]])

    # Define new array with mean values of columns that belong together in clusters
    mean_features = np.zeros((np.shape(features)[0], n))
    for i in range(0, n):
        mean_features[:,i] = np.sum(features[:, cluster_idx[i]] * weights[cluster_idx[i]], axis=1)

    df = pd.DataFrame(mean_features, columns=cluster_names, index=DF.index)

    return df

