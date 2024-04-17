import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.dates as mdates

def plot_time_series(df, columns, title = 'Time Series Plot', acc=False, pacf=False):
    """
    Plots time series data, with options for decomposition and PACF plots, formatting the x-axis to show only years.

    Parameters:
    - df (pandas.DataFrame): The dataframe containing the time series data.
    - columns (list): List of column names in df to be plotted.
    - acc (bool, optional): If True, plots decomposition plots for each column. Defaults to False.
    - pacf (bool, optional): If True, plots PACF plot for each column. Defaults to 36 lags. Defaults to False.

    Returns:
    - Plots of time series data, decomposition, and/or PACF based on input flags, with x-axis showing only years.
    """
    sns.set_style("darkgrid")

    # plot each column on the same time plot
    plt.figure(figsize=(10, 6))
    for column in columns:
        plt.plot(df.index, df[column], label=column)
        
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # set major locator to every year.
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # set formatter to only display the year.
    plt.gcf().autofmt_xdate()  # auto-rotate lables
    
    plt.title(title)
    plt.legend()
    plt.show()

    # if acc is true, plot decomposition plots for each column
    if acc:
        for column in columns:
            result = seasonal_decompose(df[column], model='additive', period=1)
            result.plot()
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # only display year
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.gcf().autofmt_xdate()  # auto-rotate
            plt.suptitle(f'Decomposition Plot for {column}', y=1.05)
            plt.tight_layout()
            plt.show()

    # if pacf is true, plot pacf for each column
    if pacf:
        for column in columns:
            plt.figure(figsize=(10, 6))
            plot_pacf(df[column], lags=36)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.gcf().autofmt_xdate() 
            plt.title(f'PACF for {column}')
            plt.show()
