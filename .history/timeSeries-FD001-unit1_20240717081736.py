# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, MinMaxScaler
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Load data
normalized_data = pd.read_csv('filtered_dataset.csv')
print(normalized_data.info())
print(normalized_data.columns)
normalized_data.rename(columns={'1':'Engine No.','1.1':'Operation Setting 1','0.0023':'Operation Setting 2',
                                '0.0003':'Operation Setting 3'}, inplace=True)

# List of sensor names
sensor_names = [
    "Fan Inlet Temperature", "LPC Outlet Temperature", "HPC Outlet Temperature", "LPT Outlet Temperature",
    "Fan Inlet Pressure", "Bypass-Duct Pressure", "HPC Outlet Pressure", "Physical Fan Speed",
    "Physical Core Speed", "Engine Pressure Ratio", "HPC Outlet Static Pressure",
    "Ratio of Fuel Flow to Ps30", "Corrected Fan Speed", "Corrected Core Speed", "Bypass Ratio",
    "Burner Fuel-Air Ratio", "Bleed Enthalpy", "Required Fan Speed", "Required Fan Conversion Speed",
    "High-Pressure Turbines Cool Air Flow", "Low-Pressure Turbines Cool Air Flow"
]

# Column indices provided in the question
column_indices = list(range(5, 27))  # Since the given columns start from index 5 to 26

# Create a dictionary to map column indices to sensor names
column_mapping = dict(zip(column_indices, sensor_names))

# Print the mapping
print(column_mapping)

normalized_data.rename(columns=column_mapping, inplace=True)

# Print DataFrame info to verify the column names
print(normalized_data.info())


normalized_data['datetime'] = pd.to_datetime(normalized_data['datetime'])
print(normalized_data.info())
#print(normalized_data['1'].value_counts())
#normalized_data = normalized_data[normalized_data['Engine No.'] == 1]
print(normalized_data)


# Columns to exclude from plotting
#exclude_columns = ['hours']

decomposition = seasonal_decompose(normalized_data['643.02'], model='additive', period=10)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()

# Plot ACF and PACF in separate subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# ACF plot
plot_acf(normalized_data['643.02'].dropna(), lags=10, ax=axes[0])
axes[0].set_title('ACF FD001 unit 1-643.02')

# PACF plot
plot_pacf(normalized_data['643.02'].dropna(), lags=10, ax=axes[1])
axes[1].set_title('PACF FD001 unit 1-643.02')

plt.tight_layout()
plt.show()

arma = ARIMA(decomposition.resid, order=(0,0,1)).fit()
hist = arma.predict()
print(hist.tail())
plt.plot(hist,label='model')
plt.legend()
plt.grid()



'''
# Iterate over the columns and plot each one separately
for col in normalized_data.columns:
    if col not in exclude_columns:
        plt.figure(figsize=(10, 5))
        plt.plot(normalized_data[col].values)
        plt.title(f'Time Series Plot for {col}')
        plt.xlabel('Index')
        plt.ylabel(col)
        plt.grid(True)
        plt.show()

        # Seasonal decomposition
        decomposition = seasonal_decompose(normalized_data[col], model='additive', period=10)
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        plt.show()

        # Plot ACF and PACF for original series
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_acf(normalized_data[col].dropna(), lags=5, ax=plt.gca())
        plt.title(f'ACF for {col}')
        
        plt.subplot(122)
        plot_pacf(normalized_data[col].dropna(), lags=5, ax=plt.gca())
        plt.title(f'PACF for {col}')
        plt.tight_layout()
        plt.show()

        # Take the first difference of the series
        normalized_data[f'{col}_diff'] = normalized_data[col].diff()

        # Check if the differenced series is constant
        if normalized_data[f'{col}_diff'].dropna().nunique() <= 1:
            print(f"The differenced series for {col} is constant. Skipping ADF test.")
            continue

        # Perform the ADF test on the first differenced series
        result = adfuller(normalized_data[f'{col}_diff'].dropna())
        print(f'Results for {col}:')
        print('ADF Statistic:', result[0]) 
        print('p-value:', result[1])
        for key, value in result[4].items():
            print(f'Critical Value ({key}): {value}')
        
        # If p-value is greater than 0.05, consider further differencing
        if result[1] > 0.05:
            # Take the second difference of the series
            normalized_data[f'{col}_diff2'] = normalized_data[f'{col}_diff'].diff()
            
            # Plot ACF and PACF for second differenced series
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plot_acf(normalized_data[f'{col}_diff2'].dropna(), lags=5, ax=plt.gca())
            plt.title(f'ACF for {col} (Second Difference)')
            
            plt.subplot(122)
            plot_pacf(normalized_data[f'{col}_diff2'].dropna(), lags=5, ax=plt.gca())
            plt.title(f'PACF for {col} (Second Difference)')
            plt.tight_layout()
            plt.show()
        else:
            # Plot ACF and PACF for first differenced series
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plot_acf(normalized_data[f'{col}_diff'].dropna(), lags=5, ax=plt.gca())
            plt.title(f'ACF for {col} (First Difference)')
            
            plt.subplot(122)
            plot_pacf(normalized_data[f'{col}_diff'].dropna(), lags=5, ax=plt.gca())
            plt.title(f'PACF for {col} (First Difference)')
            plt.tight_layout()
            plt.show()

        # Calculate error metrics for residuals
        residuals = decomposition.resid.dropna()
        maer = np.mean(np.abs(residuals))
        rmser = np.sqrt(np.mean(residuals**2))
        print(f"Mean Absolute Error of Residuals (MAER) for {col}: {maer}")
        print(f"Root Mean Square Error of Residuals (RMSER) for {col}: {rmser}")
        
        # Residual diagnostics
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        sns.histplot(residuals, kde=True)
        plt.title(f'Residuals Histogram for {col}')
        
        plt.subplot(122)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'QQ Plot for {col}')
        plt.tight_layout()
        plt.show()

        # Checking mean of residuals
        mean_residuals = np.mean(residuals)
        print(f"Mean of Residuals for {col}: {mean_residuals}")

        # Shapiro-Wilk test for normality
        shapiro_test = stats.shapiro(residuals)
        print(f"Shapiro-Wilk Test for {col}:")
        print(f"Statistic: {shapiro_test.statistic}, p-value: {shapiro_test.pvalue}")


'''