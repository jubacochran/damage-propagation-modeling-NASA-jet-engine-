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
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
'''
# Load data
FD001_data = pd.read_csv('filtered_dataset.csv')
print(FD001_data)
print(FD001_data.info())

# Columns to exclude from plotting
exclude_columns = ['datetime','hours']
'''

# Load data
FD001_data = pd.read_csv('CMAPSSData/train_FD001.txt',delim_whitespace=True)
pd.set_option('display.max_rows', None)

#FD001_data.info()

# List of sensor names
sensor_names =['Fan inlet temperature ◦R', 'LPC outlet temperature ◦R', 
'HPC outlet temperature ◦R', 'LPT outlet temperature ◦R',
'Fan inlet Pressure psia', 'bypass-duct pressure psia',
'HPC outlet pressure psia', 'Physical fan speed rpm',
'Physical core speed rpm', 'Engine pressure ratioP50/P2',
'HPC outlet Static pressure psia', 'Ratio of fuel flow to Ps30 pps/psia',
'Corrected fan speed rpm', 'Corrected core speed rpm', 'Bypass Ratio ', 
'Burner fuel-air ratio', 'Bleed Enthalpy', 'Required fan speed', 
'Required fan conversion speed', 'High-pressure turbines Cool air flow', 
'Low-pressure turbines Cool air flow']

#Setting dict to store new names for colummns 
operational_dict = {
    '1':'Unit',
    '1.1':'Cycles',
    '-0.0007': 'Operational Setting 1',
    '-0.0004': 'Operational Setting 2',
    '100.0': 'Operational Setting 3',
}

#Creating a list of columns names
old_columns = list(FD001_data.columns)

#Removing column names found in operational_dict
operational_columns_to_rename = [col for col in old_columns if col not in operational_dict.keys()]

#Combining the two lists into a dict using tuple type casting
converged_names = dict(map(lambda i,j : (i,j) , operational_columns_to_rename,sensor_names))

# Combine both dictionaries (operational_dict and converged_names) ** is just short hand combining of the dictionary
all_rename_dict = {**operational_dict, **converged_names}

# Renaming columns in the dataframe
FD001_data.rename(columns=all_rename_dict, inplace=True)
print(FD001_data.info())









# Iterate over the columns and plot each one separately
for col in FD001_data.columns:
    if col in sensor_names:
        plt.figure(figsize=(10, 5))
        plt.plot(FD001_data[col].values)
        plt.title(f'Time Series Plot for {col}')
        plt.xlabel('Index')
        plt.ylabel(col)
        plt.grid(True)
        plt.show()

        # Seasonal decomposition
        decomposition = seasonal_decompose(FD001_data[col], model='additive', period=168)
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        plt.show()

        # Plot ACF and PACF for original series
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_acf(FD001_data[col].dropna(), lags=50, ax=plt.gca())
        plt.title(f'ACF for {col}')
        
        plt.subplot(122)
        plot_pacf(FD001_data[col].dropna(), lags=50, ax=plt.gca())
        plt.title(f'PACF for {col}')
        plt.tight_layout()
        plt.show()

        # Take the first difference of the series
        FD001_data[f'{col}_diff'] = FD001_data[col].diff()

        # Check if the differenced series is constant
        if FD001_data[f'{col}_diff'].dropna().nunique() <= 1:
            print(f"The differenced series for {col} is constant. Skipping ADF test.")
            continue

        # Perform the ADF test on the first differenced series
        result = adfuller(FD001_data[f'{col}_diff'].dropna())
        print(f'Results for {col}:')
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        for key, value in result[4].items():
            print(f'Critical Value ({key}): {value}')
        
        # If p-value is greater than 0.05, consider further differencing
        if result[1] > 0.05:
            # Take the second difference of the series
            FD001_data[f'{col}_diff2'] = FD001_data[f'{col}_diff'].diff()
            
            # Plot ACF and PACF for second differenced series
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plot_acf(FD001_data[f'{col}_diff2'].dropna(), lags=50, ax=plt.gca())
            plt.title(f'ACF for {col} (Second Difference)')
            
            plt.subplot(122)
            plot_pacf(FD001_data[f'{col}_diff2'].dropna(), lags=50, ax=plt.gca())
            plt.title(f'PACF for {col} (Second Difference)')
            plt.tight_layout()
            plt.show()
        else:
            # Plot ACF and PACF for first differenced series
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plot_acf(FD001_data[f'{col}_diff'].dropna(), lags=50, ax=plt.gca())
            plt.title(f'ACF for {col} (First Difference)')
            
            plt.subplot(122)
            plot_pacf(FD001_data[f'{col}_diff'].dropna(), lags=50, ax=plt.gca())
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

