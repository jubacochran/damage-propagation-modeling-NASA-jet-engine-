# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer, PowerTransformer,MinMaxScaler
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose



normalized_data = pd.read_csv('filtered_dataset.csv')
print(normalized_data)

print(normalized_data.info())

columns = normalized_data.columns

# List of columns to exclude from plotting
exclude_columns = ['datetime', '1.1', '0.0023', '0.0003', '100.0', '1','hours']

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
        decomposition = seasonal_decompose(normalized_data[col], model='additive', period=168)
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        plt.show()

        # Plot ACF and PACF for original series
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_acf(normalized_data[col].dropna(), lags=50, ax=plt.gca())
        plt.title(f'ACF for {col}')
        
        plt.subplot(122)
        plot_pacf(normalized_data[col].dropna(), lags=50, ax=plt.gca())
        plt.title(f'PACF for {col}')
        plt.tight_layout()
        plt.show()

        # Take the first difference of the series
        normalized_data[f'{col}_diff'] = normalized_data[col].diff()

        # Plot ACF and PACF for differenced series
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_acf(normalized_data[f'{col}_diff'].dropna(), lags=50, ax=plt.gca())
        plt.title(f'ACF for {col} (First Difference)')
        
        plt.subplot(122)
        plot_pacf(normalized_data[f'{col}_diff'].dropna(), lags=50, ax=plt.gca())
        plt.title(f'PACF for {col} (First Difference)')
        plt.tight_layout()
        plt.show()

        # Calculate error metrics for residuals
        residuals = decomposition.resid.dropna()
        maer = np.mean(np.abs(residuals))
        rmser = np.sqrt(np.mean(residuals**2))
        print(f"Mean Absolute Error of Residuals (MAER) for {col}: {maer}")
        print(f"Root Mean Square Error of Residuals (RMSER) for {col}: {rmser}")