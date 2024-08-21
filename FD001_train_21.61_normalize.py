# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer, PowerTransformer,MinMaxScaler
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler



rul_FD001_training = pd.read_csv('/Users/jubacochran/Downloads/AI_ML_UC_Berkeley/capstone/CMAPSSData/test_FD001.txt', delim_whitespace=True)

print(rul_FD001_training.info())

#Exploring Sensor measurement 5: [Column '21.61'] #10
#Baseline Skew is -5.51755
print(rul_FD001_training['21.61'].skew())
print(rul_FD001_training['21.61'].var())
print(rul_FD001_training['21.61'].mean())
print(rul_FD001_training['21.61'].std())
# Create a joint plot Y is not skewed
'''
Feature, '21.61' is skewed by -5.5175579165987925
'''
sns.jointplot(
    x='21.61', 
    y='1585.29', 
    data=rul_FD001_training, 
    kind='scatter',  
    palette='Spectral',
    marginal_kws=dict(bins=50, fill=True)
)

plt.show()

print(rul_FD001_training['21.61'].value_counts)
print(rul_FD001_training['21.61'].values)
print(rul_FD001_training['21.61'].describe())

#skewed_column = '21.61'

# Normalization
scaler = MinMaxScaler()
rul_FD001_training['normalized_column'] = scaler.fit_transform(rul_FD001_training[['21.61']])

# Log Transformation
rul_FD001_training['log_transformed_column'] = np.log1p(rul_FD001_training['21.61'] - 21.6)

# IQR Method for Outliers
Q1 = rul_FD001_training['21.61'].quantile(0.25)
Q3 = rul_FD001_training['21.61'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

rul_FD001_training_filtered = rul_FD001_training[(rul_FD001_training['21.61'] >= lower_bound) & (rul_FD001_training['21.61'] <= upper_bound)]

# Plotting the distributions
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.histplot(rul_FD001_training['21.61'], kde=True)
plt.title('Original')

plt.subplot(1, 3, 2)
sns.histplot(rul_FD001_training['normalized_column'], kde=True)
plt.title('Normalized')

plt.subplot(1, 3, 3)
sns.histplot(rul_FD001_training['log_transformed_column'], kde=True)
plt.title('Log Transformed')

plt.tight_layout()
plt.show()


