# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer, PowerTransformer,MinMaxScaler
import scipy.stats as stats



rul_FD001_training = pd.read_csv('CMAPSSData/test_FD001.txt', delim_whitespace=True)

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

def quantileTransform(column):
    x = column.values.reshape(-1,1)

    #Initialize QuantileTransformer
    qt = QuantileTransformer(random_state=0)

    #Fit and transform data
    x_transformed = qt.fit_transform(x)

    column = x_transformed
    #Plot the distribtion of the transformed data
    sns.histplot(column, bins=50, kde=True)
    plt.title('Distribution of Transformed Data')
    plt.show()

    #Double check skewness
    data_skew = stats.skew(column)
    print(f"Skewness: {data_skew}")

#Trying quantile transform on 21.61 Operational sensor
quantileTransform(rul_FD001_training['21.61'])
#Maybe this feature is operating normally with these values. They only exist at 0 or 1


#Exploring Sensor measurement 12: [Column '9050.17'] #13
#Baseline Skew is 1.6546125767282374
print(rul_FD001_training['9050.17'].skew())
print(rul_FD001_training['9050.17'].var())
print(rul_FD001_training['9050.17'].mean())
print(rul_FD001_training['9050.17'].std())
# Create a joint plot Y is not skewed

'''
Feature, '9050.17' is skewed by 
'''
sns.jointplot(
    x='9050.17', 
    y='1585.29', 
    data=rul_FD001_training, 
    kind='scatter',  
    palette='Spectral',
    marginal_kws=dict(bins=50, fill=True)
)

#quantile is not a good transform because it destroys data.

#Using IRQ values to help with skew and to normalize
Q1 = rul_FD001_training['9050.17'].quantile(0.25)
Q3 = rul_FD001_training['9050.17'].quantile(0.75)
IQR = Q3 - Q1
df_filtered = rul_FD001_training[(rul_FD001_training['9050.17'] >= (Q1 - 1.5 * IQR)) & (rul_FD001_training['9050.17'] <= (Q3 + 1.5 * IQR))]
print('==================')
print(df_filtered['9050.17'].skew())

sns.jointplot(
    x='9050.17', 
    y='1585.29', 
    data=df_filtered, 
    kind='scatter',  
    palette='Spectral',
    marginal_kws=dict(bins=50, fill=True)
)
df_filtered.info()
print(df_filtered['9050.17'].var())
print(df_filtered['9050.17'].mean())
print(df_filtered['9050.17'].std())

'''
Up to this point 9057 is a normal distribution with data symmetry. 
'''

#Looking at 8125.55 to normalize
sns.jointplot(
    x='8125.55', 
    y='1585.29', 
    data=rul_FD001_training, 
    kind='scatter',  
    palette='Spectral',
    marginal_kws=dict(bins=50, fill=True)
)

#Using IRQ values to help with skew and to normalize
Q1 = rul_FD001_training['8125.55'].quantile(0.25)
Q3 = rul_FD001_training['8125.55'].quantile(0.75)
IQR = Q3 - Q1
df_filtered = rul_FD001_training[(rul_FD001_training['8125.55'] >= (Q1 - 1.5 * IQR)) & (rul_FD001_training['8125.55'] <= (Q3 + 1.5 * IQR))]
print('==================')
print(df_filtered['8125.55'].skew())

sns.jointplot(
    x='8125.55', 
    y='1585.29', 
    data=df_filtered, 
    kind='scatter',  
    palette='Spectral',
    marginal_kws=dict(bins=50, fill=True)
)
print(df_filtered['8125.55'].var())
print(df_filtered['8125.55'].mean())
print(df_filtered['8125.55'].std())

print(df_filtered.info())

sns.pairplot(df_filtered)
plt.title('')

