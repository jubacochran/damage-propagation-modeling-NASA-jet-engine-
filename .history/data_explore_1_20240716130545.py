# %%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


rul_FD001_training = pd.read_csv('CMAPSSData/test_FD001.txt', delim_whitespace=True)
rul_1 = pd.read_csv('CMAPSSData/RUL_FD001.txt')
print(rul_FD001_training.info())
rul_FD001_training.isnull().sum().plot.bar()
rul_FD001_training.isna().sum().plot.bar()
print('============')
print(rul_1.describe())
print(rul_1.head(3))
print(rul_1.info())

#Initialize list of skewed columns list
columns_to_normalize = []

# Loop through each column and create an individual plot
for column in rul_FD001_training.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=rul_FD001_training[column])
    plt.title(f'Distribution of {column}')
    plt.show()
    print(rul_FD001_training[column].skew())
    skewness = rul_FD001_training[column].skew()
    print(f'Skewness of {column}: {skewness}')
    
    # Check if the skewness is outside the range of -0.5 to 0.5
    if skewness < -0.5 or skewness > 0.5:
        columns_to_normalize.append(column)

# Print the columns that need normalization
print(f'Columns that need normalization: {columns_to_normalize}')

print(rul_FD001_training.head(8))

rul_1.plot.line()
print(rul_1.describe())
rul_FD001_training['1.1'].plot.line()


print(rul_FD001_training[1].unique)
    