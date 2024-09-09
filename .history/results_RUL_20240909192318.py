from FirstSteps import column_converter,feature_skew_check
from stall_margin_formulas import HPCStallMargin
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np


FD001_data_training, _ = column_converter('CMAPSSData/train_FD001.txt')

FD001_data_test, _ = column_converter('CMAPSSData/test_FD001.txt')

(print(FD001_data_training.shape()))
print(FD001_data_test.shape())



