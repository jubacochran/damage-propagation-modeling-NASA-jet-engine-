# %%

import sys
import os

# Add the project root directory to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import *

from sklearn.cluster import k_means,DBSCAN

x = rul_FD001_training
print(rul_FD001_training.columns)

dbscan = DBSCAN(eps=3, min_samples=3).fit(x)
sns.scatterplot(x,dbscan.labels_)