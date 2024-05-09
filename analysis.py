# analysis.py
# Author: Jake Daly
# Analysis of the iris data set

import pandas as pd
import numpy as np
from tabulate import tabulate

# download iris data and read it into a dataframe
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# use df.describe() function to get summary statistics of the variables, https://discuss.datasciencedojo.com/t/how-to-get-summary-statistics-of-a-pandas-dataframe-in-python/1137/2
summary = df.describe()

# format the data into a simple table using the tabulate module, https://pypi.org/project/tabulate/
table = tabulate(summary, headers='keys', tablefmt='grid')

# save the table as a .txt file
with open('summary.txt', 'w') as f:
    f.write(table)