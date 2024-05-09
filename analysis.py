# analysis.py
# Author: Jake Daly
# Analysis of the iris data set

import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
 

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

# creating a histogram for each variable in the dataframe, https://www.geeksforgeeks.org/plotting-histogram-in-python-using-matplotlib/
for column in df: # loop through each column in the dataframe, https://www.geeksforgeeks.org/loop-or-iterate-over-all-or-certain-columns-of-a-dataframe-in-python-pandas/
    plt.figure(figsize=(8, 6)) # sets figure size 
    plt.hist(df[column], bins = 20, color = "skyblue", edgecolor='black') # plots histogram of current column
    plt.title(f'Histogram of {column}') # gives a title for each histogram with corresponding variable name
    plt.xlabel(f'{column} in centimetres') # variable name on x axis
    plt.ylabel('Frequency') # label y axis
    plt.savefig(f'{column}_histogram.png') # saves histogram with corresponding colum name to .png file, https://stackoverflow.com/questions/73912257/saving-histogram-as-jpg