# analysis.py
# Author: Jake Daly
# Analysis of the iris data set

import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
 
 # Iris data set
 # https://archive.ics.uci.edu/dataset/53/iris, linked by Andrew Beatty, was using this version of the data set for the first while but encounterd errors when i was trying to plot the scatter graphs.
 # https://github.com/mwaskom/seaborn-data/blob/master/iris.csv, one im using, raw data here was used by Ian Mcloughlinn in his lectures

# download iris data and read it into a dataframe
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv' # found here https://github.com/mwaskom/seaborn-data/blob/master/iris.csv
df = pd.read_csv(url)

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

# creating a scatter plot for each pair of variables coloured by species, https://www.geeksforgeeks.org/python-seaborn-pairplot-method/
sns.pairplot(df, hue='species')
plt.savefig('scatter_plots_species.png')
plt.show()