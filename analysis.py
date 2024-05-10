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

# creating a scatter plot sepal length vs sepal width with rgression line, https://seaborn.pydata.org/generated/seaborn.regplot.html
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species")
sns.regplot(data=df, x="sepal_length", y="sepal_width", scatter=False)
plt.title('Sepal length vs Sepal width')
plt.xlabel('Sepal length(cm)')
plt.ylabel('Sepal width(cm)') 
plt.savefig('scatter_plot_sepal_length_vs_sepal_width')
plt.show()

# creating scatter plot petal length vs petal width with regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species")
sns.regplot(data=df, x="petal_length", y="petal_width", scatter=False)
plt.title('Petal length vs Petal width')
plt.xlabel('Petal length(cm)')
plt.ylabel('Petal width(cm)')
plt.savefig('scatter_plot_petal_length_vs_petal_width') 
plt.show()

# filtering data so we only use the data from the setosa species, https://medium.com/@AIWatson/how-to-filter-data-in-python-pandas-dataframes-using-conditional-statements-69c4bb842de8
setosa_filter = df[df['species'] == 'setosa'] # creating a new dataframe that only selects the data containg 'setosa' species

# scatter plot sepal length vs sepal width but only using the setosa data
plt.figure(figsize=(8, 6))
sns.scatterplot(data=setosa_filter, x="sepal_length", y="sepal_width", hue="species")
sns.regplot(data=setosa_filter, x="sepal_length", y="sepal_width", scatter=False)
plt.title('Sepal length vs Sepal width (Setosa)')
plt.xlabel('Sepal length(cm)')
plt.ylabel('Sepal width(cm)')
plt.savefig('setosa_only_scatter_plot') 
plt.show()

# filtering dataframe so it just contains the numeric data, https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html
numeric_df = df.select_dtypes(include='number')

# heatmap to show the correlation between each pair of variables, https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e
# NOTE, when this code was originally ran, an issue was encountered where the correlation coefficents were only displaying on the first row, numerous attempts were made to correct the code but nothing worked
# after checking forums online it was found that there is an actual bug with MATPLOTLIB version 3.8.0 (the one that was installed) that causes the heatmap to generate incorrectly, https://github.com/microsoft/vscode-jupyter/issues/14363
# with this new information, MATPLOTLIB was uninstalled and a previous version (MATPLOTLIB version 3.7.3) was installed in its place, this fixed the issue
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True)
plt.savefig('Correlation heatmap of each pair of variables')
plt.show()

