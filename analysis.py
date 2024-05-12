# analysis.py
# Author: Jake Daly
# Analysis of the iris data set

# modules imported for data visulaisation and analysis
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# modules used for machine learning
from sklearn.model_selection import train_test_split # splits data sets into random subsets of train and test
from sklearn.preprocessing import LabelEncoder # used to convert catagorical data to numerical data for machine learning purposes
from sklearn.model_selection import cross_val_score # used to evaluate the models performance using cross-validation

# machine learning models used 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
 
 # Iris data set
 # https://github.com/mwaskom/seaborn-data/blob/master/iris.csv

# download iris data and read it into a dataframe
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv' # found here https://github.com/mwaskom/seaborn-data/blob/master/iris.csv
df = pd.read_csv(url)

# use df.describe() function to get summary statistics of the variables, https://discuss.datasciencedojo.com/t/how-to-get-summary-statistics-of-a-pandas-dataframe-in-python/1137/2
summary = df.describe()
table = tabulate(summary, headers='keys', tablefmt='grid') # format the data into a simple table using the tabulate module, https://pypi.org/project/tabulate/

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
    if column != 'species':  # check if the column is not 'species'
        plt.xlabel(f'{column} in centimetres')
    else:
        plt.xlabel('Species')
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
plt.savefig('correlation_heatmap_of_each_pair_of_variables')
plt.show()

# box plot of each variable, https://seaborn.pydata.org/generated/seaborn.boxplot.html
fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # tutorials on subplots. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
sns.boxplot(data=df, x="species", y="sepal_length", ax=axes[0, 0]).set_ylabel("Sepal Length(cm)")
sns.boxplot(data=df, x="species", y="sepal_width", ax=axes[0, 1]).set_ylabel("Sepal Width(cm)")
sns.boxplot(data=df, x="species", y="petal_length", ax=axes[1, 0]).set_ylabel("Petal Length(cm)")
sns.boxplot(data=df, x="species", y="petal_width", ax=axes[1, 1]).set_ylabel("Petal Width(cm)")
plt.savefig('box_plot_distribution')
plt.show()

# violin plot of each variable, https://seaborn.pydata.org/generated/seaborn.violinplot.html
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.violinplot(data=df, x="species", y="sepal_length", fill=False, ax=axes[0, 0]).set_ylabel("Sepal Length(cm)")
sns.violinplot(data=df, x="species", y="sepal_width", fill=False, ax=axes[0, 1]).set_ylabel("Sepal Width(cm)")
sns.violinplot(data=df, x="species", y="petal_length", fill=False, ax=axes[1, 0]).set_ylabel("Petal Length(cm)")
sns.violinplot(data=df, x="species", y="petal_width", fill=False, ax=axes[1, 1]).set_ylabel("Petal Width(cm)")
plt.savefig('violin_plot_distribution')
plt.show()

# machine learning, https://www.hackersrealm.net/post/iris-dataset-analysis-using-python

# preparing the data set for machine learning
le = LabelEncoder()
df['species'] = le.fit_transform(df['species']) # converts the catagorical 'species' data to numerical values 
X = df.drop(columns=['species']) # input variables, all the data in the iris dataset without the species variable
Y = df['species'] # output variable or target variable, just the species column in the dataset

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30) # train_test_split function splits the data 70:30, 70% of the data to training set and 30% to the testing set

# logistic regression model
sk_model= LogisticRegression()
# model training
sk_model.fit(x_train, y_train) # trains the model using the training data (x_train and y_train)
# print metric to get performance
sk_model_score = cross_val_score(sk_model, X, Y, cv=3) #https://www.kaggle.com/code/amarsharma768/cross-val-score
sk_model_mean_score = sk_model_score.mean()
print('Average score of the Logistic Regression model:', sk_model_mean_score)


# knn - k-nearest neighbours model
knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
# print metric to get performance
knn_model_score = cross_val_score(knn_model, X, Y, cv=3) # evaluatets model performance using cross-validation, cv=3 means 3 fold cross validation i.e dataset is split into 3 different subsets that are used to train the model
knn_model_mean_score = knn_model_score.mean() # average of the cross_val_score
print('Average score of the Knneighbours model:', knn_model_mean_score) # how well the model is able to predict the Y variable based on the X variable, score out of 100

# decision tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)
# print metric to get performance
dt_model_score = cross_val_score(dt_model, X, Y, cv=3)
dt_model_mean_score = dt_model_score.mean()
print('Average score of the Decision Tree model:', dt_model_mean_score)

# saving scores as .txt file
with open("model_scores.txt", "w") as file:
    file.write(f"Logistic Regression model score: {sk_model_mean_score}\n")
    file.write(f"K Nearest Neighbors model score: {knn_model_mean_score}\n")
    file.write(f"Decision Tree model score: {dt_model_mean_score}\n")
