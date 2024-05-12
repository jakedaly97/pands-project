# pands-project

This project is submitted for the Programming and Scripting module of the course Higher Diploma in Science in Computing in Data Analytics.

## The Iris Dataset

The Iris dataset was first made popular by British biologist and statistician Ronald Fischer. In 1939, he produced a paper called The use of multiple measurements in taxonomic problems, using the Iris dataset as an example of linear discriminant analysis. The dataset contains 150 samples of iris flowers, consisting of a total of 3 species: Setosa, Versicolor, and Virginica. Each sample includes 4 variables: sepal length, sepal width, petal length, and petal width, all of which are measured in centimetres.

* **Sepal Length:** The length of the flowers sepal(the outer part of the flower that protects the bud during development).
* **Sepal Width:** The Width of the flowers sepal
* **Petal Length:** The length of the flowers petal(the colourful leaves of a flower, surrounds the reproductive parts).
* **Petal Width:** The width of the flowers petal.

The Iris dataset is popular amongst beginners in the field of data analytics, statistics, and machine learning due to its simplicity and clear distinction between the feautures of each species of Iris flower in the dataset.



## Code Description

1. **Importing Libraries:** The code starts by importing necessary libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn.

2. **Loading Data:** The Iris dataset is loaded from a URL using pandas read_csv() function.

3. **Summary Statistics:** Summary statistics of the dataset are computed using describe() function and formatted into a table using the tabulate module. The table is then saved as a text file.

4. **Data Visualization:**

* Histograms are created for each variable in the dataset and saved as individual PNG files.
* Scatter plots are generated for each variable pair and displayed on a single pair plot, coloured by species and saved as a PNG file
* Two specific scatter plots (Sepal length vs Sepal width and Petal length vs Petal width) are generated with added regrression line to show correlation. Saved as seperate PNG files.
* A scatterplot of Sepal length vs Sepal width is generated using only Setosa species data with added regression line, saved as PNG file.
* Box plots and violin plots are created to visualize the distribution of each variable across different species, both saved as PNG files.

5. **Correlation Analysis:** A heatmap is generated to visualize the correlation between each pair of numeric variables.

6. **Machine Learning:**

* Data preprocessing: Label encoding is applied to convert categorical species labels into numerical values.
* The dataset is split into training and testing sets.
* Three machine learning models (Logistic Regression, K Nearest Neighbors, Decision Tree) are trained using the training data.
* Model performance is evaluated using cross-validation with 3 folds.
* The average scores of each model are calculated and saved into a .txt file.



## Use of This Project



## Resources



## Get Started

To get started with this project, users need an up-to-date version of Python installed on their systems. They can then download or clone the "pands-project" repository directly from GitHub.


## Get Help

For assistance, users can refer to the comments within the project, providing insights into the written code. Additional clarification can be found in the Python documentation available at https://docs.python.org/3/, or through numerous tutorials on Python, accessible on the W3Schools website at https://www.w3schools.com/python/default.asp. For further assistance, users can contact fellow project contributors. Additionally, the Palmer penguins dataset is widely used by data analysts to learn the fundamental tools of data analysis, other projects using the dataset can be found online.

## Contributions

Currently, Jake Daly is the sole contributor to this project.


## Author

Jake Daly is a part-time student enrolled in the Higher Diploma in Science in Computing in Data Analytics course at Atlantic Technological University. For inquiries or further information, Jake Daly can be contacted via his student email at g00439324@atu.ie or through his personal email at jakedaly1997@hotmail.com.
