# Table of contents

#### 1. Introduction

#### 2. Description of the dataset

     2.1 Head of our Dataset
    
     2.2 Tail of our Dataset

#### 3. Preprocessing and Data Cleaning for EDA

     3.1 Detecting Null values in our dataset

     3.2 Evaluating the best technique for handling null values

     3.3 After Treating Null values

     3.4 Assessing Dataset Balance

     3.5 Label Encoder

  #### 4. Exploratory Data Analysis

     4.1 Overview of dataset using statistics

     4.2 Barplot to know distribution of age column

     4.3 Barplot to know Number of Cigarettes smoked by different age groups

     4.4 Pie chart to illustrate the distribution of daily drink consumption

     4.5 Barplot to know smoking and drinking habits

     4.6 Histogram to explore the age distribution of individuals affected by lung cancer

     4.7 Scatter plot to understand relationship between age and smoking habits for the individuals in the dataset

     4.8 Boxplot to visually analyze the distribution

     4.9 Heatmap

  #### 5. Polishing our dataset for model training

     5.1 Outlier detection and treating using Z-score

     5.2 One-Hot Encoding 

     5.3 Standarization

  #### 6. Model Building and Model Evaluation
     
     6.1 Calculating accuracy of our models

     6.2 Calculating Precision, R2 score and F1 score

     6.3 Calculating confusion Matrix

  #### 7. Actual vs. Predicted Values

  #### 8. Conclusion 

  #### 9. References 




# 1. Introduction:
This README describes work done on the "Employee Compensation and Satisfaction Insights" Dataset. Resources used include Python and associated packages Google Colab, matplotlib, Seaborn, scikit-learn, statsmodels, and SciPy. We have addressed the regression challenge to uncover the factors impacting salaries, employing Support Vector Machine, Random Forest, and Linear Regression.

# 2. Description of the dataset:

#### 2.1 Head of our Dataset:
The very first step is always to check if the data needs cleaning by looking for duplicate rows, zero values or NaNs where they shouldn't be etc. The head of our dataset looks like:

![Screenshot 2024-05-22 131152](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/f2726f91-9bda-4453-a483-3ac0367b11bb)

#### 2.2 Tail of our Dataset:
The tail of our dataset looks like : 

![Screenshot 2024-05-22 132115](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/0270b766-bcb5-4109-9e14-48c67f9b4811)

Visually, we found out that there are few NaN values in our dataset

# 3. Preprocessing and Data Cleaning for EDA

#### 3.1 Detecting Null values in our dataset:

![Screenshot 2024-05-22 132813](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/667e87e4-f5cc-4458-b100-74d6e5615383)

There are 2 each null values in Smokes and Alkhol columns. There is one null value in Age column. 

#### 3.2 Evaluating the best technique for handling null values:

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/50a20522-fae1-4db2-9f70-4ff7ef46dd1d)

Plot is left skewed, hence we will use median technique for age column.

![test2](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/83fe3959-52b8-4dad-a7aa-4b0f68a4cda5)

 Plot is right skewed, hence we will use median technique for Smokes column.

![test3](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/e01501b0-2c3f-41b8-ad67-2b96ce8512ae)

Plot is slightly left skewed, hence we will use median technique for Alkhol column.

#### 3.3 After Treating Null values:

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/21b08ffb-e135-4110-8327-5b1fcae24ba5)

We have replaced our null values with median values in three columns : Age, Smokes and Alkhol

#### 3.4 Assessing Dataset Balance: 

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/7fdb2e12-a8d0-4743-b83f-bc26df76f5dd)

We have ratio of 32-29, so its fairly balanced dataset, so no further action is needed to balance the dataset

#### 3.5 Label Encoder:

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/02431943-2216-41dd-8d5c-fecb38b6e757)

We applied Label Encoder as we are solving classification problem and it helps for our model for better prediction. Label Encoder to transform categorical data into numerical form.

# 4. Exploratory Data Analysis

#### 4.1 Overview of dataset using statistics:

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/d5b38c22-cae2-4f8d-bc3c-689f69cd1e0e)

Approximately 50% people who smoke are below age 40. This group smokes approximately 15 cigarettes and consumes 3 drinks a day.

#### 4.2 Barplot to know distribution of age column

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/5be14dbc-dd4b-417b-b8ff-dcdf5bf1d41f)

Majority of the people in our dataset are from age groups 20 to 39.

#### 4.3 Barplot to know Number of Cigarettes smoked by different age groups

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/f800bbae-ec5b-42fb-9045-b0ef96718cae)

Approximately 19 cigarettes are smoked by people of age group 50 to 59

#### 4.4 Pie chart to illustrate the distribution of daily drink consumption

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/26cc1388-f290-4be4-bce4-04407baf40f5)

Approximately 33% people take 2 to 3 drinks per day

#### 4.5 Barplot to know smoking and drinking habits

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/0062e1db-4d16-46cf-958f-2ea3fef6cab5)

From our barplot, we come to know that majority of people do both smoking and drinking.

#### 4.6 Histogram to explore the age distribution of individuals affected by lung cancer

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/2bc678db-8207-4ee4-aac0-ef8afa29d600)

Majority of people affected by Lung cancer are approximately above age 47.

#### 4.7 Scatter plot to understand relationship between age and smoking habits for the individuals in the dataset

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/b637479c-0ede-4527-98ce-6a261c6aa381)

1) There is an outlier in our dataset where a user smokes approximately 35 cigarettes at the age of 27.

2) Our data appears to be scattered.
   
#### 4.8 Boxplot to visually analyze the distribution

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/c5a59816-dca0-41ad-bac2-19a089b0c959)

We plotted a box plot to visually analyze the distribution and spread of the 'Age' and 'Smokes' variables in the dataset. We were not able to see any outliers in our plot

#### 4.9 Heatmap

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/56bb8c13-49e6-44e7-8969-38cb933a3297)

Heatmap shows us correlation between our numeric columns

# 5. Polishing our data for model training

#### 5.1 Outlier detection and treating using Z-score

Shape of our original dataset :

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/4b8baa6b-60e4-450a-920e-58fdb2c0ac41)

Shape after treating outliers :

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/4b8baa6b-60e4-450a-920e-58fdb2c0ac41)

No outliers were detected.

#### 5.2 One-Hot Encoding :

We are doing One-Hot Encoding to convert categorical variables are converted into binary vectors.

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/a50eb7dc-4034-45dc-8983-5174475de613)

#### 5.3 Standarization

We need to bring all the values of each column onto a common scale which will help us to train our model effiency.

Our x_train after standarization :

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/32f9a6dc-f184-416f-b549-c5d932e08c8b)

Our x_test after standarization :

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/211856bb-5483-44fc-89f3-3e73e115cb6c)

# 6. Model Building and Model Evaluation

#### 6.1 Calculating accuracy of our models

We used Logisitc Regression, Decision Tree, Random Forest and Support Vector Machine to train our model.

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/f34ccbe2-42eb-4649-b02c-f0fd63d79669)

We are getting high accuracy for all four of our machine learning models.

#### 6.2 Calculating Precision, R2 score and F1 score

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/0334b109-77ea-4066-b53b-1d762ae4d107)

Precision, Recall and F1-score of all our four models is excellent

#### 6.3 Calculating confusion Matrix

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/0f901358-70a9-4269-b579-cdf7b34dcf48)

Our confusion matrix is working well


# 7. Actual vs. Predicted Values:

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/17bcb751-e8d8-4178-a098-0bf4fb54fc2d)

![test2](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/8286ab87-7994-4748-a256-631d6bf03da1)


# 8. Conclusion : 

1. Lung cancer is caused among people who are above approximately 39 years old.
2. All the 4 models : Logisitc Regression, Decision Tree, Random Forest and Support Vector Machine used provided high accuracy.

# 9. References : 

General:

[1] Anaconda Distribution https://www.anaconda.com/

[2] Python Software Foundation https://www.python.org/

[3] seaborn: statistical data visualization https://seaborn.pydata.org/index.html#

[4] matplotlib: Python plotting library https://matplotlib.org/

[5] "Employee Compensation and Satisfaction Insights" Dataset from data.world https://github.com/mwaskom/seaborn-data/blob/master/tips.csv

[6] scikit-learn: Machine Learning in Python https://scikit-learn.org/stable/index.html

[7] statsmodels: Statistics in Python https://www.statsmodels.org/stable/index.html

[8] scipy.stats : Statistics with SciPy https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html

Exploratory data analysis:

[9] Exploratory Statistical Data Analysis with a Real data set using Pandas https://towardsdatascience.com/exploratory-statistical-data-analysis-with-a-real-data set-using-pandas-208007798b92

[10] How to investigate a data set with Python https://towardsdatascience.com/hitchhikers-guide-to-exploratory-data-analysis-6e8d896d3f7e

[11] Data analysis with Python https://medium.com/@onpillow/01-investigate-tmdb-movie-data set-python-data-analysis-project-part-1-data-wrangling-3d2b55ea7714

[12] Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython. Wes McKinney. ISBN-13: 978-1491957660 ISBN-10: 1491957662

[13] Pandas In 10 Minutes || Wes McKinney https://www.youtube.com/watch?v=1MGCD8SQp3k

[14] Good description of quartiles on Seaborn plots https://towardsdatascience.com/analyze-the-data-through-data-visualization-using-seaborn-255e1cd3948e

Regression:

[15] Ordinary Least Squares in statsmodels https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html

[16] Generalized Linear Models in scikit-learn https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares

[17] How to run Linear regression in Python scikit-Learn https://bigdata-madesimple.com/how-to-run-linear-regression-in-python-scikit-learn/

[18] A beginner’s guide to Linear Regression in Python with Scikit-Learn https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

[19] Regression Analysis: How Do I Interpret R-squared and Assess the Goodness-of-Fit? https://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit

[20] Python and R Tips To Learn Data Science: Pearson and Spearman Correlation in Python https://cmdlinetips.com/2019/08/how-to-compute-pearson-and-spearman-correlation-in-python/

















