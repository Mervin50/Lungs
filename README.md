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



6.3.2 Recall explanation

6.3.3 F1-score explanation

6.3.4 F1, Precision and Recall score of Linear Regression

6.3.5 F1, Precision and Recall score of Random Forest

6.3.6 F1, Precision and Recall score of SVM
6.4 Confusion Matrix

6.4.1 Confusion Matrix of Random Forest and SVM
Conclusion

References




# 1. Introduction:
This README describes work done on the "Employee Compensation and Satisfaction Insights" Dataset. Resources used include Python and associated packages Google Colab, matplotlib, Seaborn, scikit-learn, statsmodels, and SciPy. We have addressed the regression challenge to uncover the factors impacting salaries, employing Support Vector Machine, Random Forest, and Linear Regression.

# 2. Description of the dataset:

2.1 Head of our Dataset:
The very first step is always to check if the data needs cleaning by looking for duplicate rows, zero values or NaNs where they shouldn't be etc. The head of our dataset looks like:

![Screenshot 2024-05-22 131152](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/f2726f91-9bda-4453-a483-3ac0367b11bb)

2.2 Tail of our Dataset:
The tail of our dataset looks like : 

![Screenshot 2024-05-22 132115](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/0270b766-bcb5-4109-9e14-48c67f9b4811)

Visually, we found out that there are few NaN values in our dataset

# 3. Preprocessing and Data Cleaning for EDA

3.1 Detecting Null values in our dataset:

![Screenshot 2024-05-22 132813](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/667e87e4-f5cc-4458-b100-74d6e5615383)

There are 2 each null values in Smokes and Alkhol columns. There is one null value in Age column. 

3.2 Evaluating the best technique for handling null values:

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/50a20522-fae1-4db2-9f70-4ff7ef46dd1d)

Plot is left skewed, hence we will use median technique for age column.

![test2](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/83fe3959-52b8-4dad-a7aa-4b0f68a4cda5)

 Plot is right skewed, hence we will use median technique for Smokes column.

![test3](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/e01501b0-2c3f-41b8-ad67-2b96ce8512ae)

Plot is slightly left skewed, hence we will use median technique for Alkhol column.

3.3 After Treating Null values:

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/21b08ffb-e135-4110-8327-5b1fcae24ba5)

We have replaced our null values with median values in three columns : Age, Smokes and Alkhol

3.4 Assessing Dataset Balance: 

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/7fdb2e12-a8d0-4743-b83f-bc26df76f5dd)

We have ratio of 32-29, so its fairly balanced dataset, so no further action is needed to balance the dataset

3.5 Label Encoder:

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/02431943-2216-41dd-8d5c-fecb38b6e757)

We applied Label Encoder as we are solving classification problem and it helps for our model for better prediction. Label Encoder to transform categorical data into numerical form.

# 4. Exploratory Data Analysis

4.1 Overview of dataset using statistics:

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/d5b38c22-cae2-4f8d-bc3c-689f69cd1e0e)

Approximately 50% people who smoke are below age 40. This group smokes approximately 15 cigarettes and consumes 3 drinks a day.

4.2 Barplot to know distribution of age column

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/5be14dbc-dd4b-417b-b8ff-dcdf5bf1d41f)

Majority of the people in our dataset are from age groups 20 to 39.

4.3 Barplot to know Number of Cigarettes smoked by different age groups

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/f800bbae-ec5b-42fb-9045-b0ef96718cae)

Approximately 19 cigarettes are smoked by people of age group 50 to 59

4.4 Pie chart to illustrate the distribution of daily drink consumption

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/26cc1388-f290-4be4-bce4-04407baf40f5)

Approximately 33% people take 2 to 3 drinks per day

4.5 Barplot to know smoking and drinking habits

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/0062e1db-4d16-46cf-958f-2ea3fef6cab5)

From our barplot, we come to know that majority of people do both smoking and drinking.

4.6 Histogram to explore the age distribution of individuals affected by lung cancer

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/2bc678db-8207-4ee4-aac0-ef8afa29d600)

Majority of people affected by Lung cancer are approximately above age 47.

4.7 Scatter plot to understand relationship between age and smoking habits for the individuals in the dataset

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/b637479c-0ede-4527-98ce-6a261c6aa381)

1) There is an outlier in our dataset where a user smokes approximately 35 cigarettes at the age of 27.

2) Our data appears to be scattered.
   
4.8 Boxplot to visually analyze the distribution

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/c5a59816-dca0-41ad-bac2-19a089b0c959)

We plotted a box plot to visually analyze the distribution and spread of the 'Age' and 'Smokes' variables in the dataset. We were not able to see any outliers in our plot

4.9 Heatmap

![test1](https://github.com/Mervin50/ML_Project2_LungCancer_Classification/assets/167336864/56bb8c13-49e6-44e7-8969-38cb933a3297)

Heatmap shows us correlation between our numeric columns









