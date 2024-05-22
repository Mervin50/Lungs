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







3.4 Distribution of Base salary

3.5 Distribution of 2020 Longetivity pay

3.6 Top 10 Departments based on average base salary

3.7 Top 5 divisions based on 2020 overtime pay distribution

3.8 Distribution of Base Salary based on Gender

3.9 Heatmap of Top 10 Divisions vs Gender

Data Cleaning

4.1 Outlier Detection and Treatment

4.1.1 Z-score Method

4.1.2 Winsorizing Technique

4.1.3 Implementation

4.1.4 Winsorizing explanation of code
Data Splitting and Preprocessing

Model Selection and Evaluation

6.1 MSE

6.1.1 MSE model performance

6.1.2 MSE model accuracy evaluation
6.2 R^2 score

6.2.1 R^2 scores of all 3 models

6.2.2 R^2 score evualation
6.3 Precision, Recall and F1-score

6.3.1 Precision explanation

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









