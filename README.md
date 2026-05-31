## Developer Salary Prediction.

This project uses data from the Stack Overflow Developer Survey to build a machine learning model capable of predicting a developer's annual salary based on demographic, educational, employment, and technical experience information

[Stack Overflow](https://survey.stackoverflow.co/) - 2025 Dataset

## The project follows a complete machine learning workflow:
- Exploratory Data Analysis (EDA)
- Data Cleaning & Feature Engineering
- Data Preprocessing
- Model Training
- Model Evaluation
- Model Serialization for Deployment

The final model is built using XGBoost Regressor

## Project Structure
```
dev_salary_prediction/
│
├── data/
│   ├── raw/
│   │   └── results.csv
│   │
│   ├── cleaned/
│   │   └── cleaned_result.csv
│   │
│   └── predictions_plot.png
│
├── models/
│   └── salary_pipeline.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── preprocessing.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── evaluation.py
│   └── train.py
│
├── requirements.txt
└── README.md
```

## Target Variable:
ConvertedCompYearly - represents the respondent's annual compensation converted to USD.

- To improve model quality, salaries are filtered to >= 10000, <= 500000

## Exploratory Data Analysis (01_eda.ipynb)
The EDA notebook is responsible for understanding the dataset before modeling.

- Loading raw survey data.
- Exploring dataset dimensions.
- Inspecting missing values.
- Analyzing salary distribution.
- Examining developer demographics.
- Exploring education levels.
- Investigating employment status and analyzing programming language usage.
- Identifying outliers and alary trends by experience.

## Data Preprocessing Notebook (preprocessing.ipynb)
Purpose is to prepare a cleaned dataset for machine learning.
### Selected features:
- Country
- YearsCode
- EdLevel
- Employment
- LanguageHaveWorkedWith

## Preprocess.py 
Contains reusable data cleaning functions used throughout the project.

1. Removing missing salaries, unrealistic salary values and education Standardization
2. Converting long education descriptions into simplified categories e.g. - Bachelor’s degree to Bachelor's
3. Employment cleaning: Groups employment status into Full-time, Freelance/Self-employed, Student or Other.
4. Language Feature Engineering : Converts programming language lists into counts. eg Python;SQL;JavaScript;HTML/CSS becomes 4
5. Country Grouping : Keeps only the top 15 countries. All others are converted to Other.
6. Returning categorical_columns and numeric_columns for pipeline creation.

## evaluation.py
Contains utilities for measuring model performance.
### Metrics Computed
1. Mean Absolute Error (MAE)
2. Root Mean Squared Error (RMSE)
3. r2 Score

## Visualization Functions
- Actual vs Predicted Plot
- Actual Salary vs Predicted Salary

## train.py
contains end-to-end training script.
### Main function.
1. Load and Clean Data
2. Save Processed Dataset
3. Split Data
4. Build Preprocessing Pipeline
5. Train XGBoost Model
6. Evaluate Performance
7. Save Trained Pipeline

