'''
data cleaning utilities for developer salary raw csv.
will use these functions in train.py and app.py
'''
import pandas as pd
import numpy as np

# Constants.
TARGET = 'ConvertedCompYearly'
Salary_min = 10000
Salary_max = 500000
Selected_features = ['Country','YearsCode', 'EdLevel', 'Employment', 'LanguageHaveWorkedWith']
Top_countries = 15

def clean_years_code(series: pd.Series) -> pd.Series:
    '''Converting YearsCode to numeric'''
    series = pd.to_numeric(series)
    return series

def clean_education(series: pd.Series) -> pd.Series:
    '''Standardizing Edlevel into a set of clean categories'''
    mapping = {
        "Bachelor’s degree (B.A., B.S., B.Eng., etc.)" :"Bachelor's",
        "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)" : "Masters",
        "Some college/university study without earning a degree" : "Some college",
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)" : "High school",
        "Associate degree (A.A., A.S., etc.)" : "Associate's",
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Professional",
        "Other (please specify)" : "Other",
        "Primary/elementary school" : "Primary school"
    }
    return series.map(mapping).fillna('Other')

def clean_employment(series: pd.Series) -> pd.Series:
    '''Cleaning  employment column'''
    
    def simplify(val):
        if pd.isna(val):
            return np.nan
        val = str(val)
        if 'employed'in val.lower() or 'full-time' in val.lower():
            return 'Full-time'
        elif 'Independent contractor, freelancer, or self-employed' in val:
            return 'Freelance/Self-employed'
        elif 'student' in val.lower():
            return 'Student'
        else:
            return 'Other'
    
    return series.apply(simplify)

def count_languages(series : pd.Series) -> pd.Series:
    '''Convert comma separated language list into count
    
       Example: 'Dart;HTML/CSS;JavaScript;TypeScript' -> 4
    '''
    def _count(val):
        if pd.isna(val) or val == "":
            return np.nan
        return len(str(val).split(";"))
    return series.apply(_count)

def group_rare_countries(series : pd.Series, top_n: int = Top_countries) -> pd.Series:
    """
    Keeping only the top N most common countries (default = 15). Replace all others with 'Other'.
    """
    top_countries = series.value_counts().head(top_n).index.tolist()
    return series.apply(lambda x : x if x in top_countries else 'Other')

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Loading the raw Stack Overflow survey csv and return a clean dataframe.
    The DF will be ready for the scikit-learn pipeline.
    
    pass a filepath to the raw csv file
    
    returns pd.DataFrame(Df with features + target column)
    """
    df = pd.read_csv(filepath, low_memory=False)
    print(f'This is the raw shape: {df.shape}')
    
    #step 1 - Keep rows with a valid salary
    df = df.dropna(subset = [TARGET])
    df = df[df[TARGET].between(Salary_min, Salary_max)]
    print(f'Shape after salary filter: {df.shape}')
    
    # Step 2 - Check wheather the features and target columns exist
    cols_needed = Selected_features + [TARGET]
    cols_available = [c for c in cols_needed if c in df.columns]
    
    missing_cols = set(cols_needed) - set(cols_available)
    if missing_cols:
        print(f'Columns not found in dataset: {missing_cols}')
        
    df = df[cols_available].copy()    
    print(f'Selected {len(cols_available)} columns, expecting 6')
    
    # Step 3 Cleaning Specific columns
    if 'YearsCode' in df.columns:
        df['YearsCode'] = clean_years_code(df['YearsCode'])
        
    if 'EdLevel' in df.columns:
        df['EdLevel'] = clean_education(df['EdLevel'])
        
    if 'Employment' in df.columns:
        df['Employment'] = clean_employment(df['Employment'])
        
    if 'LanguageHaveWorkedWith' in df.columns:
        df['LanguageHaveWorkedWith'] = count_languages(df['LanguageHaveWorkedWith'])  
        
    if 'Country' in df.columns:
        df['Country'] = group_rare_countries(df['Country'])
        
    # Drop rows where all the features are NaN (Handling this edge case)
    df = df.dropna(how='all')
    print(f'Clean data Shape: {df.shape}')
    print(f'Missing values per column: \n {df.isna().sum().to_string()} \n')
    
    return df

def get_feature_columns(df: pd.DataFrame) -> tuple[list,list]:
    '''
    Returns (Categorical columns, numeric columns) from the cleaned DF, excluding the target variable
    '''
    non_target = [c for c in df.columns if c != TARGET]
    cat_cols = df[non_target].select_dtypes(include=['object','category']).columns.to_list()
    num_cols = df[non_target].select_dtypes(include=['number']).columns.to_list()
    
    return cat_cols, num_cols