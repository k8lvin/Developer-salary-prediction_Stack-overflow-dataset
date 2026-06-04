'''
data cleaning utilities for developer salary raw csv.
will use these functions in train.py and app.py
'''
import pandas as pd
import numpy as np

# Constants.
TARGET = 'ConvertedCompYearly'
LOG_TARGET = 'log_salary'
Salary_min = 10000
Salary_max = 500000
Selected_features = ['Country','YearsCode', 'EdLevel', 'Employment', 'LanguageHaveWorkedWith',
                     #new selected features, v2
                     'DevType', #developer's role
                     'OrgSize', #Company's size
                     'RemoteWork', #Hybrid or remote
                     'WorkExp', #Years of professional experience
                     'Industry',#tech, finance, healthcare, etc
                     'Age',
                     'ICorPM', #Individual contributor or manager
                     'DatabaseHaveWorkedWith',
                     'PlatformHaveWorkedWith',
                     'ToolCountWork'
                     ]
Top_countries = 25

# features to target encode
TARGET_ENCODE_FEATURES = ("Country", "devType", "Industry")

# Ordinal mappings
ED_LEVEL_ORDINAL: dict[str, int] =  {
        "Bachelor's" : 5,
        "Masters" : 6,
        "Some college" : 3,
        "High school" : 2,
        "Associate's" : 4,
        "Professional" : 7,
        "Other" : 0,
        "Primary school" : 1
    }

REMOTE_ORDINAL : dict[str, int] = {
    "In person" : 0,
    "Hybrid" : 1,
    "Remote" : 2,
    " Other" : 1
}

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
    incomplete = series.map(mapping).fillna("Other")
    return incomplete.map(ED_LEVEL_ORDINAL)

def clean_work_exp(series: pd.Series) -> pd.Series:
    series = series.copy()
    series = pd.to_numeric(series, errors='coerce')
    return series

def clean_age(series: pd.Series) -> pd.Series:
    "Mapping age bands to ordinal integers"
    mapping = {
        "18-24 years old" : 0,
        "25-34 years old" : 1,
        "35-44 years old" : 2,
        "45-54 years old" : 3,
        "55-64 years old" : 4,
        "65 years or older" : 5,
        "Prefer not to say" : np.nan 
    }
    return series.map(mapping)

def clean_org_size(series: pd.Series) -> pd.Series:
    " Mapping org-size bands to ordinal intergers"
    mapping = {
        "Just me - I am a freelancer, sole proprietor, etc.": 0,
        "Less than 20 employees":1,
        "20 to 99 employees": 2,
        "100 to 499 employees": 3,
        "500 to 999 employees": 4,
        "1,000 to 4,999 employees": 5,
        "5,000 to 9,999 employees": 6,
        "10,000 or more employees": 7,
        "I don't know": np.nan
    }
    return series.map(mapping)

def clean_icorpm(series: pd.Series) -> pd.Series:
    '''Mapping ICorPM to binary 1 - manager, 0 for IC'''
    def _map(val):
        if pd.isna(val):
            return np.nan
        v = str(val).lower()
        if "manager" in v or "lead" in v:
            return 1
        return 0
    return series.apply(_map)

def clean_industry(series: pd.Series) -> pd.Series:
    "keep only top industries(10) merge the rest into 'other'"
    top = series.value_counts().head(10).index.tolist()
    return series.apply(lambda x: x if x in top else 'Other').fillna('Other')

def clean_dev_type(series: pd.Series) -> pd.Series:
    "picking the primary dev role"
    def _primary(val):
        if pd.isna(val):
            return "Other"
        low = str(val).lower()
        if "full-stack" in low:
            return "Full-stack"
        if "back-end" in low:
            return "Back-end"
        if "front-end" in low:
            return "Front-end"
        if "data scientist" in low or "machine learning" in low or "ml" in low:
            return "Data/ML"
        if "data engineer" in low or "data analyst" in low:
            return "Data/ML"
        if "devops" in low or "cloud" in low or "site reliability" in low:
            return "DevOps/Cloud"
        if "mobile" in low:
            return "Mobile"
        if "embedded" in low or "hardware" in low:
            return "Embedded/Hardware"
        if "security" in low:
            return "Security"
        if "manager" in low or "executive" in low or "director" in low:
            return "Management"
        return "Other"
    return series.apply(_primary)

def count_items(series : pd.Series) -> pd.Series:
    "count semi colon separated items in a column: NaN if blank"
    def _count(val):
        if pd.isna(val) or val == "":
            return np.nan
        return len(str(val).split(";"))
    
    return series.apply(_count)

#languages that consistently pay above average in so salary surveys
HIGH_PAY_LANGUAGES ={
    "Go", "Rust", "Scala", "Elixir", "Clojure", "Kotlin", "Swift", "F#", "Erlang", "Zig", "OCmal", "Haskell"
}

def has_high_pay_languages(series: pd.Series) ->pd.Series:
    """return 1 if the respondent knows any high paying language"""
    def _check(val):
        if pd.isna(val):
            return 0
        langs = {lang.strip() for lang in str(val).split(";")}
        return 1 if langs & HIGH_PAY_LANGUAGES else 0 # & - intersection operator
    return series.apply(_check)
 

def clean_remote(series: pd.Series) -> pd.Series:
    '''Map remote work into ordinal integers'''
    mapping = {
        "Remote" : "Remote",
        "Hybrid (some in-person, leans heavy to flexibility)" : "Hybrid",
        "Hybrid (some remote, leans heavy to in-person)" : "Hybrid",
        "In-person" : "In-person",
        "Your choice (very flexible, you can come in when you want or just as needed)" : np.nan
    }
    incomplete = series.map(mapping).fillna("Other")
    return incomplete.map(REMOTE_ORDINAL)

def add_interaction_features(df):
        df['yearsCode_sq'] = df['YearsCode'] ** 2
        df['WorkExp_sq'] = df['WorkExp'] ** 2
        df['Exp_ratio'] = df['WorkExp']/ (df['YearsCode'] + 1)
        df['Tech_breadth'] = (df['LanguageHaveWorkedWith'] + df['DatabaseCount'] + df['PlatformCount']).fillna(0)
        return df
    
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
    
    #Step 1.5 - v2 of model 
    df['log_salary'] = np.log1p(df[TARGET])
    
    # Step 2 - Check wheather the features and target columns exist
    cols_needed = Selected_features + ['log_salary']
    cols_available = [c for c in cols_needed if c in df.columns]
    
    missing_cols = set(cols_needed) - set(cols_available)
    if missing_cols:
        print(f'Columns not found in dataset: {missing_cols}')
        
    df = df[cols_available].copy()    
    print(f'Selected {len(cols_available)} columns, expecting 16')
    
    # Step 3 Cleaning Specific columns
    if 'YearsCode' in df.columns:
        df['YearsCode'] = clean_years_code(df['YearsCode'])
        
    if 'workExp'in df.columns:
        df['WorkExp'] = clean_work_exp(df['WorkExp'])    
        
    if 'EdLevel' in df.columns:
        df['EdLevel'] = clean_education(df['EdLevel'])
        
    if 'Employment' in df.columns:
        df['Employment'] = clean_employment(df['Employment'])
        
    if 'Age' in df.columns:
        df['Age'] = clean_age(df['Age'])
    
    if 'OrgSize' in df.columns:
        df['Age'] = clean_org_size(df['OrgSize'])
    
    if 'RemoteWork' in df.columns:
        df['RemoteWork'] = clean_remote(df['RemoteWork'])
        
    if 'Industry' in df.columns:
        df['Industry'] = clean_industry(df['Industry'])
        
    if 'DevType' in df.columns:
        df['DevType'] = clean_dev_type(df['DevType'])
            
    if 'ICorPM' in df.columns:
        df['ICorPM'] = clean_icorpm(df['ICorPM'])
            
    if 'LanguageHaveWorkedWith' in df.columns:
        df['has high paying language'] = has_high_pay_languages(df['LanguageHaveWorkedWith'])
        df['LanguageHaveWorkedWith'] = count_languages(df['LanguageHaveWorkedWith'])  
    
    if 'DatabaseHaveWorkedWith' in df.columns:
        df['DatabaseCount'] = count_items(df['DatabaseHaveWorkedWith'])
        df = df.drop(columns=['DatabaseHaveWorkedWith'])
    
    if 'PlatformHaveWorkedWith' in df.columns:
        df['PlatformCount'] = count_items(df['PlatformHaveWorkedWith'])
        df = df.drop(columns=['PlatformHaveWorkedWith'])
        
    if 'ToolCount' in df.columns:
        df['ToolCountWork'] = pd.to_numeric(df['ToolCountWork'], errors = 'coerce')
               
    if 'Country' in df.columns:
        df['Country'] = group_rare_countries(df['Country'])
        
    EMPLOYMENT_KEEP = ['Full-time','Freelance/Self-employed']    
        
    #  step 4 filter employment (keeping only fulltime and freelance)
    if 'Employment' in df.columns:
        before = len(df)
        df = df[df['Employment'].isin(EMPLOYMENT_KEEP)]   
        df['Employment'] = (df['Employment'] == "Full-time").astype(int)
         
        print(f"Employment filter: {before} -> {len(df)} rows, "
              f"we only kept Full-time and freelance")
        
    # step 5 adding interaction features (polynormial features)
    df = add_interaction_features(df)
        
    # step 6 Drop rows where all the features are NaN (edge case)
    df = df.dropna(how='all')
    
    print(f'Clean data Shape: {df.shape}')
    print(f'Missing values per column: \n {df.isna().sum().to_string()} \n')
    
    return df

def get_feature_columns(df: pd.DataFrame) -> tuple[list,list]:
    '''
    Returns (Categorical columns, numeric columns) from the cleaned DF, excluding the target variable
    '''
    non_target = [c for c in df.columns if c != 'log_salary']
    target_enc = [c for c in TARGET_ENCODE_FEATURES if c in non_target] 
    
    num_cols = df[non_target].select_dtypes(include=['number']).columns.to_list()
    
    print(f'Expecting {len(non_target)} got {len(target_enc) + len(num_cols)} columns')
    return target_enc, num_cols