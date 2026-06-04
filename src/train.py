'''
End to end training script for the dev salary prediction model.

OUTPUTS:
1. Saved pipeline.
2. Cleaned dataset.
'''
import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

from preprocess import load_and_clean, get_feature_columns, LOG_TARGET
from evaluation import evaluate_model, plot_predictions,  print_observations

# adding /src to path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

#configuration
RAW_DATA_PATH = 'data/raw/results.csv'
PROCESSED_DATA = 'data/cleaned/cleaned_result.csv'
MODEL_OUTPUT_PATH = 'models/salary_pipeline.pk1'

RANDOM_STATE = 42
TEST_SIZE = 0.2

XGBOOST_PARAMS = {
    'n_estimators' : 600,
    'max_depth' : 6,
    'learning_rate' :0.03,
    'random_state' : RANDOM_STATE,
    'verbosity' : 0,
    'subsample' : 0.8,
    'colsample_bytree' : 0.8,
    'tree_method' : 'hist',  # makes execution fast for large datasets
    'reg_alpha' : 0.05, #L1
    'reg_lambda' : 1.0 #L2
}

def build_preprocessor(cat_cols: list, num_cols: list) -> ColumnTransformer:
    '''
    Build and return the scikit-learn ColumnTransformer
    
    Numeric pipeline:
        1. Simple Imputer - fill NaN with median
        2. StandardScaler - center and scale
        
    Categorical pipeline:
        1. Simple Imputerb - fill Nan with most frequent value.
        2. OneHotEncoder - convert categories into binary columns, handle_unknown = 'ignore', unseen categories become zeros
        
    '''
    num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', TargetEncoder(smooth='auto', target_type='continuous'))
    ])
    
    preprocesser = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cols', cat_pipeline, cat_cols)
    ], remainder='drop')
    
    return preprocesser

def build_pipeline(cat_cols: list, num_cols: list) -> Pipeline:
    '''
    Combine preprocessor  + Model into one sklearn pipeline
    '''
    preprocessor = build_preprocessor(cat_cols, num_cols)
    model = XGBRegressor(**XGBOOST_PARAMS)
    
    pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
    ])
    
    return pipeline

def main():
    print('Developer salary prediction - training')
    
    # 1. Load and clean the data
    df = load_and_clean(RAW_DATA_PATH)
    
    # Save preprocessed data
    df.to_csv(PROCESSED_DATA, index=False)
    print(f"preprocessed data saved to: {PROCESSED_DATA}\n \n")
    
    # 2. Split features and target
    X =df.drop(columns = [LOG_TARGET])
    y = df[LOG_TARGET]


    cat_cols, num_cols = get_feature_columns(df)
    print(f"Numeric features: {num_cols}")
    print(f"Categorical features: {cat_cols}\n")
    
    # 3. Train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=TEST_SIZE,random_state=RANDOM_STATE)
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}\n")
    
    # 4. Build and train pipeline
    print("Building pipeline...")
    pipeline = build_pipeline(cat_cols, num_cols)
    
    print('Traning XGBoost model')
    pipeline.fit(X_train, y_train)
    print('Training complete. \n')
    
    # 5. Evaluate the model
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    train_metrics = evaluate_model(y_train, y_pred_train, title='Training set performance')
    test_metrics = evaluate_model(y_test, y_pred_test, title= 'Test set performance') 
    
    print_observations(test_metrics)
    print_observations(train_metrics)
    plot_predictions(y_test.values, y_pred_test, save_path= 'data/predictions_plot.png')
    
    # 6. Save the pipeline
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"\n Model saved to: {MODEL_OUTPUT_PATH}")
    
    # Example prediction
    print('\n Sample prediction: \n')
    
    '''
    
    
    sample =  pd.DataFrame([{
        'Country' : "Ukraine",
        'YearsCode' : 10.0,
        'EdLevel' : "Bachelor's",
        'Employment' : "Full-time",
        'LanguageHaveWorkedWith' : 4
    }])
    
    pred = pipeline.predict(sample)[0]
    print(f"input: {sample.to_dict(orient='records')[0]}")
    
    mae = test_metrics['mae']
    print(f"Predicted salary: ${pred:,.0f} +/- ${mae}")
    
    '''
    
    print("\n Training script complete. \n")
    
if __name__ == '__main__':
        main()