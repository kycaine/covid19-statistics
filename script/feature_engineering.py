import pandas as pd
import os
import numpy as np

def feature_engineering(df_cleaned):
    df_cleaned['case_fatality_rate'] = df_cleaned['deaths'] / df_cleaned['confirmed_cases']
    df_cleaned['recovery_rate'] = df_cleaned['recoveries'] / df_cleaned['confirmed_cases']
    return df_cleaned

def save_featured_data(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'covid_feature_engineered.csv'), index=False)
