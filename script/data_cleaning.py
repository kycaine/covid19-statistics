import pandas as pd
from fuzzywuzzy import process
import os

def find_closest_country(country, valid_countries, threshold=80):
    if pd.notna(country) and isinstance(country, str):
        country = country.strip()
        closest_match, score = process.extractOne(country, valid_countries)
        if score >= threshold:
            return closest_match
    return None

def clean_and_filter_data(file_path, output_dir, valid_countries, threshold=80):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None, None
    else:
        df = pd.read_csv(file_path)
        print("File loaded successfully.")
        
        if 'country' not in df.columns:
            print("Error: 'country' column not found in the dataset.")
            return None, None

        df['country'] = df['country'].apply(lambda x: find_closest_country(x, valid_countries, threshold))

        unmatched_countries = df[df['country'].isna()]
        if not unmatched_countries.empty:
            print(f"Unmatched countries (score < threshold):\n{unmatched_countries[['country']]}")
        
        numeric_fields = ['deaths', 'recoveries']
        df[numeric_fields] = df[numeric_fields].fillna(0)
        
        valid_df = df[
            df['country'].notna() &
            (df['deaths'] >= 0) & 
            (df['recoveries'] >= 0)
        ]
        
        invalid_df = df[
            ~(
                df['country'].notna() & 
                (df['deaths'] >= 0) & 
                (df['recoveries'] >= 0)
            )
        ]
        
        valid_output_csv = os.path.join(output_dir, 'valid_filtered_covid19_data.csv')
        valid_df.to_csv(valid_output_csv, index=False)
        # valid_output_excel = os.path.join(output_dir, 'valid_filtered_covid19_data.xlsx')
        # valid_df.to_excel(valid_output_excel, index=False)
        print(f"Filtered valid data saved to {output_dir}")
        
        
        invalid_output_csv = os.path.join(output_dir, 'invalid_filtered_covid19_data.csv')
        invalid_df.to_csv(invalid_output_csv, index=False)
        # invalid_output_excel = os.path.join(output_dir, 'invalid_filtered_covid19_data.xlsx')
        # invalid_df.to_excel(invalid_output_excel, index=False)
        print(f"Filtered invalid data saved to {output_dir}")
        
        return valid_df, invalid_df
