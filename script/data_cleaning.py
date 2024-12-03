import pandas as pd
from fuzzywuzzy import process
import os

def clean_and_filter_data(file_path, valid_countries, threshold=80):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None, None
    else:
        df = pd.read_csv(file_path)
        print("File loaded successfully.")
        
        def find_closest_country(country, valid_countries, threshold=80):
            if pd.notna(country) and isinstance(country, str):
                country = country.strip()
                closest_match, score = process.extractOne(country, valid_countries)
                if score >= threshold:
                    return closest_match
            return None  

        df['country'] = df['country'].apply(lambda x: find_closest_country(x, valid_countries))

        numeric_fields = ['confirmed_cases', 'deaths', 'recoveries', 'total_confirmed', 'total_deaths', 'total_recoveries']
        valid_df = df[(
            df['country'].notna() &  
            df[numeric_fields].ge(0).all(axis=1
        )]

        invalid_df = df[~df.index.isin(valid_df.index)]

        output_dir = os.path.join(os.path.dirname(file_path), '..', 'output', 'data_cleaning')
        os.makedirs(output_dir, exist_ok=True)  

        valid_output_file = os.path.join(output_dir, 'valid_filtered_covid19_data.csv')
        valid_df.to_csv(valid_output_file, index=False)
        print(f"Valid data saved at {valid_output_file}")

        invalid_output_file = os.path.join(output_dir, 'invalid_filtered_covid19_data.csv')
        invalid_df.to_csv(invalid_output_file, index=False)
        print(f"Invalid data saved at {invalid_output_file}")

        return valid_df, invalid_df
