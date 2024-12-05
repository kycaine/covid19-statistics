import pandas as pd
import math
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def optimize_model(data, output_dir):
    print("Data before cleaning:")
    print(data.head())

    data_clean = data.dropna(subset=['Actual', 'Predicted'])

    print("Data after cleaning:")
    print(data_clean.head())

    y = data_clean['Actual']
    predictions = data_clean['Predicted']

    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = math.sqrt(mse)  
    r2 = r2_score(y, predictions)

    results = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }

    results_file = os.path.join(output_dir, 'model_optimization_results.txt')
    with open(results_file, 'w') as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")
    
    print(f"OPtimization results saved to {results_file}")
    return results
