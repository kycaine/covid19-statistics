import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

def evaluate_model(input_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    y_true = input_df['Actual']
    y_pred = input_df['Predicted']

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse) 
    r2 = r2_score(y_true, y_pred)

    evaluation_results = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }

    results_file = os.path.join(output_dir, 'model_evaluation_results.txt')
    with open(results_file, 'w') as f:
        for metric, value in evaluation_results.items():
            f.write(f"{metric}: {value}\n")
    
    print(f"Evaluation results saved to {results_file}")
    return evaluation_results
