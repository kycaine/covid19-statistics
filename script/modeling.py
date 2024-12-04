import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def train_model(df, target_column, feature_columns, output_dir):
    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    metrics_output_path = os.path.join(output_dir, 'model_metrics.txt')
    with open(metrics_output_path, 'w') as f:
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"R-squared: {r2}\n")

    predictions_output_path = os.path.join(output_dir, 'predictions.csv')
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions_df.to_csv(predictions_output_path, index=False)

    return model, X_test, y_test, y_pred

def plot_results(y_test, y_pred, output_dir):
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')

    plot_output_path = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(plot_output_path)
    plt.close()
