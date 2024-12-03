import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def summary_statistics(df):
    summary = df.describe()
    summary.to_csv('output/eda/summary_statistics.csv')
    print("Summary statistics saved to 'summary_statistics.csv'")

def plot_histograms(df, columns):
    for column in columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True)
        plt.title(f'{column} Distribution')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        os.makedirs('output/eda/histograms', exist_ok=True)
        plt.savefig(f'output/eda/histograms/{column}_histogram.png')
        plt.close()
        print(f"Histogram for {column} saved.")

def plot_boxplots(df, columns):
    for column in columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[column])
        plt.title(f'{column} Boxplot')
        plt.xlabel(column)
        os.makedirs('output/eda/boxplots', exist_ok=True)
        plt.savefig(f'output/eda/boxplots/{column}_boxplot.png')
        plt.close()
        print(f"Boxplot for {column} saved.")

def plot_correlation_heatmap(df):
    df_cleaned = df.apply(pd.to_numeric, errors='coerce')
    
    nan_columns = df_cleaned.columns[df_cleaned.isna().any()].tolist()
    if nan_columns:
        print(f"Columns with non-numeric values or NaNs after conversion: {nan_columns}")
        print(df_cleaned[nan_columns].isna().sum())  
        print(df_cleaned[df_cleaned[nan_columns].isna().any(axis=1)]) 

    df_cleaned = df_cleaned.dropna()

    numeric_df = df_cleaned.select_dtypes(include=[np.number])

    if numeric_df.empty:
        print("No numeric columns found for correlation heatmap.")
        return

    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')

    os.makedirs('output/eda', exist_ok=True)
    plt.savefig('output/eda/correlation_heatmap.png')
    plt.close()
    print("Correlation heatmap saved.")


def perform_eda():
    df_cleaned = pd.read_csv('output/data_cleaning/valid_filtered_covid19_data.csv')
    df_cleaned = df_cleaned.select_dtypes(include=['number'])
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors='coerce')
    df_cleaned = df_cleaned.dropna()

    print("Columns after converting to numeric and dropping NaNs:")
    print(df_cleaned.head())
    print(df_cleaned.info())

    summary_statistics(df_cleaned)
    plot_histograms(df_cleaned, ['confirmed_cases', 'deaths', 'recoveries'])
    plot_boxplots(df_cleaned, ['deaths', 'recoveries'])
    plot_correlation_heatmap(df_cleaned)

perform_eda()

