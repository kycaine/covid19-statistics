import subprocess
import os
import asyncio

notebooks = [
    'notebook/data_cleaning.ipynb',
    'notebook/eda.ipynb',
    'notebook/feature_engineering.ipynb',
    'notebook/modeling.ipynb',
    'notebook/model_evaluation.ipynb',
    'notebook/model_optimization.ipynb',
]

models = [
    'model/train.py',
    'model/predict.py'
]

def notebook_command(notebook_path):
    """Runs a Jupyter notebook using nbconvert."""
    command = [
        'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
        '--inplace', notebook_path
    ]
    subprocess.run(command, check=True)

def deeplearning_command(model):
    comand = [
        'python', model
    ]
    subprocess.run(comand, check=True)
    
def run_notebook(notebooks):
    for notebook in notebooks:
        print(f"Running {notebook}...")
        notebook_command(notebook)
        print(f"{notebook} finished.\n")
        
def run_deeplearning(models):
    for model in models:
        print(f"Running {model}...")
        deeplearning_command(model)
        print(f"{model} finished.\n")

if __name__ == "__main__":
    run_notebook(notebooks)
    run_deeplearning(models)