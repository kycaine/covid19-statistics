import subprocess
import os

notebooks = [
    'notebook/data_cleaning.ipynb',
    'notebook/eda.ipynb',
    'notebook/feature_engineering.ipynb',
    'notebook/modeling.ipynb',
    'notebook/model_evaluation.ipynb',
    'notebook/model_optimization.ipynb',
]

def run_notebook(notebook_path):
    """Runs a Jupyter notebook using nbconvert."""
    command = [
        'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
        '--inplace', notebook_path
    ]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    for notebook in notebooks:
        print(f"Running {notebook}...")
        run_notebook(notebook)
        print(f"{notebook} finished.\n")
