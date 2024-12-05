# HOW TO RUN
Navigate to the covid19 project directory and run the following commands:
1. pip install -r requirements.txt
2. jupyter notebook

Run the .ipynb files in the notebook folder in the following order:
1. data cleaning
2. eda
3. feature engineering
4. modeling
5. model evaluation
6. model optimization

or 

run "run_notebooks.py" it will automatically run all notebooks

Success if?
The process is successful when the output folder is created.


# I/O
1. data cleaning
    - input : raw data (dir. data)
    - output : cleaned data (dir. output/data_cleaning)  
2. eda
    - input : output from data data cleaning (dir. output/data_cleaning) 
    - output : visualization data statistics (dir. output/eda)
3. feature engineering
    - input : output from data data cleaning (dir. output/data_cleaning) 
    - output : transformed feature for enhance model (dir. output/feature_enginnering) 
4. modeling
    - input : output from feature enginnering (dir. output/feature_enginnering) 
    - output : the predictions (dir. output/modeling) 
5. model evaluation
    - input : the predictions from modeling (dir. output/modeling) 
    - output : assesment performance of machine learning (dir. output/model_evaluationn) 
6. model optimization
    - input: 
        - output from model evaluation (dir. output/model_evaluation)
        - the predictions from modeling (dir. output/modeling) 
    - output : optimization performance from model evaluation (dir. output/model_optimization) 


