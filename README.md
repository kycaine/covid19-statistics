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

run "app.py" it will automatically run all notebooks and deep learning model

Success if?
The process is successful when the output folder is created.

# I/O

1. Data Cleaning

   - Input : Raw data (dir. data)
   - Output : Cleaned data (dir. output/data_cleaning)

2. Exploratory Data Analysis (EDA)

   - Input : Output from data cleaning (dir. output/data_cleaning)
   - Output : Visualization & data statistics (dir. output/eda)

3. Feature Engineering

   - Input : Output from data cleaning (dir. output/data_cleaning)
   - Output : Transformed features to enhance the model (dir. output/feature_engineering)

4. Modeling

   - Input : Output from feature engineering (dir. output/feature_engineering)
   - Output : The predictions (dir. output/modeling)

5. Model Evaluation

   - Input : The predictions from modeling (dir. output/modeling)
   - Output : Assessment of model performance (dir. output/model_evaluation)

6. Model Optimization
   - Input:
     - Output from model evaluation (dir. output/model_evaluation)
     - The predictions from modeling (dir. output/modeling)
   - Output : Optimized model performance (dir. output/model_optimization)

# End-to-End ML Process

1. Data Cleaning → Removes noise, handles missing values, and corrects errors in the dataset.

2. Exploratory Data Analysis (EDA) → Visualizes and analyzes data patterns to gain insights.

3. Feature Engineering → Creates new features or transforms existing ones to improve model performance.

4. Modeling → Trains a machine learning model using the processed data.

5. Model Evaluation → Measures model performance using metrics such as MAE, RMSE, R², or accuracy.

6. Model Optimization → Enhances model performance through hyperparameter tuning or other optimization techniques.
