# Importing NumPy for numerical operations
import numpy as np
# Importing XGBoost for machine learning models
import xgboost as xgb
# Importing custom utility functions for data loading and scaling
from utils import load_data, scale_data
# Importing os for operating system interactions
import os

# Defining a function to create sequences from the data
def create_sequences(data, sequence_length):
    x, y = [], []
    # Looping through the data to create sequences
    for i in range(len(data) - sequence_length):
        x.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(x), np.array(y)

# Defining a function to train the model
def train_model(data_path, save_path='models/xgb_model.json'):
    # Loading the data from the specified path
    df = load_data(data_path)
    # Scaling the 'Revenue' column data
    data_scaled, scaler = scale_data(df[['Revenue']].values)
    # Setting the sequence length for the model input
    SEQ_LEN = 12
    # Creating sequences from the scaled data
    x, y = create_sequences(data_scaled, SEQ_LEN)
    # Creating DMatrix for the training data
    dmatrix = xgb.DMatrix(x, label=y)
    # Setting parameters for the XGBoost model
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'eta': 0.1
    }
    # Training the XGBoost model
    model = xgb.train(params, dmatrix, num_boost_round=50)
    # Creating the directory to save the model if it doesn't exist
    os.makedirs('models', exist_ok=True)
    # Saving the trained model to the specified path
    model.save_model(save_path)
    return model, scaler, SEQ_LEN
