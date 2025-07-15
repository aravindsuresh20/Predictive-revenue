# Importing the NumPy library for numerical operations
import numpy as np  
# Importing the XGBoost library for machine learning models
import xgboost as xgb  
# Importing custom utility functions for data loading and scaling
from utils import load_data, scale_data  

# Defining a function to forecast the next n months
def forecast_next(model_path, data_path, n_months=6):  
    # Initializing an XGBoost model
    model = xgb.Booster()  
    # Loading the pre-trained model from the specified path
    model.load_model(model_path)  
    # Loading the data from the specified path
    df = load_data(data_path)  
    # Scaling the 'Revenue' column data
    data_scaled, scaler = scale_data(df[['Revenue']].values)  
    # Setting the sequence length for the model input
    SEQ_LEN = 12  
    # Extracting the last sequence of scaled data
    last_sequence = data_scaled[-SEQ_LEN:]  
    # Initializing an empty list to store predictions
    predictions = []  
    # Setting the current sequence to the last sequence
    current_sequence = last_sequence  
    # Looping through the number of months to forecast
    for _ in range(n_months):  
        # Creating a DMatrix for the current sequence
        dmatrix = xgb.DMatrix(current_sequence.reshape(1, SEQ_LEN))  
        # Predicting the next value using the model
        pred = model.predict(dmatrix)[0]  
        # Appending the prediction to the list
        predictions.append(pred)  
        # Updating the current sequence with the new prediction
        current_sequence = np.append(current_sequence[1:], [pred], axis=0)  
    # Inversely transforming the predictions to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))  
    # Returning the final predictions
    return predictions  
