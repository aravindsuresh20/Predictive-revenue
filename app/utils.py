# Importing pandas for data manipulation
import pandas as pd
# Importing MinMaxScaler for data scaling
from sklearn.preprocessing import MinMaxScaler

# Defining a function to load data from a CSV file
def load_data(file_path):
    # Reading the CSV file and parsing dates
    df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
    # Selecting only the 'Date' and 'Revenue' columns
    df = df[['Date', 'Revenue']].copy()
    # Sorting the dataframe by date
    df.sort_values('Date', inplace=True)
    # Setting the 'Date' column as the index
    df.set_index('Date', inplace=True)
    return df

# Defining a function to scale the data
def scale_data(data):
    # Initializing the MinMaxScaler
    scaler = MinMaxScaler()
    # Scaling the data
    scaled = scaler.fit_transform(data)
    return scaled, scaler
