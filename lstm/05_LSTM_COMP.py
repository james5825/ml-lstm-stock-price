# Step 1: Import Necessary Libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.models import Sequential

# Use 60 days' data to predict the 61th day's data
look_back = 60


# Step 2: Load and Preprocess the Dataset
def load_data(filename: str, date_field: str, stock_name: str, reverse_time: bool) -> DataFrame:
    data = pd.read_csv(filename)

    # Extract the closing price of the stock of Amazon
    target_stock_data = data[[date_field, stock_name]].copy()
    target_stock_data[date_field] = pd.to_datetime(target_stock_data[date_field])

    if reverse_time:
        target_stock_data = target_stock_data.iloc[::-1]
    target_stock_data.set_index(date_field, inplace=True)

    return target_stock_data


def normalization(target_stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(target_stock_data.values), scaler


def split_data(scaled_data: DataFrame, rate: float):
    # Split the data into train (80%) and test (20%) datasets
    train_size = int(len(scaled_data) * rate)
    return train_size, scaled_data[:train_size], scaled_data[train_size - look_back:]


def split_xy(input_data):
    X_data, y_data = [], []
    for i in range(look_back, len(input_data)):
        X_data.append(input_data[i - look_back:i, 0])
        y_data.append(input_data[i, 0])
    X_data, y_data = np.array(X_data), np.array(y_data)
    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))

    return X_data, y_data


def setup_model(shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(shape, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def train_test_model(model, scaler, X_train, y_train, X_test, y_test, save_prediction):
    model.fit(X_train, y_train, epochs=1, batch_size=42)
    # # Step 5: Test the Model
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Undo scaling

    # # Step 6: Evaluation, calculate RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((predictions - scaler.inverse_transform(y_test.reshape(-1, 1))) ** 2))
    print(f'Root Mean Squared Error: {rmse}')

    # # : Save prediction
    df = pd.DataFrame(predictions, columns=["prediction"])
    if save_prediction:
        df.to_csv('predictions.csv', index=False)

    return df


# # Load Data
stock_data = load_data('HistoricalData_1697501124425.csv', 'Date', 'Close/Last', True)
scaled_data, scaler = normalization(stock_data)

# # Prepare Data
train_size, train_data, test_data = split_data(scaled_data, 0.8)
X_train, y_train = split_xy(train_data)
X_test, y_test = split_xy(test_data)

model = setup_model(X_train.shape[1])

# predictions = train_test_model(model, scaler, X_train, y_train, X_test, y_test, True)
predictions = pd.read_csv('predictions.csv')

# # Step 7: Visualization, Plot the Results
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('NASDAQ Composite Index')
plt.plot(stock_data[:train_size]['Close/Last'], label='Train Data')
plt.plot(stock_data[train_size:].index, stock_data[train_size:]['Close/Last'], label='True Test Price')
plt.plot(stock_data[train_size:].index, predictions['prediction'], label='Prediction Price')
plt.legend(loc='lower right')
plt.show()

print('Done')
