import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.models import Sequential


def load_data(filename: str, date_field: str, stock_name: str, ascending_time: bool) -> DataFrame:
    full_stock_data = pd.read_csv(filename)

    # Extract the closing price of the stock of Amazon
    target_stock_data = full_stock_data[[date_field, stock_name]].copy()
    target_stock_data[date_field] = pd.to_datetime(target_stock_data[date_field])

    if ascending_time:
        target_stock_data = target_stock_data.iloc[::-1]
        full_stock_data = full_stock_data.iloc[::-1]
    target_stock_data.set_index(date_field, inplace=True)

    return target_stock_data, full_stock_data


def normalization(target_stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(target_stock_data.values), scaler


def split_data(scaled_data: DataFrame, rate: float):
    # Split the data into train (80%) and test (20%) datasets
    train_size = int(len(scaled_data) * rate)
    return train_size, scaled_data[:train_size], scaled_data[train_size - var_look_back:]


def split_xy(input_data):
    X_data, y_data = [], []
    for i in range(var_look_back, len(input_data)):
        X_data.append(input_data[i - var_look_back:i, 0])
        y_data.append(input_data[i, 0])
    X_data, y_data = np.array(X_data), np.array(y_data)
    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))

    return X_data, y_data


def setup_model(shape, units, dropout):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(shape, 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def train_test_model(model, scaler, X_train, y_train, X_test, y_test, epochs):
    model.fit(X_train, y_train, epochs=epochs, batch_size=42)
    # # Step 5: Test the Model
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Undo scaling

    # # Step 6: Evaluation, calculate RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((predictions - scaler.inverse_transform(y_test.reshape(-1, 1))) ** 2))
    print(f'Root Mean Squared Error: {rmse}')

    # # : Save prediction
    df = pd.DataFrame(predictions, columns=["Price"])

    return df


def date_to_plotly(stock_chart_data, predictions_chart_data):
    set1 = {'x': stock_chart_data['Date'],
            'open': stock_chart_data['Open'], 'close': stock_chart_data['Close/Last'],
            'high': stock_chart_data['High'], 'low': stock_chart_data['Low'],
            'type': 'candlestick', }

    last_train_size_dates = stock_chart_data['Date'].tail(len(stock_chart_data) - train_size).reset_index(drop=True)
    df2_with_dates = pd.DataFrame(last_train_size_dates, columns=['Date'])
    predictions_chart_data = predictions_chart_data.join(df2_with_dates)
    set2 = {'x': stock_chart_data['Date'], 'y': stock_chart_data['Close/Last'],
            'type': 'scatter', 'mode': 'lines', 'line': {'width': 1, 'color': 'blue'}, 'name': 'Close/Last'}
    set3 = {'x': predictions_chart_data['Date'], 'y': predictions_chart_data['Price'],
            'type': 'scatter', 'mode': 'lines', 'line': {'width': 1, 'color': 'red'}, 'name': 'Prediction'}

    fig = go.Figure(data=[set1, set2, set3])
    st.plotly_chart(fig)

    pass


# 1: load data
uploaded_file = st.file_uploader("Choose a historical file")
if uploaded_file is not None:
    stock_data, stock_data_original = load_data(uploaded_file, 'Date', 'Close/Last', True)

    tab1, tab2 = st.tabs(["New Prediction", "Upload Saved Prediction"])
    with tab1:
        var_look_back = st.slider('Prediction based on dates(Look Back):', 1, 120, 60)
        var_rate = st.slider('train/test rate:', 0.0, 1.0, 0.8)
        var_unit = st.slider('unit:', 1, 120, 50)
        var_epochs = st.slider('Epoch:', 1, 30, 5)
        var_dropout = st.slider('dropout:', 0.01, 0.9, 0.2)

        # 2: Prepare Data
        scaled_data, scaler = normalization(stock_data)
        train_size, train_data, test_data = split_data(scaled_data, var_rate)
        X_train, y_train = split_xy(train_data)
        X_test, y_test = split_xy(test_data)

        # 3: prepare model & run
        model = setup_model(X_train.shape[1], var_unit, var_dropout)
        if st.button('Run new prediction'):
            predictions = train_test_model(model, scaler, X_train, y_train, X_test, y_test, var_epochs)

            # 4: display
            date_to_plotly(stock_data_original, predictions)

            # 5: save result to CSV
            csv = predictions.to_csv(index=False).encode('utf-8')
            st.download_button("Press to Download Prediction Data", csv, "prediction.csv", "text/csv",
                               key='download-csv')

    with tab2:
        # 3: load previous prediction
        uploaded_file_prediction = st.file_uploader("Choose saved prediction file")
        if uploaded_file_prediction is not None:
            predictions = pd.read_csv(uploaded_file_prediction)

            # 4: display
            chart = date_to_plotly(stock_data_original, predictions)
