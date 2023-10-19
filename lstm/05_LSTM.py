# Step 1: Import Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout

# # from tensorflow import keras
#
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout

# Step 2: Load and Preprocess the Dataset
# Load the dataset
filename = 'portfolio_data.csv'
data = pd.read_csv(filename)

# Extract the closing price of the stock of Amazon
stock_data = data[['Date', 'AMZN']].copy()
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.set_index('Date', inplace=True)

# Normalize the data
# We need to use the scaler in the future, to retrieve the true price
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data.values)

# # Choose the number of past days to use for the prediction
look_back = 60
# Use 60 days' data to predict the 61th day's data

# Split the data into train (80%) and test (20%) datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - look_back:]

# Create X_train, y_train, X_test, y_test
X_train, y_train = [], []
for i in range(look_back, len(train_data)):
    X_train.append(train_data[i - look_back:i, 0])
    y_train.append(train_data[i, 0])
# Use data [0, 1, 2, ,59] to predict data [60], predict the next value
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the input to be [samples, time steps, features] as the lstm input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# time steps = X_train.shape[1] = 60
# features = 1

X_test, y_test = [], []
for i in range(look_back, len(test_data)):
    X_test.append(test_data[i - look_back:i, 0])
    y_test.append(test_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
# Reshape the test data to be [samples, time steps, features]
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



# Step 3: Build the LSTM Model
model = Sequential()  # 3层

model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
# tf.keras.layers.LSTM() 的默认输出大小为 [batch_size, units]，就是只使用最后一个 time step 的输出
# 想要得到每个 time step 的输出，指定 return_sequences=True, 每个time step与下一层LSTM相连，形成多层LSTM
# 输出size=[32,60,50]=[batch_size, time step, units]

# Dropout的提出，源于2012年Hinton的一篇论文——《Improving neural networks by preventing co-adaptation of feature detectors》。
# 论文中描述了当数据集较小时而神经网络模型较大较复杂时，训练时容易产生过拟合
# Dropout：层之间加入dropout层, 简单粗暴，任意丢弃神经网络层中的输入，该层可以是数据样本中的输入变量或来自先前层的激活。
# 它能够模拟具有大量不同网络结构的神经网络，防治over-fitting, 并且反过来使网络中的节点更具有鲁棒性。
# 当dropout rate=0.2时，实际上，保留概率为0.8

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Dropout小故事：
# 2012年，大名鼎鼎的AlexNet网络的论文——《ImageNet Classification with Deep Convolutional Neural Networks》中，
# 应用了Dropout,并且证明了其在提高模型精度和降低过拟合方面效果出色。由于AlexNet有效的网络结构+Dropout的应用，
# 此模型在12年的ImageNet分类赛上以大幅优势领先第二名，从而使得深度卷积神经网络CNN在图像分类上的应用掀起一波热潮～
# 三个人的初创公司, DNN Research Inc., 被竞拍卖给 Google

model.add(LSTM(units=50))
model.add(Dropout(0.2))
# 输出size=[32,60,50]=[batch_size, time step, units]

model.add(Dense(units=1))
# Dense全连接层,即一层的ANN神经网络, 输出size=[32, 1]=[batch_size, units]

model.compile(optimizer='adam', loss='mean_squared_error')



# Step 4: Train the Model
model.fit(X_train, y_train, epochs=1, batch_size=32)



# Step 5: Test the Model
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Undo scaling



# Step 6: Evaluation, calculate RMSE (Root Mean Square Error)
rmse = np.sqrt(np.mean((predictions - scaler.inverse_transform(y_test.reshape(-1, 1))) ** 2))
print(f'Root Mean Squared Error: {rmse}')



# Step 7: Visualization, Plot the Results
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('AMZN Price USD ($)')
plt.plot(stock_data[:train_size]['AMZN'], label='Train Data')
plt.plot(stock_data[train_size:].index, stock_data[train_size:]['AMZN'], label='True Test Price')
plt.plot(stock_data[train_size:].index, predictions, label='Prediction Price')
plt.legend(loc='lower right')
plt.show()

print('Stop')
