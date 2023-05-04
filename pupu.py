from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

# initialising starting and ending point
start = "2010-01-01"
end = "2022-7-30"


st.title('Global Financial Analysis')

user_input = st.text_input('Enter Stock Ticker', "AAPL")
# data frame = data to which we are refering
df = data.DataReader(user_input, 'yahoo', start, end)

# Describing Data
st.subheader("Data from 2010-2022")
st.write(df.describe())

# Visualisations
st.subheader('Clossing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Clossing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Clossing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


# splitting data into training and testing in 70 to 30 percent ratio

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# scaling down the data to 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))

# scalling the training data set
data_training_array = scaler.fit_transform(data_training)


# Load my model
model = load_model("keras_model.h5")

# Testing Part
# to predict the values of testing data we need previous 100 days
past_100_days = data_training.tail(100)

# appending past 100 dataset to the data testing
final_df = past_100_days.append(data_testing, ignore_index=True)
# now scalling down the data
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

# converting them to numpy array
x_test, y_test = np.array(x_test), np.array(y_test)

# Making Predictions

y_predicted = model.predict(x_test)

# scalling the valuesback again
# finding the factor
scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Final Graph
# ploting the predicted vs original values
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time(in DATE)')
plt.ylabel('Price(As per chosen currency)')
plt.legend()
st.pyplot(fig2)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import yfinance as yf

# df= yf.Ticker("^NSEBANK").history(period='5y').reset_index()
x = "AAPL"
df = yf.Ticker(x).history(period='5y').reset_index()

import plotly.express as px

fig = px.line(df, x='Date', y="Open")
fig.show()

print(df.Date.max())
print(df.Date.min())

fig, axes = plt.subplots(2, 2, sharex=True, figsize=(20, 32))
plt.grid(True)
sns.lineplot(ax=axes[0, 0], data=df, x='Date', y='Open')
axes[0, 0].set_title(x)
sns.lineplot(ax=axes[0, 1], data=df, x='Date', y='Open')
axes[0, 1].set_title(x)
sns.lineplot(ax=axes[1, 0], data=df, x='Date', y='Open')
axes[1, 0].set_title(x)
sns.lineplot(ax=axes[1, 1], data=df, x='Date', y='Open')
axes[1, 1].set_title(x)

print(df.shape)
date_train = pd.to_datetime(df['Date'])
date_train

Scale = StandardScaler()


def data_prep(df, lookback, future, Scale):
    date_train = pd.to_datetime(df['Date'])
    df_train = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
    df_train = df_train.astype(float)

    df_train_scaled = Scale.fit_transform(df_train)

    X, y = [], []
    for i in range(lookback, len(df_train_scaled) - future + 1):
        X.append(df_train_scaled[i - lookback:i, 0:df_train.shape[1]])
        y.append(df_train_scaled[i + future - 1:i + future, 0])

    return np.array(X), np.array(y), df_train, date_train


Lstm_x, Lstm_y, df_train, date_train = data_prep(df, 30, 1, Scale)


def Lstm_fallback(X, y):
    model = Sequential()

    model.add(LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y.shape[1], activation='relu'))

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(
        loss='mse',
        optimizer=opt,
    )

    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    model.fit(X, y, epochs=100, verbose=1, callbacks=[es], validation_split=0.1, batch_size=16)
    return model


def Lstm_model1(X, y):
    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    regressor.fit(X, y, epochs=50, validation_split=0.1, batch_size=64, verbose=1, callbacks=[es])
    return regressor


def Lstm_model2(X, y):
    model = Sequential()

    model.add(LSTM(20, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    # model.add(LSTM(15,return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    model.add(LSTM(15))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    adam = optimizers.Adam(0.001)
    model.compile(loss='mean_squared_error', optimizer=adam)

    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    model.fit(X, y, validation_split=0.2, epochs=30, batch_size=64, verbose=1, callbacks=[es])
    return model


def predict_open(model, date_train, Lstm_x, df_train, future, Scale):
    forecasting_dates = pd.date_range(list(date_train)[-1], periods=future, freq='3d').tolist()
    predicted = model.predict(Lstm_x[-future:])
    predicted1 = np.repeat(predicted, df_train.shape[1], axis=-1)
    predicted_descaled = Scale.inverse_transform(predicted1)[:, 0]
    return predicted_descaled, forecasting_dates


def output_prep(forecasting_dates, predicted_descaled):
    dates = []
    for i in forecasting_dates:
        dates.append(i.date())
    df_final = pd.DataFrame(columns=['Date', 'Open'])
    df_final['Date'] = pd.to_datetime(dates)
    df_final['Open'] = predicted_descaled
    return df_final


def results(df, lookback, future, Scale, x):
    Lstm_x, Lstm_y, df_train, date_train = data_prep(df, lookback, future, Scale)
    model = Lstm_model1(Lstm_x, Lstm_y * 1.2)
    loss = pd.DataFrame(model.history.history)
    loss.plot()
    future = 30
    predicted_descaled, forecasting_dates = predict_open(model, date_train, Lstm_x, df_train, future, Scale)
    results = output_prep(forecasting_dates, predicted_descaled)
    print(results.head())
    plt.show()
    fig = px.area(results, x="Date", y="Open", title=x)
    fig.update_yaxes(range=[results.Open.min() - 10, results.Open.max() + 10])
    fig.show()


results(df, 30, 1, Scale, x)

st.subheader("Predicted Price NEXT 30 Days ")
fig3 = plt.figure(figsize=(12,6))
plt.plot(results(df, 30, 1, Scale, x))
st.pyplot(fig2)
