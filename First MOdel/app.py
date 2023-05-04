
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
# initialising starting and ending point
from datetime import date

today = date.today()

start = "2010-01-01"
end = today


st.title('Stock Prediction')

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
plt.xlabel('Time(Days)')
plt.ylabel('Price(As per chosen Currency)')
plt.legend()
st.pyplot(fig2)





