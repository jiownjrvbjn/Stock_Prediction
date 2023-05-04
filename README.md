# Stock_Prediction
The goal of this project was to create a website that predicts the stock growth rate on the Indian and USA stock markets. First of all, I used python to create a stock market prediction software and used yahoo finance to collect data on one of the stable stocks in the USA “AAPL”. I then hosted my python model prediction on Streamlit.

Modules Used: 
from sklearn.preprocessing import MinMaxScaler 
numpy 
pandas 
matplotlib.pyplot 
pandas_datareader 
from keras.models import load_model 
streamlit

Note: 
I have created a data training model using python jupyter, saved in lstm model.
Thereby the trained model is saved in keras_model.h5 .
To run the model you need to run streamlit with pupu.py file. Which is used to showcase the prediction on website using streamlit command " streamlit run pupu.py " on windows terminal with all the files saved in one folder and that olders address is neded as the directory to run the command.
