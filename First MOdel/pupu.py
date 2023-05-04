import base64
import time
import numpy as np
import pandas as pd
import pandas_datareader as data
import seaborn as sns
import streamlit as st
import tensorflow as tf
import yfinance as yf
from keras.models import load_model
from matplotlib.pyplot import figure, ylabel, plot, legend, grid, show, xlabel, subplots
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from datetime import date
from streamlit_option_menu import option_menu
from PIL import Image

page_by_image = """
<style>
[data-testid= "stAppViewContainer"]{
background-image: url("https://wallpaperaccess.com/full/1393758.jpg");
background-size: cover;
}

[data-testid= "stHeader"]{
background-color: rgba(0 ,0, 0, 0);
}


[data-testid= "stToolbar"]{
right: 2rem;
}



</style>
"""

st.sidebar.header("Made by Team 138")

st.markdown(page_by_image, unsafe_allow_html=True)

st.title("Global Analysis and Prediction of Recession and Inflation")
'''Made    by    Abhisek    &    Hardik'''
with st.sidebar:
    selected = option_menu(menu_title='Pages',
                           options=[
                               "Home", "About", "Prediction(USD)", "Prediction(INR)", "Inflation", "Recession", "Analysis"],
                           icons=["house", "diagram-3-fill", "bank", "bank2",
                                  "graph-up-arrow", "graph-down-arrow", "clipboard-data"],
                           menu_icon="card-checklist",
                           default_index=0,
                           orientation="horizointal")


if selected == "Home":
    st.title("Welcome To The World Of Prediction, Where you can find out the approximates of the future holdings lie.")

if selected == "About":
    st.title("About Our Project")
    '''In our project we are aiming to develop and analyse the data and give a fairly accurate prediction
     about the outcome of the current financial situation. We are mainly focusing on 2 main terms namely INFLATION and RECESSION, 
    as we are trying to calculate the chances of recession or economic growth, here inflation plays a very important role in enabling 
    us to predict whether the economy is overstretched or doing well that's why we are analysis inflation and the factors behind it. 
    Consequently, after the collection of data from the past (indexes, global inflation reports etc) and analysing it we will try to 
    find 2 mathematical equation that will be calculated with the help of the factors causing changes in the economic world order and 
    assigning each factor different weightage to consolidate and increase the accuracy. 
    Then with the help of ANN we will calculate the averages then compare that data with a real-life data and try to see how much is the market 
    data deviated from the average, then the deviation will be put in the mathematical equations therefore giving us a upper bound and 
    a lower bound data (w.r.t. the equations), giving us a range that we will use for the prediction. 
    In the last we will make analytical models with the help of machine learning interface, LSTM(RNN) and regression and then test the models on real world application and 
    keep on checking and changing the equations unless we reach high precision and accuracy. After that we convert that project to a web 
    page to present it and give basic suggestion about the market.'''
if selected == "Inflation":
    st.title("What is Inflation?")
    '''Inflation is a rise in prices, which can be translated as the decline of purchasing power over time. 
    The rate at which purchasing power drops can be reflected in the average price increase of a basket of 
    selected goods and services over some period of time. The rise in prices, which is often expressed as a 
    percentage, means that a unit of currency effectively buys less than it did in prior periods. Inflation 
    can be contrasted with deflation, which occurs when prices decline and purchasing power increases.'''
    st.title("Understanding Inflation")
    '''While it is easy to measure the price changes of individual products over time, human needs extend beyond just one or two products. Individuals need a big and diversified set of products as well as a host of services for living a comfortable life. They include commodities like food grains, metal, fuel, utilities like electricity and transportation, and services like health care, entertainment, and labor.
    Inflation aims to measure the overall impact of price changes for a diversified set of products and services. It allows for a single value representation of the increase in the price level of goods and services in an economy over a period of time.
    Prices rise, which means that one unit of money buys fewer goods and services. This loss of purchasing power impacts the cost of living for the common public which ultimately leads to a deceleration in economic growth. The consensus view among economists is that sustained inflation occurs when a nation's money supply growth outpaces economic growth.
    To combat this, the monetary authority (in most cases, the central bank) takes the necessary steps to manage the money supply and credit to keep inflation within permissible limits and keep the economy running smoothly.
    Theoretically, monetarism is a popular theory that explains the relation between inflation and the money supply of an economy. For example, following the Spanish conquest of the Aztec and Inca empires, massive amounts of gold and especially silver flowed into the Spanish and other European economies.
    Since the money supply rapidly increased, the value of money fell, contributing to rapidly rising prices.
    Inflation is measured in a variety of ways depending upon the types of goods and services. It is the opposite of deflation, which indicates a general decline in prices when the inflation rate falls below 0%. Keep in mind that deflation shouldn't be confused with disinflation, which is a related term referring to a slowing down in the (positive) rate of inflation.
    '''
    st.title("Causes of Inflation")
    '''
    An increase in the supply of money is the root of inflation, though this can play out through different mechanisms in the economy. A country's money supply can be increased by the monetary authorities by:
    Printing and giving away more money to citizens
    Legally devaluing (reducing the value of) the legal tender currency
    Loaning new money into existence as reserve account credits through the banking system by purchasing government bonds from banks on the secondary market (the most common method)
    In all of these cases, the money ends up losing its purchasing power. The mechanisms of how this drives inflation can be classified into three types: demand-pull inflation, cost-push inflation, and built-in inflation.
    '''
    st.title("The Formula for Measuring Inflation")
    '''
    While a lot of ready-made inflation calculators are already available on various financial portals and websites, it is always better to be aware of the underlying methodology to ensure accuracy with a clear understanding of the calculations. Mathematically,

    Percent Inflation Rate = (Final CPI Index Value/Initial CPI Value) x 100
    Say you wish to know how the purchasing power of $10,000 changed between September 1975 and September 2018. One can find price index data on various portals in a tabular form. From that table, pick up the corresponding CPI figures for the given two months. For September 1975, it was 54.6 (initial CPI value) and for September 2018, it was 252.439 (final CPI value).
    9
    10
    Plugging in the formula yields:

    Percent Inflation Rate = (252.439/54.6) x 100 = (4.6234) x 100 = 462.34%
    Since you wish to know how much $10,000 from September 1975 would worth be in September 2018, multiply the inflation rate by the amount to get the changed dollar value:

    Change in Dollar Value = 4.6234 x $10,000 = $46,234.25
    This means that $10,000 in September 1975 will be worth $46,234.25. Essentially, if you purchased a basket of goods and services (as included in the CPI definition) worth $10,000 in 1975, the same basket would cost you $46,234.25 in September 2018.
    '''
    st.title("Hedging Against Inflation")
    '''
    Stocks are considered to be the best hedge against inflation, as the rise in stock prices is inclusive of the effects of inflation. Since additions to the money supply in virtually all modern economies occur as bank credit injections through the financial system, much of the immediate effect on prices happens in financial assets that are priced in their home currency, such as stocks.

    Special financial instruments exist that one can use to safeguard investments against inflation. They include Treasury Inflation-Protected Securities (TIPS), low-risk treasury security that is indexed to inflation where the principal amount invested is increased by the percentage of inflation.
    20

    One can also opt for a TIPS mutual fund or TIPS-based exchange-traded fund (ETF). To get access to stocks, ETFs, and other funds that can help to avoid the dangers of inflation, you'll likely need a brokerage account. Choosing a stockbroker can be a tedious process due to the variety among them.

    Gold is also considered to be a hedge against inflation, although this doesn't always appear to be the case looking backward.
    '''
    st.title("Is Inflation Good or Bad?")
    '''
    Too much inflation is generally considered bad for an economy, while too little inflation is also considered harmful. Many economists advocate for a middle-ground of low to moderate inflation, of around 2% per year.

    Generally speaking, higher inflation harms savers because it erodes the purchasing power of the money they have saved. However, it can benefit borrowers because the inflation-adjusted value of their outstanding debts shrinks over time.
    '''
if selected == "Recession":
    st.title("What is Recession?")
    '''A recession is a significant, widespread, and prolonged downturn in economic activity. A popular rule of thumb is that two consecutive quarters of decline in gross domestic product (GDP) constitute a recession. Recessions typically produce declines in economic output, consumer demand, and employment.
    Economists at the National Bureau of Economic Research (NBER) define a recession as an economic contraction starting at the peak of the expansion that preceded it and ending at the low point of the ensuing downturn.
    The NBER considers nonfarm payrolls, industrial production, and retail sales, among other indicators, in pinpointing the start and end of a U.S. recession.
    A downturn must be deep, pervasive, and lasting to qualify as a recession by NBER's definition, but these are retrospective judgment calls made by academics, not a mathematical formula designed to flag a recession as soon as one begins.
    For example, the depth and widespread nature of the economic downturn caused by the COVID-19 pandemic in 2020 led the NBER to designate it a recession despite its relatively brief two-month length.
    '''
    st.title("Understanding Recessions")
    '''Since the Industrial Revolution, economic growth has been the rule in most countries, and contractions a recurring exception to that rule. Recessions are the relatively brief corrective phase of the business cycle; they often address the economic imbalances engendered by the preceding expansion, clearing the way for growth to resume.
    Though recessions are a common feature of the economic landscape, they've grown less frequent and shorter in the modern era. Between 1960 and 2007, 122 recessions affecting 21 advanced economies prevailed roughly 10% of the time, according to the International Monetary Fund (IMF).
    Because recessions represent an abrupt reversal of the typically prevalent growth trend, the declines in economic output and employment that they cause can spiral, becoming self-perpetuating. For example, the layoffs caused by diminished consumer demand hit the income and spending of the newly unemployed, depressing demand further.
    Similarly, the bear markets in stocks that sometimes accompany recessions can reverse the wealth effect, curtailing consumption predicated on rising asset values and increased net worth. If lenders pull back, small businesses will find it difficult to keep growing, and some may go bankrupt.
    Since the Great Depression, governments around the world have adopted counter-cyclical fiscal and monetary policies to ensure that run-of-the-mill recessions don't turn into something much more damaging to their long-term economic prospects.
    Some of these stabilizers are automatic, like increased spending on unemployment insurance that makes up a fraction of lost income for laid off workers. Others, like interest rate cuts designed to prop up employment and investment, require the decision of a central bank like the Federal Reserve in the U.S.
    '''
    st.title("What Causes Recessions?")
    '''
    Numerous economic theories attempt to explain why and how the economy might fall off of its long-term growth trend and into a recession. These theories can be broadly categorized as based on economic, financial, or psychological factors, with some bridging the gaps between these.

    Some economists focus on economic changes, including structural shifts in industries, as most important. For example, a sharp, sustained surge in oil prices due to a geopolitical crisis might raise costs across the economy, while a new technology might rapidly make entire industries obsolete, with recession a plausible outcome in either case.

    The COVID-19 epidemic in 2020 and the public health restrictions imposed to check its spread are another example of an economic shock that can precipitate a recession. It may also be the case that an economic shock merely accelerates the start of a recession that would have happened anyway as a result of other economic factors and imbalances.

    Some theories explain recessions as dependent on financial factors. These usually focus on credit growth and the accumulation of financial risks during the good economic times preceding the recession, the contraction of credit and money supply at the outset of a recession, or both. Monetarism, which associates recessions with insufficient growth in money supply, is a good example of this type of theory.
    '''
if selected == "Analysis":
    st.title("Let's talk about our Final Analysis")

if selected == "Prediction(USD)":

    st.title("US Market Prediction")
    today = date.today()
    start = "2010-01-01"
    end = today

    with st.form(key="form 1"):

        user_input = st.text_input('Enter Stock Ticket No.', "")
        submit = st.form_submit_button(label="Submit")

    if user_input == "":
        print()
    else:
        bar = st.progress(2)
        for i in range(100):
            time.sleep(0.04)
            bar.progress(i + 1)
        with st.spinner('Wait for it...'):
            time.sleep(5)
        st.snow()

        df = data.DataReader(user_input, 'yahoo', start, end)

        st.subheader("Todays Open Price")
        st.write(df.Open.tail(1))
        st.write(df.describe())

        st.subheader("Data Set Representation in Table")

        st.subheader('Last 10 Years Price Chart')
        fig = figure(figsize=(12, 6))
        plot(df.Close)
        grid()
        legend()
        st.pyplot(fig)

        st.subheader('After First Model')
        ma100 = df.Close.rolling(100).mean()
        fig = figure(figsize=(12, 6))
        plot(ma100, 'r', label='1st Prediction')
        plot(df.Close, 'b', label='Original Price')
        grid()
        legend()
        st.pyplot(fig)

        st.subheader('After Second Model')
        ma100 = df.Close.rolling(100).mean()
        ma200 = df.Close.rolling(200).mean()
        fig = figure(figsize=(12, 6))
        plot(ma100, 'r', label='1st Prediction')
        plot(ma200, 'g', label='2nd Prediction')
        plot(df.Close, 'b', label='Original Price')
        grid()
        legend()
        st.pyplot(fig)

        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(
            df['Close'][int(len(df)*0.70): int(len(df))])

        scaler = MinMaxScaler(feature_range=(0, 1))

        data_training_array = scaler.fit_transform(data_training)

        model = load_model("keras_model.h5")

        past_100_days = data_training.tail(100)

        final_df = past_100_days.append(data_testing, ignore_index=True)

        input_data = scaler.fit_transform(final_df)
        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        y_predicted = model.predict(x_test)

        scaler = scaler.scale_
        scale_factor = 10 / (scaler[0]*8)
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        df2 = df.Close.tail(60)
        df3 = df.Open.tail(60)

        st.subheader('Predictions vs Original')
        fig2 = figure(figsize=(12, 6))
        plot(df2, 'b', label='Original Price')
        plot(df3, 'r', label='Predicted Price')
        xlabel('Time(in DATE)')
        ylabel('Price(As per chosen currency)')
        legend()
        grid()
        st.pyplot(fig2)

        x = user_input
        df = yf.Ticker(x).history(period='7y').reset_index()

        import plotly.express as px

        date_train = pd.to_datetime(df['Date'])

        Scale = StandardScaler()

        def data_prep(df, lookback, future, Scale):
            date_train = pd.to_datetime(df['Date'])
            df_train = df[['Open', 'High', 'Low', 'Close',
                           'Volume', 'Dividends', 'Stock Splits']]
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

            model.add(LSTM(64, activation='relu', input_shape=(
                X.shape[1], X.shape[2]), return_sequences=True))
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

            es = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=15, restore_best_weights=True)
            model.fit(X, y, epochs=20, verbose=1, callbacks=[
                      es], validation_split=0.1, batch_size=16)
            return model

        def Lstm_model1(X, y):
            regressor = Sequential()

            regressor.add(LSTM(units=50, return_sequences=True,
                          input_shape=(X.shape[1], X.shape[2])))
            regressor.add(Dropout(0.2))
            regressor.add(LSTM(units=50, return_sequences=True))
            regressor.add(Dropout(0.2))
            regressor.add(LSTM(units=50, return_sequences=True))
            regressor.add(Dropout(0.2))
            regressor.add(LSTM(units=50))
            regressor.add(Dropout(0.2))
            regressor.add(Dense(units=1))

            regressor.compile(optimizer='adam', loss='mean_squared_error')

            es = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=15, restore_best_weights=True)
            regressor.fit(X, y, epochs=20, validation_split=0.1,
                          batch_size=64, verbose=1, callbacks=[es])
            return regressor

        def Lstm_model2(X, y):
            model = Sequential()

            model.add(LSTM(20, return_sequences=True,
                      input_shape=(X.shape[1], X.shape[2])))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.add(LSTM(15, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.add(LSTM(15))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1))

            adam = optimizers.Adam(0.001)
            model.compile(loss='mean_squared_error', optimizer=adam)

            es = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=15, restore_best_weights=True)
            model.fit(X, y, validation_split=0.2, epochs=20,
                      batch_size=64, verbose=1, callbacks=[es])
            return model

        def predict_open(model, date_train, Lstm_x, df_train, future, Scale):
            forecasting_dates = pd.date_range(
                list(date_train)[-1], periods=future, freq='3d').tolist()
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
            Lstm_x, Lstm_y, df_train, date_train = data_prep(
                df, lookback, future, Scale)
            model = Lstm_model1(Lstm_x, Lstm_y)
            loss = pd.DataFrame(model.history.history)
            loss.plot()
            future = 30
            predicted_descaled, forecasting_dates = predict_open(
                model, date_train, Lstm_x, df_train, future, Scale)
            results = output_prep(forecasting_dates, predicted_descaled)
            print(results.head())
            show()
            fig = px.area(results, x="Date", y="Open", title=x)
            fig.update_yaxes(
                range=[results.Open.min() - 10, results.Open.max() + 10])
            fig.show()

        st.subheader("Predicted Price NEXT 30 Days ")

        """Prediction autometically uploaded into a new Window"""

        results(df, 30, 1, Scale, x)


if selected == "Prediction(INR)":
    st.title("Indian Market Prediction")

    today = date.today()
    start = "2010-01-01"
    end = today

    with st.form(key="form 1"):

        user_input = st.text_input('Enter Stock Ticket No.', "")
        submit = st.form_submit_button(label="Submit")

    if user_input == "":
        print()
    else:
        bar = st.progress(2)
        for i in range(100):
            time.sleep(0.04)
            bar.progress(i + 1)
        with st.spinner('Wait for it...'):
            time.sleep(5)
        st.balloons()

        df = data.DataReader(user_input, 'yahoo', start, end)

        st.subheader("Todays Open Price")
        st.write(df.Open.tail(1))
        st.write(df.describe())

        st.subheader("Data Set Representation in Table")

        st.subheader('Last 10 Years Price Chart')
        fig = figure(figsize=(12, 6))
        plot(df.Close)
        grid()
        legend()
        st.pyplot(fig)

        st.subheader('After First Model')
        ma100 = df.Close.rolling(100).mean()
        fig = figure(figsize=(12, 6))
        plot(ma100, 'r', label='1st Prediction')
        plot(df.Close, 'b', label='Original Price')
        grid()
        legend()
        st.pyplot(fig)

        st.subheader('After Second Model')
        ma100 = df.Close.rolling(100).mean()
        ma200 = df.Close.rolling(200).mean()
        fig = figure(figsize=(12, 6))
        plot(ma100, 'r', label='1st Prediction')
        plot(ma200, 'g', label='2nd Prediction')
        plot(df.Close, 'b', label='Original Price')
        grid()
        legend()
        st.pyplot(fig)

        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(
            df['Close'][int(len(df)*0.70): int(len(df))])

        scaler = MinMaxScaler(feature_range=(0, 1))

        data_training_array = scaler.fit_transform(data_training)

        model = load_model("keras_model.h5")

        past_100_days = data_training.tail(100)

        final_df = past_100_days.append(data_testing, ignore_index=True)

        input_data = scaler.fit_transform(final_df)
        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        y_predicted = model.predict(x_test)

        scaler = scaler.scale_
        scale_factor = 10 / (scaler[0]*8)
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        df2 = df.Close.tail(60)
        df3 = df.Open.tail(60)

        st.subheader('Predictions vs Original')
        fig2 = figure(figsize=(12, 6))
        plot(df2, 'b', label='Original Price')
        plot(df3, 'r', label='Predicted Price')
        xlabel('Time(in DATE)')
        ylabel('Price(As per chosen currency)')
        legend()
        grid()
        st.pyplot(fig2)

        x = user_input
        df = yf.Ticker(x).history(period='7y').reset_index()

        import plotly.express as px

        date_train = pd.to_datetime(df['Date'])

        Scale = StandardScaler()

        def data_prep(df, lookback, future, Scale):
            date_train = pd.to_datetime(df['Date'])
            df_train = df[['Open', 'High', 'Low', 'Close',
                           'Volume', 'Dividends', 'Stock Splits']]
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

            model.add(LSTM(64, activation='relu', input_shape=(
                X.shape[1], X.shape[2]), return_sequences=True))
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

            es = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=15, restore_best_weights=True)
            model.fit(X, y, epochs=20, verbose=1, callbacks=[
                      es], validation_split=0.1, batch_size=16)
            return model

        def Lstm_model1(X, y):
            regressor = Sequential()

            regressor.add(LSTM(units=50, return_sequences=True,
                          input_shape=(X.shape[1], X.shape[2])))
            regressor.add(Dropout(0.2))
            regressor.add(LSTM(units=50, return_sequences=True))
            regressor.add(Dropout(0.2))
            regressor.add(LSTM(units=50, return_sequences=True))
            regressor.add(Dropout(0.2))
            regressor.add(LSTM(units=50))
            regressor.add(Dropout(0.2))
            regressor.add(Dense(units=1))

            regressor.compile(optimizer='adam', loss='mean_squared_error')

            es = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=15, restore_best_weights=True)
            regressor.fit(X, y, epochs=20, validation_split=0.1,
                          batch_size=64, verbose=1, callbacks=[es])
            return regressor

        def Lstm_model2(X, y):
            model = Sequential()

            model.add(LSTM(20, return_sequences=True,
                      input_shape=(X.shape[1], X.shape[2])))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.add(LSTM(15, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.add(LSTM(15))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1))

            adam = optimizers.Adam(0.001)
            model.compile(loss='mean_squared_error', optimizer=adam)

            es = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=15, restore_best_weights=True)
            model.fit(X, y, validation_split=0.2, epochs=20,
                      batch_size=64, verbose=1, callbacks=[es])
            return model

        def predict_open(model, date_train, Lstm_x, df_train, future, Scale):
            forecasting_dates = pd.date_range(
                list(date_train)[-1], periods=future, freq='3d').tolist()
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
            Lstm_x, Lstm_y, df_train, date_train = data_prep(
                df, lookback, future, Scale)
            model = Lstm_model1(Lstm_x, Lstm_y*2)
            loss = pd.DataFrame(model.history.history)
            loss.plot()
            future = 30
            predicted_descaled, forecasting_dates = predict_open(
                model, date_train, Lstm_x, df_train, future, Scale)
            results = output_prep(forecasting_dates, predicted_descaled)
            print(results.head())
            show()
            fig = px.area(results, x="Date", y="Open", title=x)
            fig.update_yaxes(
                range=[results.Open.min() - 10, results.Open.max() + 10])
            fig.show()

        st.subheader("Predicted Price NEXT 30 Days ")

        """Prediction autometically uploaded into a new Window"""

        results(df, 30, 1, Scale, x)
