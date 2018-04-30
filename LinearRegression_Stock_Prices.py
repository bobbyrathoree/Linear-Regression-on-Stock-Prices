import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing # for scaling data
from sklearn import cross_validation # for training and testing samples, splitting, shuffling
from sklearn import svm # can use for regression
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

df = quandl.get('WIKI/GOOGL') # imported wiki stock prices data into our panda dataframe, choose any dataset you want from quandl.com

# See all the features using print(df.head())
# We only need a few features that will actually help in our regression model

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['High_Low_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 # High Low %
df['Percent_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 # % change in stock prices

df = df[['Adj. Close', 'High_Low_PCT', 'Percent_Change', 'Adj. Volume']] # We only care about these features.

forecast_column = 'Adj. Close'

# if any NaN values, fill them with -112233, like an outlier
df.fillna(-112233, inplace=True)
how_many_days = int(math.ceil(0.01 * len(df))) # we want these number of days to predict for, next.
df['label'] = df[forecast_column].shift(-how_many_days) # shift a few rows above

X = np.array(df.drop(['label'], axis=1)) # X are features, so take everything except label
y = np.array(df['label'])
X = preprocessing.scale(X) # subtracts the mean and divides by the standard deviation of your dataset along a given axis

# Seperate data to predict upon
X_lately = X[-how_many_days:] # for prediction
X = X[:-how_many_days]
df.dropna(inplace=True)

# Train and test set split: 80-20
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
stock_prediction = clf.predict(X_lately) # to predict the next how_many_days days

print('Next {0} days prediction: {1}'.format(how_many_days, stock_prediction))

# Plotting

df['Stock Forecast'] = np.nan

last_date = df.iloc[-1].name # name of last date in dataset
last_date_unix = last_date.timestamp()
one_day_seconds = 86400
next_date_unix = last_date_unix + one_day_seconds

for i in stock_prediction:
    next_date = datetime.datetime.fromtimestamp(next_date_unix)
    next_date_unix += one_day_seconds
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    
df['Adj. Close'].plot()
df['Stock Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
