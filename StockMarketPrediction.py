import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


df = pd.read_csv('DJIA_table2.csv')
dfplt = df[['Close']]
forecast_out = int(30)
df = df[['Close']]
df['Prediction'] = df[['Close']].shift(-forecast_out)

X = np.array(df.drop(['Prediction'],1))
X = preprocessing.scale(X)

X_forecast = X
X = X[:-forecast_out]

y = np.array(df['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

confidence = clf.score(X_test, y_test)
print("confidence: {0:2.2%}".format(confidence))
forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)

b = np.array(dfplt)

fig, ax = plt.subplots()
ax.plot(b, color="green", label='Real Market')
ax.plot(forecast_prediction, color="blue", label='My Prediction')
legend = ax.legend(shadow=False, fontsize='x-large')
plt.show()
