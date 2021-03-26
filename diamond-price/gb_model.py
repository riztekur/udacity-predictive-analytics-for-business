import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv('./data/train.csv')

df = df.iloc[:,1:]
df.drop(columns=['cut','clarity'], inplace=True)
df = pd.get_dummies(df)

X = df.drop(columns='price')
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

pickle.dump(model, open('./models/gb_model.sav', 'wb'))