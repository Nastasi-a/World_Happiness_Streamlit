import streamlit as st
import pandas as pd

df=pd.read_csv("merged_happiness_dataframe.csv")

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

feats = df.drop(['Ladder score'], axis=1)
target = df[['Ladder score']]

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state=42) #Splitting the data on the train and test sets

cat = ['Regional indicator', 'Country name']
oneh = OneHotEncoder(drop = 'first', sparse_output = False, handle_unknown = 'ignore')

X_train_encoded = oneh.fit_transform(X_train[cat])
X_test_encoded = oneh.transform(X_test[cat])

X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=oneh.get_feature_names_out(cat)) #Creating of a new dataframe
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=oneh.get_feature_names_out(cat)) #Creating of a new dataframe

X_train.reset_index(drop=True, inplace=True) #Resetting of indices to avoid not matching indices while concatenation
y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

X_train = pd.concat([X_train.drop(columns=cat), X_train_encoded_df], axis=1) #Concatenate encoded categorical features with other features in X_test
X_test = pd.concat([X_test.drop(columns=cat), X_test_encoded_df], axis=1)#Concatenate encoded categorical features with other features in X_train

#Scaling numerical variables.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Training Linear Regression model.

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

#Training Random Forest Regression model.

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit (X_train, y_train)

#Training Decision Tree Regression model.

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

import joblib

#Dumping models on joblib
joblib.dump(lr, 'trained_lr.joblib')
joblib.dump(rf, 'trained_rf.joblib')
joblib.dump(dt, 'trained_dt.joblib')
