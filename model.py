# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 05:42:28 2020

@author: paras
"""

import pandas as pd
import numpy as np
from sklearn.externals import joblib 

df = pd.read_csv("Churn_Modelling.csv")

df_x = df.iloc[:, 3:13]
df_y = df.iloc[:, 13]

main_cols = df_x.columns
joblib.dump(main_cols, "columns.pkl")

# df.isna().sum()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

encoder = LabelEncoder()
df_x.iloc[:, 2] = encoder.fit_transform(df_x.iloc[:, 2])

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
col_tnf = columnTransformer.fit_transform(df_x)
df_x = np.array(col_tnf, dtype = np.str)
df_x = df_x[:, 1:]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0,0]+cm[1, 1])/(cm[0,0]+cm[1, 1]+cm[1,0]+cm[0, 1])

joblib.dump(classifier, 'prediction_classifier.pkl') 

