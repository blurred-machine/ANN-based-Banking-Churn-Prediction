#!/usr/bin/env python
# coding: utf-8


# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 05:42:28 2020

@author: paras
"""

import pandas as pd
import numpy as np
from sklearn.externals import joblib 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("Churn_Modelling.csv")

df_x = df.iloc[:, 3:13]
df_y = df.iloc[:, 13]

df_x.head()

# df.isna().sum()



def clean_data(df):

    le = LabelEncoder()
    df.Gender = le.fit_transform(df.Gender)
    df = pd.get_dummies(data = df, columns=["Geography"], drop_first = False)
    df = df.sort_index(axis=1)
    return df




df_x = clean_data(df_x)
# df_x.head()



# df_one = pd.DataFrame(df_x.iloc[2, :])
# df_one = df_one.T
# df_one
# clean_data(df_one)





# columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
# col_tnf = columnTransformer.fit_transform(df_x)
# df_x = np.array(col_tnf, dtype = np.str)
# df_x = df_x[:, 1:]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 0)

joblib.dump(X_train.columns, "columns.pkl")



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "std_scaler.pkl")
print(X_test)
print(X_train.shape[1])



import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 30)



y_pred = classifier.predict(X_test)
print("\nPredicted values: "+str(y_pred)+"\n")
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0,0]+cm[1, 1])/(cm[0,0]+cm[1, 1]+cm[1,0]+cm[0, 1])
print("\nTest Accuracy: "+str(accuracy)+"\n")
joblib.dump(classifier, 'prediction_classifier.pkl') 



