# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 06:43:26 2020

@author: paras
"""

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import json
from sklearn.externals import joblib 


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
   
import flask

app = Flask(__name__)

main_cols = joblib.load("columns.pkl")
    

def clean_data(df_x):
    le = LabelEncoder()
    df_x.Gender = le.fit_transform(df_x.Gender)
    df_x = pd.get_dummies(data = df_x,  columns=["Geography"], drop_first = False)
    return df_x


def standardize_data(dta):
    scaler = joblib.load("std_scaler.pkl")
    X_transformed = scaler.transform(dta)
    return X_transformed


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    form_data = request.form.to_dict()
    
    df_input = pd.DataFrame.from_records([form_data], )
    df_input = df_input.drop(['submitBtn'], axis=1)
    df_input = pd.DataFrame(df_input)
 
    sample_df = pd.DataFrame(columns = main_cols)
    clean_df = clean_data(df_input)
    main_df = sample_df.append(clean_df)
    main_df = main_df.fillna(0)
    print(main_df)
    
         
    std_df = standardize_data(main_df)
    print(std_df)
    
    clf = joblib.load('prediction_classifier.pkl')
    pred = clf.predict(std_df)
    print(pred, pred[0], pred[0][0])
    x = round(pred[0][0]*100, 2)
    
    print(x)
    
    return flask.render_template('index.html', predicted_value="Customer Churn rate: {}%".format(str(x)))
    # return jsonify({'prediction': str(x)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)