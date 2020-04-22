# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 06:43:26 2020

@author: paras
"""

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.externals import joblib 


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
   
import flask

app = Flask(__name__)

main_cols = joblib.load("columns.pkl")

def clean_data(data):
    data.loc[:, ["Gender"]] = gender["Female"]
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    data = np.array(columnTransformer.fit_transform(data), dtype = np.str)
    data = data[:, 1:]
    return data


def standardize_data(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    form_data = request.form.to_dict()
    
    df_input = pd.DataFrame.from_records([form_data], )
    df_input = df_input.drop(['submitBtn'], axis=1)
    print(type(df_input))
    
    main_df = pd.DataFrame(columns=main_cols)
    print(main_df)
    main_df = main_df.append(df_input)
    print(main_df)
   
    
    clf = joblib.load('prediction_classifier.pkl')
    clean_df = clean_data(df_input)
    std_df = standardize_data(clean_df)
    pred = clf.predict(std_df)


    return jsonify({'prediction': pred})


main_cols = joblib.load("columns.pkl")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)