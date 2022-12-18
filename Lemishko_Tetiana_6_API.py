# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
#import numpy as np
import json5
from fastapi.logger import logger
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()

# Chargement des données 

df = pd.read_csv('test_sample.csv',sep=';')

print('df shape = ', df.shape)

#Chargement du modèle
model = pickle.load(open('lgbm.pkl', 'rb'))

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.post('/')
def index():
    return {'message': 'Hello, World'}

@app.get('/predict')
def credit(id_client):

    print('id client = ', id_client)
    
    #Récupération des données du client en question
    ID = int(id_client)
    X = df[df['SK_ID_CURR'] == ID]
    
    ignore_features = ['SK_ID_CURR', 'INDEX', 'TARGET']
    relevant_features = [col for col in df.columns if col not in ignore_features]

    X = X[relevant_features]
    
    print('X shape = ', X.shape)
    
    proba = model.predict_proba(X)
    prediction = model.predict(X)

    #DEBUG
    #print('id_client : ', id_client)
  
    dict_final = {
        'prediction' : int(prediction),
        'proba' : float(proba[0][1])
        }

    print('New prediction : \n', dict_final)

    return dict_final

    

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload