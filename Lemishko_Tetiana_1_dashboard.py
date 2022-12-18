#APP STREAMLIT : (commande : streamlit run XX/dashboard.py depuis le dossier python)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
#import time
import math
from urllib.request import urlopen
import json
import requests
import plotly.graph_objects as go 
import shap
#from sklearn.impute import SimpleImputer
#from sklearn.neighbors import NearestNeighbors
#from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")


def get_response(url):
    response = requests.get(url)
    print(response)
    return response.json()

#Chargement des données
df = pd.read_csv('app_test.csv')
df_full = pd.read_csv('application_test.csv')   
ignore_features = ['Unnamed: 0','SK_ID_CURR', 'INDEX', 'TARGET']
relevant_features = [col for col in df if col not in ignore_features]    



#Chargement du modèle
model = pickle.load(open('lgbm.pkl', 'rb'))

    #######################################
    # SIDEBAR
    #######################################

logo_image = "logo_oc.png"
shap_image = "Shap_features_importances.png"
stage_image = "stage.png"
gender_fail = "gender.png"
age_fail = "age.png"
loans = "loans.png"

with st.sidebar:
 st.header("💰 Client analysis")

 st.write("## Client ID")
 id_list = df["SK_ID_CURR"].tolist()
 id_client = st.selectbox(
            "Select Customer ID", id_list)

 st.write("## Actions")
 
 show_client_details = st.checkbox("View cliend info")
 show_comparison = st.checkbox("View comparison with other clients")
 show_credit_decision = st.checkbox("View credit decision")
 shap_local = st.checkbox("View local SHAP features importances")
 shap_general = st.checkbox("View global SHAP features importances")
            

    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################

    #Titre principal

html_temp = """
    <div style="background-color: gray; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Dashboard - Scoring Credit</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">
    Credit decision support for customer relationship managers</p>
    """
st.markdown(html_temp, unsafe_allow_html=True)

with st.expander("What is this app for?"):
        st.write("This app is used to predict the capacity of the client to repay the loan") 
        st.image(logo_image)


    #Afficher l'ID Client sélectionné
st.write("Select client ID :", id_client)

#-------------------------------------------------------
# Afficher la info du client
#-------------------------------------------------------

if (show_client_details):
    df_client = df[df['SK_ID_CURR'] == id_client]
    df_client_full = df_full[df_full['SK_ID_CURR'] == id_client]
    total_income = df_client['AMT_INCOME_TOTAL']
    duration_imployed = - df_client['DAYS_EMPLOYED']
    gender = df_client_full['CODE_GENDER']
    loan = df_client['AMT_CREDIT']
    age = abs(df_client_full['DAYS_BIRTH'])//365
    st.write('### Age of the client: ', age)
    st.write('### Gender of the client: ', gender)
    st.write('### Total income of the client: ', total_income)
    st.write('### Credit requested by the client: ', loan)
    st.write('### Working experience of the client in days: ', int(duration_imployed))
    
    
#-------------------------------------------------------
# Afficher la camparison avec les autres clients
#-------------------------------------------------------
    
if (show_comparison):
    st.header('‍Failure to repay the loan by age group')
    st.image('age.png')
    st.header('‍Failure to repay the loan by gender')
    st.image('gender.png')
    st.header('Avarage total income of the bank clients: 168797.91')
    st.header('‍10 highest credits granted by the bank:')
    st.image('loans.png')
    st.header('‍General distribution of the working experience among the clients of the bank')
    st.image('stage.png')


    
#-------------------------------------------------------
# Afficher la décision de crédit
#-------------------------------------------------------

if (show_credit_decision):
    st.header('‍⚖️ Scoring and Decision of the model')

            #Appel de l'API : 

    API_url = "https://fastapilemishko.herokuapp.com/predict?id_client=" + str(id_client)
    json_url = get_response(API_url)
    st.write("## Json {}".format(json_url))
    API_data = json_url
    
    classe_predite = API_data['prediction']
    if classe_predite == 1:
        decision = 'Bad Prospect (Loan Refused)'
        st.write(decision)
    else:
        decision = 'Good prospect (Loan Granted)'
        st.write(decision)

        
    #-------------------------------------------------------
# Afficher SHAP features importances du client
#-------------------------------------------------------
    
if (shap_local):
    st.header('‍Importance of the features defining the ability of the client to repay the loan:')
    shap.initjs()
    number = st.slider('Select the number of features to display?', \
                                   2, 20, 8)
    X = df[df['SK_ID_CURR']==int(id_client)]
    X = X[relevant_features]

    fig, ax = plt.subplots(figsize=(15, 15))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values[0], X, plot_type ="bar", \
    max_display=number, color_bar=False, plot_size=(8, 8))


    st.pyplot(fig)

        #-------------------------------------------------------
        # Afficher la feature importance globale
        #-------------------------------------------------------

if (shap_general):
    st.header('‍Global feature importance')
    st.image('Shap_features_importances.png')
    

    
#streamlit run streamlit_app.py