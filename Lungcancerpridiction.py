import streamlit as st
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


st.markdown(""" <style> .font {font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;} </style> """, unsafe_allow_html=True)

st.markdown('<p class="font">DID YOU REALLY SAFE FROM LUNG CANCER?', unsafe_allow_html=True)

st.write('DISCLAIMER: This is just a prediction by using Naive Bayes model with a precision 96%.')

st.sidebar.write("""
This is a web app demo using python libraries such as Streamlit, Sklearn etc
""")

st.sidebar.write ("For more info, please contact:")

st.sidebar.write("<a href='www.linkedin.com/in/ikmal-fazli-876621234'>Mr. Ikmal Fazli</a>", unsafe_allow_html=True)


data = pd.read_csv(r'https://raw.githubusercontent.com/IkmalFazli/Lung-cancer-pridiction/main/survey%20lung%20cancer.csv')

labelencoder1 = LabelEncoder()
labelencoder2 = LabelEncoder()
labelencoder3 = LabelEncoder()
labelencoder4 = LabelEncoder()
labelencoder5 = LabelEncoder()
labelencoder6 = LabelEncoder()
labelencoder7 = LabelEncoder()
labelencoder8 = LabelEncoder()
labelencoder9 = LabelEncoder()
labelencoder10 = LabelEncoder()
labelencoder11 = LabelEncoder()
labelencoder12 = LabelEncoder()
labelencoder13 = LabelEncoder()
labelencoder14 = LabelEncoder()

data['GENDER'] = labelencoder1.fit_transform(data['GENDER'])
data['SMOKING'] = labelencoder2.fit_transform(data['SMOKING'])
data['YELLOW_FINGERS'] = labelencoder3.fit_transform(data['YELLOW_FINGERS'])
data['ANXIETY'] = labelencoder4.fit_transform(data['ANXIETY'])
data['PEER_PRESSURE'] = labelencoder5.fit_transform(data['PEER_PRESSURE'])
data['CHRONIC DISEASE'] = labelencoder6.fit_transform(data['CHRONIC DISEASE'])
data['FATIGUE'] = labelencoder7.fit_transform(data['FATIGUE'])
data['ALLERGY'] = labelencoder8.fit_transform(data['ALLERGY'])
data['WHEEZING'] = labelencoder9.fit_transform(data['WHEEZING'])
data['ALCOHOL CONSUMING'] = labelencoder10.fit_transform(data['ALCOHOL CONSUMING'])
data['COUGHING'] = labelencoder11.fit_transform(data['COUGHING'])
data['SHORTNESS OF BREATH'] = labelencoder12.fit_transform(data['SHORTNESS OF BREATH'])
data['SWALLOWING DIFFICULTY'] = labelencoder13.fit_transform(data['SWALLOWING DIFFICULTY'])
data['CHEST PAIN'] = labelencoder14.fit_transform(data['CHEST PAIN'])

X = data.drop(['LUNG_CANCER'], axis = 1)
y = data['LUNG_CANCER']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)


def prediction(gen, Age, smok, yf, Anx, PP, cd, Fat, All, whe, Alc, cough, Sho, swa, cp):
    nb = GaussianNB()
    nb.fit(Xtrain, ytrain)
    heart_data2 = pd.DataFrame(columns = ['GENDER','AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',  'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN' ])
    heart_data2 = heart_data2.append({'GENDER': gen,'AGE':Age, 'SMOKING':smok, 'YELLOW_FINGERS':yf, 'ANXIETY':Anx, 'PEER_PRESSURE':PP, 'CHRONIC DISEASE':cd, 'FATIGUE':Fat, 'ALLERGY':All, 'WHEEZING':whe,  'ALCOHOL CONSUMING':Alc, 'COUGHING':cough, 'SHORTNESS OF BREATH':Sho, 'SWALLOWING DIFFICULTY':swa, 'CHEST PAIN':cp}, ignore_index = True)
    ypred = nb.predict(heart_data2)
    if ypred =='YES':
                st.markdown("""<style>.header-style {font-size:25px;font-family:Cooper Black;color:#800000;}</style>""",unsafe_allow_html=True)
                st.markdown(f'<p class="header-style">DEAR {user_name} YOU MIGHT HAVE A LUNG CANCER, GO GET AN APPOINMENT WITH A DOCTOR!', unsafe_allow_html=True)
    elif ypred == 'NO':
                st.markdown("""<style>.newstyle{font-size:25px;font-family:Cooper Black;color:#0000FF;}</style>""",unsafe_allow_html=True)
                st.markdown(f"<p class='newstyle'> DEAR {user_name} YOU'RE SAVE!!! (for now.. hehhehehe)", unsafe_allow_html=True)
      
with st.form('my_form'):

    user_name = st.text_input("Name")
    B= st.text_input("Age")
    gender= st.radio("Select Gender: ", ('Male', 'Female'))
    if gender == 'Male':
        A = 1
    else :
        A = 0             
    smok= st.radio("Currently smoking? ", ('YES', 'NO'))
    if smok == 'YES':
        C = 1
    else :
        C = 0    
    yf= st.radio("Yellow Fingers? ", ('YES', 'NO'))
    if yf == 'YES':
        D = 1
    else :
        D = 0 
    Anx= st.radio("Having anxiety recently? ", ('YES', 'NO'))  
    if Anx == 'YES':
        E = 1
    else :
        E = 0
    PP= st.radio("Pear-pressure? ", ('YES', 'NO'))
    if PP == 'YES':
        F = 1
    else :
        F = 0
    cd= st.radio("Any Chronic disease? ", ('YES', 'NO'))
    if cd == 'YES':
        G = 1
    else :
        G = 0     
    Fat= st.radio("Fatique? ", ('YES', 'NO'))
    if Fat == 'YES':
        H = 1
    else :
        H = 0
    All= st.radio("Having any allergy? ", ('YES', 'NO'))
    if All == 'YES':
        I = 1
    else :
        I = 0        
    whe= st.radio("Wheezing?", ('YES', 'NO'))
    if whe == 'YES':
        J = 1
    else :
        J = 0       
    Alc= st.radio("Any alcohol consuming? ", ('YES', 'NO'))
    if Alc == 'YES':
        K = 1
    else :
        K = 0     
    cough= st.radio("Coughing? ", ('YES', 'NO'))
    if cough == 'YES':
        L = 1
    else :
        L = 0      
    Sho= st.radio("Having Shortness of breath? ", ('YES', 'NO'))
    if Sho == 'YES':
        M = 1
    else :
        M = 0   
    swa= st.radio("Any difficulty to swallow? ", ('YES', 'NO'))
    if swa == 'YES':
        N = 1
    else :
        N = 0     
    cp= st.radio("Any Chest pain? ", ('YES', 'NO'))
    if cp == 'YES':
        O = 1
    else :
        O = 0      
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    prediction(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O)

