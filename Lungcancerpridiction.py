import streamlit as st
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


st.title('DID YOU REALLY SAVE FROM LUNG CANCER?')

st.sidebar.write("""
This is a web app demo using python libraries such as Streamlit, Sklearn etc
""")

st.sidebar.write ("For more info, please contact:")

st.sidebar.write("<a href='www.linkedin.com/in/ikmal-fazli-876621234'>Mr. Ikmal Fazli</a>", unsafe_allow_html=True)

choice = st.sidebar.radio(
    "Choose a model to predict:",   
    ('Random Forest','KNN Neughbors', 'support vector machine','logistic regression','Gaussian Naive Bayes'),
    index = 0
)



form = st.form(key='my_form')
user_name = form.text_input("Name")
Age = form.text_input("Age")
gender= form.text_input("Gender(M for Male, F for Female)")
smok= form.text_input("Currently smoking?(YES/NO)")
yf= form.text_input("Yellow Fingers? (YES/NO)")
Anx= form.text_input("Having anxiety recently?(YES/NO)")
PP= form.text_input("Pear-pressure?(YES/NO)")
cd= form.text_input("Any Chronic disease?(YES/NO)")
Fat= form.text_input("Fatique?(YES/NO)")
All= form.text_input("Any Allergy reaction?(YES/NO)")
whe= form.text_input("Wheeing?(YES/NO)")
Alc= form.text_input("Any alcohol consuming?(YES/NO)")
cough= form.text_input("Coughing?(YES/NO)")
Sho= form.text_input("Having Shortness of breath?(YES/NO)")
swa= form.text_input("Any difficulty to swallow?(YES/NO)")
cp= form.text_input("Any Chest pain?(YES/NO)")

submit_button = form.form_submit_button(label='Predict')

if submit_button:
         heart_data2 = pd.DataFrame(columns = ['GENDER','AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',  'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN' ])
         heart_data2 = heart_data2.append({'GENDER':gender,'AGE':Age, 'SMOKING':smok, 'YELLOW_FINGERS':yf, 'ANXIETY':Anx, 'PEER_PRESSURE':PP, 'CHRONIC DISEASE':cd, 'FATIGUE':Fat, 'ALLERGY':All, 'WHEEZING':whe,  'ALCOHOL CONSUMING':Alc, 'COUGHING':cough, 'SHORTNESS OF BREATH':sho, 'SWALLOWING DIFFICULTY':swa, 'CHEST PAIN':cp}, ignore_index = True)
         data = pd.read_csv('survey lung cancer.csv', index_col = False)
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
         data['FATIGUE '] = labelencoder7.fit_transform(data['FATIGUE '])
         data['ALLERGY '] = labelencoder8.fit_transform(data['ALLERGY '])
         data['WHEEZING'] = labelencoder9.fit_transform(data['WHEEZING'])
         data['ALCOHOL CONSUMING'] = labelencoder10.fit_transform(data['ALCOHOL CONSUMING'])
         data['COUGHING'] = labelencoder11.fit_transform(data['COUGHING'])
         data['SHORTNESS OF BREATH'] = labelencoder12.fit_transform(data['SHORTNESS OF BREATH'])
         data['SWALLOWING DIFFICULTY'] = labelencoder13.fit_transform(data['SWALLOWING DIFFICULTY'])
         data['CHEST PAIN'] = labelencoder14.fit_transform(data['CHEST PAIN'])

         X = data.drop(['LUNG_CANCER'], axis = 1)
         y = data['LUNG_CANCER']

         if choice == 'Random Forest':
             RandomForest = RandomForestClassifier(n_estimators=10000, max_depth=15)
             RandomForest.fit(Xtrain, ytrain)
             ypred = RandomForest.predict(heart_data2)
             if ypred == '0':
                 st.write("You might have lung cancer, go get a doctor now!)
             else:
                 st.write("You're as healthy as a horse")

         elif choice == 'KNN Neughbors':
             knn = KNeighborsClassifier()
             knn.fit(X, y)
             ypred = knn.predict(heart_data2)
             if ypred == '0':
                 st.write("You might have lung cancer, go get a doctor now!)
             else:
                 st.write("You're as healthy as a horse")

         elif choice == 'logistic regression':
             logreg = LogisticRegression()
             logreg.fit(Xtrain, ytrain)
             ypred = logreg.predict(heart_data2)
             if ypred == '0':
                 st.write("You might have lung cancer, go get a doctor now!)
             else:
                 st.write("You're as healthy as a horse")

         else:
             nb = GaussianNB()
             nb.fit(Xtrain, ytrain)
             ypred = nb.predict(heart_data2)
             if ypred == '0':
                 st.write("You might have lung cancer, go get a doctor now!)
             else:
                 st.write("You're as healthy as a horse")

    
