import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

st.set_page_config(page_title="My project", page_icon=":tada", layout="wide")
st.markdown("""
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .half-width{
            width =50%;
    }
    .button{
            background-color:red;
            color:white;
            border-radius:5px;
            padding:10px 20px;
            border:none;
            text:center;
    }
    </style>
    <div class="centered">
        <h1>Heart Disease Prediction Webpage</h1>
    </div>
    """, unsafe_allow_html=True)

name = st.text_input("Name:")

gender_options = ['Male', 'Female']
gender = st.selectbox("Gender:", options=gender_options)

age = st.number_input("Age:", step=1)

hypertension_options = ["Yes", "No"]
hypertension = st.selectbox("Hypertension:", options=hypertension_options)

Heart_disease_options = ["Yes", "No"]
Heart_disease = st.selectbox("Heart Disease:", options=Heart_disease_options)

married_options = ["Yes", "No"]
married = st.selectbox("Married:", options=married_options)

work_type_options = ["Private", "Self-Employed", "Govt Jobs", "Children", "Never Worked"]
work = st.selectbox("Work Type:", options=work_type_options)

residence_options = ["Rural", "Urban"]
residence = st.selectbox("Residence Type:", options=residence_options)

glucose = st.number_input("Glucose Level:")
bmi = st.number_input("BMI:")

smoking_options = ["Never Smoke", "Formerly Smoke", "Smokes", "Unknown"]
smoking = st.selectbox("Smoking Status:", options=smoking_options)

st.markdown("""
    <div style="display: flex; justify-content: center; margin-top: 20px; border-radius:10px; border:none;">
        <button class="button_style" onclick="alert('Form submitted!')">Submit</button>
    </div>
    """, unsafe_allow_html=True)


g = 0 if gender == 'Female' else 1
ht = 1 if hypertension == 'Yes' else 0
h = 1 if Heart_disease == 'Yes' else 0
m = 1 if married == 'Yes' else 0

work_mapping = {
    "Private": 0,
    "Self-Employed": 1,
    "Govt Jobs": 2,
    "Children": 3,
    "Never Worked": 4
}
w = work_mapping[work]

residence_mapping = {
    "Rural": 0,
    "Urban": 1
}
r = residence_mapping[residence]

smoking_mapping = {
    "Never Smoke": 0,
    "Formerly Smoke": 1,
    "Smokes": 2,
    "Unknown": 3
}
s = smoking_mapping[smoking]

userlist = [g, age, ht, h, m, w, r, glucose, bmi,s]
userdata = np.array([userlist])


data= pd.read_csv(r"https://raw.githubusercontent.com/Sabbish99/Python_project/main/new%20numaric%20value.csv")

X = data.iloc[:, 0:10]  
y = data.iloc[:, 10] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
userscaled=scaler.transform(userdata)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.subheader('Accuracy')
container=st.container(border=True)
container.write(f"Accuracy: {acc*100:.2f}%")

userresult=knn.predict(userscaled)

st.subheader('Result')
output ='  '  
if userresult[0] == 0:
        output='Negative'
        container=st.container(border=True)
        container.write(output)
else:
        output='Possitive'
        container=st.container(border=True)
        container.write(output)
  

