import streamlit as st
import requests
import json

# Streamlit form
st.title("Titanic Survival Prediction")
pclass = st.selectbox("Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Erkek", "Kadın"])
title = st.selectbox("Title", ["Bay", "Hanım", "Bayan", "Usta", "Doktor", "Özgü"])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
embarked = st.selectbox("Embarked", ["Southampton, İngiltere", "Cherbourg, Fransa", "Queesntown, İrlanda"])

if st.button("Predict"):
    # Send request to FastAPI
    response = requests.post("http://fastapi:8000/predict", json={
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "embarked": embarked,
        "title": title
    })
    if response.status_code == 200:
        result = response.json()
        st.write(f"Prediction: {result['survive']}")
        st.write(f"Probability: {result['proba']:.2f}%")
    else:
        st.error("Error in prediction")