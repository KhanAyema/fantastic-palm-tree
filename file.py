import pickle
import streamlit as st
import numpy as np
model1=pickle.load(open("Titanic.pkl","rb"))

def myf1():
    st.title("Titanic Survived Prediction")
    pclass=st.number_input("Enter the pclass:")
    age=st.number_input("Enter the age:")
    gender=st.number_input("Enter the gender:")
    pred=st.button("Predict")

    if pred:
        input_data=np.array([[pclass,age,gender]])
        op=model1.predict(input_data)
        st.write("Predicted output: \n 1=survived & 0=not survived",op)

myf1()