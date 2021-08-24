import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import random
import os
import statsmodels.api as sm


#os.chdir("/Users/JC/Documents/Python Course/Real projects/housing_project")

st.title("DO YOU WANT TO KNOW YOUR HOUSE PRICE?")

training_data = pd.read_csv("https://raw.githubusercontent.com/KevinJCbed/houseprice/main/train.csv")

st.caption("Here is the sample training data used to predict your house price")
st.dataframe(training_data)

model = sm.OLS(
    training_data[['Price']],
    training_data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']])

results = model.fit()
#print(results.summary())

area_income = st.number_input("Enter your average area income")
area_house_age = st.number_input("Enter your average area house age")
area_rooms = st.number_input("Enter your average number of rooms")
area_bedroom = st.number_input("Enter your average number of bedrooms")
area_pop = st.number_input("Enter your area's population")


st.write(" ")
output = results.predict([area_income, area_house_age, area_rooms, area_bedroom, area_pop])[0]

st.write(f"Based on your selection, your house price prediction is ***${output.round()/10**6} Million***")
