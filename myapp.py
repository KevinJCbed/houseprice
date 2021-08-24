import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.api as sm

st.title("DO YOU WANT TO KNOW YOUR HOUSE PRICE?")
st.image("https://cdn-res.keymedia.com/cms/images/ca/155/0348_637357588589988146.jpg")
training_data = pd.read_csv("https://raw.githubusercontent.com/KevinJCbed/houseprice/main/train.csv")

#st.caption("Here is the sample training data used to predict your house price")
#st.dataframe(training_data)
Y = training_data[['Price']]
X = training_data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]

model = sm.OLS(Y,X)
results = model.fit()

show_para = st.checkbox('Do you want to see model parameters?')
if show_para:
    st.write(results.summary())
predicted = results.predict(X)
plot1 = pd.DataFrame(data = {"actual":training_data['Price'], "predicted":predicted})

# fig, ax = plt.subplots()
# ax.scatter(plot1['actual'], plot1['predicted'])
# ax.legend()
# st.pyplot(fig)

area_income = st.number_input("Enter your average area income")
area_house_age = st.number_input("Enter your average area house age")
area_rooms = st.number_input("Enter your average number of rooms")
area_bedroom = st.number_input("Enter your average number of bedrooms")
area_pop = st.number_input("Enter your area's population")

output = results.predict([area_income, area_house_age, area_rooms, area_bedroom, area_pop])[0]

st.write(f"Based on your selection, your house price prediction is ***${output.round()/10**6} Million***")
