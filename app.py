# create environment for windows
# python -m venv myenv
# activate environment
# myenv\Scripts\activate
# pip install  streamlit pandas numpy seaborn matplotlib scikit-learn

# importing all the libraries used in the project.
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#load model
model=pickle.load(open('model_gb.pkl','rb'))

#load scaler
scaler= MinMaxScaler()
#give the title to ur web page
st.title('INSURANCE PREMIUM PRICE PREDICTION APP')

#to take inputs from user.
age=st.number_input('Age',min_value=1,max_value=100,value=25)
gender=st.selectbox('Gender',('male','female'))
bmi=st.number_input('BMI',min_value=10.0,max_value=100.0,value=30.0) # BMI is always in float value
smoker=st.selectbox('Smoker',('yes','no'))
children=st.number_input('Number of Children',min_value=0,max_value=10,value=0)
region=st.selectbox('Region',('southwest','southeast','northwest','northeast'))

# encoding

# smoker
Smoker = 1 if smoker=='yes' else 0
# gender
sex_female	= 1 if gender=='female' else 0
sex_male = 1 if gender=='male' else 0

# region
region_dict ={'southwest':0,'northwest':1,'northeast':2,'southeast':3}
Region = region_dict[region]

# dataframe
input_features = pd.DataFrame({
    'age':[age],
    'bmi':[bmi],
    'children':[children],
    'Smoker':[Smoker],
    'sex_female':[sex_female],
    'sex_male':[sex_male]

})

scaler = MinMaxScaler()
input_features[['age','bmi']]= scaler.fit_transform(input_features[['age','bmi']])

# predictions
if st.button('Predict'):
  predictions=model.predict(input_features)
  output = round(np.exp(predictions[0]),2)
  st.success(f"Price Prediction: ${output}")

