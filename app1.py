import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import tensorflow as tf 
from tensorflow.keras.models import load_model
import pickle
import streamlit as st

## load the pickle files
model=load_model('model.h5')

## load other pickle files 
with open('label_encoder.pkl','rb') as file:
    label_encoder=pickle.load(file)

with open('oneHot_encoder.pkl','rb') as file:
    onehot_encoder=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

st.title('Customer chrin prediction')

## input data
geography=st.selectbox('Geography',onehot_encoder.categories_[0])
gender=st.selectbox('Gender',label_encoder.classes_)
age=st.slider('Age',15,98)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Slider',0,20)
no_of_products=st.slider('No of Products',0,5)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_mem=st.selectbox('Is Active Member',[0,1])


## preparing the input data

data=pd.DataFrame({
    'Gender':[label_encoder.transform([gender])[0]],
    'CreditScore':[credit_score],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[no_of_products],
    'HasCrCard':[has_cr_card],
    'IsActivemember':[is_active_mem],
    'EstimatedSalary':[estimated_salary]

})

## oneHot encoding Geography

geo_encoded = onehot_encoder.transform([[geography]]).toarray()
df=pd.DataFrame(geo_encoded,columns=onehot_encoder.get_feature_names_out(["Geography"]))

input = pd.concat([data.reset_index(drop=True),df],axis=1)

## scale the data

scaled_data=scaler.transform(input.values
                             )

prediction=model.predict(scaled_data)
prediction_prob=prediction[0][0]

st.write('probability is :',prediction_prob)
if prediction_prob >= 0.5:
    st.write('Custermer likely to join')
else:
    st.write('Customer not likely to join')
