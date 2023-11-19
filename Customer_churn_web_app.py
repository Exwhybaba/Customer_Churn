# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 21:49:29 2023

@author: Administrator
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st

#loading the model
path = r"C:\Users\Administrator\Documents\AIsat\Group_Project\trained2_model.sav"

loaded_model = pickle.load(open(path, mode= 'rb'))

#prediction function
def churn_prediction(Total_Revolving_Bal, Total_Trans_Amt, Total_Trans_Ct, NB_Classifier_2):
    data = {
        'Total_Revolving_Bal' : [Total_Revolving_Bal], 
        'Total_Trans_Amt' : [Total_Trans_Amt], 
        'Total_Trans_Ct' : [Total_Trans_Ct], 
        'NB_Classifier_2' : [NB_Classifier_2],
    }
    #convert the data to pandas
    df = pd.DataFrame(data)

    #convert data numpy array
    df2array = np.asarray(df)
    #reshape the array
    reshape_array = df2array.reshape(1, -1)

    
    prediction = loaded_model.predict(reshape_array)

    if prediction[0] == 1:
        return 'The customer is on the verge of churning.'
    else:
        return'The customer is not on the verge of churning'
    
    
    
def main(debug = True):
    #giving a title
    st.title('Customer Churn Prediction Web App')
    st.image(r"C:\Users\Administrator\Documents\AIsat\Group_Project\Customer-Churn.png", width=200)

    
    #getting the input data from the user
    Total_Revolving_Bal = st.number_input('Kindly input the total revolving balance')
    Total_Trans_Amt = st.number_input('Kindly input total transaction amount')
    Total_Trans_Ct1 = st.number_input('Kindly input total transaction Ct1')
    NB_Classifier_2 = st.number_input('Kindly input NB_Classifier_2')

    #code for prediction
    
    attrition = ''
    
    #creating a button for prediction
    if st.button('Customer churn result'):
        attrition = churn_prediction(Total_Revolving_Bal, 
                                     Total_Trans_Amt, Total_Trans_Ct1, NB_Classifier_2)
    st.success(attrition)
    
if __name__ == '__main__':
    main()

    
    
    
    
    
    
