# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 21:49:29 2023

@author: Administrator
"""

import numpy as np
import pickle
import streamlit as st

#loading the model
path = https://github.com/Exwhybaba/Customer_Churn/blob/main/trained_model.sav

loaded_model = pickle.load(open(path, mode= 'rb'))

#prediction function
def churn_prediction(Customer_Age, Total_Relationship_Count, Months_Inactive_12_mon,
                     Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal,
                     Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Ct_Chng_Q4_Q1,
                     Avg_Utilization_Ratio, ordIncome__Income_Category,
                     cat__Gender_F, cat__Gender_M):
    
    input = (Customer_Age, Total_Relationship_Count, Months_Inactive_12_mon,
                     Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal,
                     Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Ct_Chng_Q4_Q1,
                     Avg_Utilization_Ratio, ordIncome__Income_Category,
                     cat__Gender_F, cat__Gender_M)
    input2np = np.asarray(input)

    #reshape the array as we are predicing for one instance
    reshape_array = input2np.reshape(1,-1)

    prediction = loaded_model.predict(reshape_array)
    

    if prediction[0] == 1:
        return 'The customer is on the verge of churning.'
    else:
        return 'The customer is not on the verge of churning'
    
    
    
    
def main():
    #giving a title
    st.title('Customer Churn Prediction Web App')
    
    #getting the input data from the user
    
    
    Customer_Age = st.text_input('Input the customer age')
    Total_Relationship_Count = st.text_input('What is the customer relationship count')
    Months_Inactive_12_mon = st.text_input('Months inactive for 12 months')
    Contacts_Count_12_mon = st.text_input('Contact count for 12 months')
    Credit_Limit = st.text_input('Credit limit')
    Total_Revolving_Bal = st.text_input('Total revolving balance')
    Total_Amt_Chng_Q4_Q1 = st.text_input('Total amount change Q4-Q1')
    Total_Trans_Amt = st.text_input('Total transaction amount')
    Total_Ct_Chng_Q4_Q1 = st.text_input('Total count change Q4-Q1')
    Avg_Utilization_Ratio = st.text_input('Average utilization ratio')
    ordIncome__Income_Category = st.text_input('Income category')
    cat__Gender_F = st.text_input("Gender: Input 1 if it is female; otherwise, input 0.")
    cat__Gender_M = st.text_input("Gender: Input 1 if it is male; otherwise, input 0.")
    
    #code for prediction
    
    attrition = ''
    
    #creating a button for prediction
    if st.button('Customer churn result'):
        attrition = churn_prediction(Customer_Age, Total_Relationship_Count,
                                     Months_Inactive_12_mon,Contacts_Count_12_mon,Credit_Limit,
                                     Total_Revolving_Bal,Total_Amt_Chng_Q4_Q1,Total_Trans_Amt,
                                     Total_Ct_Chng_Q4_Q1,Avg_Utilization_Ratio,
                                     ordIncome__Income_Category,cat__Gender_F,cat__Gender_M)
    
    st.success(attrition)
    
if __name__ == '__main__':
    main()

    
    
    
    
    
    
