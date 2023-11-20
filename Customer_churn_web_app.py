import numpy as np
import pandas as pd
import pickle
import streamlit as st
import requests

# Loading the model
url = "https://raw.githubusercontent.com/Exwhybaba/Customer_Churn/main/trained2_model.sav"
response = requests.get(url)

if response.status_code == 200:
    loaded_model = pickle.loads(response.content)
else:
    st.error("Failed to retrieve the model file. Status code: {}".format(response.status_code))
    st.stop()

# Prediction function
def churn_prediction(Total_Revolving_Bal, Total_Trans_Amt, Total_Trans_Ct, NB_Classifier_2):
    data = {
        'Total_Revolving_Bal': [Total_Revolving_Bal],
        'Total_Trans_Amt': [Total_Trans_Amt],
        'Total_Trans_Ct': [Total_Trans_Ct],
        'NB_Classifier_2': [NB_Classifier_2],
    }
    df = pd.DataFrame(data)
    df2array = np.asarray(df)
    reshape_array = df2array.reshape(1, -1)

    prediction = loaded_model.predict(reshape_array)

    if prediction[0] == 1:
        return 'The customer is on the verge of churning.'
    else:
        return 'The customer is not on the verge of churning'

def main(debug=True):
    st.title('Customer Churn Prediction Web App')
    import streamlit as st
    st.image("https://raw.githubusercontent.com/Exwhybaba/Customer_Churn/main/Customer-Churn.png", use_container_width=True)


    Total_Revolving_Bal = st.number_input('Kindly input the total revolving balance')
    Total_Trans_Amt = st.number_input('Kindly input total transaction amount')
    Total_Trans_Ct1 = st.number_input('Kindly input total transaction Ct1')
    NB_Classifier_2 = st.number_input('Kindly input NB_Classifier_2')

    attrition = ''

    if st.button('Customer churn result'):
        attrition = churn_prediction(Total_Revolving_Bal, Total_Trans_Amt, Total_Trans_Ct1, NB_Classifier_2)
    st.success(attrition)

if __name__ == '__main__':
    main()
