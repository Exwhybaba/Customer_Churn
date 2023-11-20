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


# prediction function
def churn_prediction(Gender, Total_Revolving_Bal, Total_Trans_Amt, Total_Trans_Ct, Total_Relationship_Count,
                     Months_Inactive_12_mon):
    # Convert 'Gender' to numerical value
    gender_mapping = {'Female': 0, 'Male': 1}
    Gender = gender_mapping[Gender]

    data = {
        'Gender': [Gender],
        'Total_Revolving_Bal': [Total_Revolving_Bal],
        'Total_Trans_Amt': [Total_Trans_Amt],
        'Total_Trans_Ct': [Total_Trans_Ct],
        'Total_Relationship_Count': [Total_Relationship_Count],
        'Months_Inactive_12_mon': [Months_Inactive_12_mon],
    }
    # convert the data to pandas
    df = pd.DataFrame(data)

    # convert data numpy array
    df2array = np.asarray(df)
    # reshape the array
    reshape_array = df2array.reshape(1, -1)

    prediction = loaded_model.predict(reshape_array)

    if prediction[0] == 1:
        return 'The customer is on the verge of churning.'
    else:
        return 'The customer is not on the verge of churning'


def main(debug=True):
    # Setting page layout with wide mode
    st.set_page_config(layout="wide")

    # giving a title with custom CSS
    st.title('🚀 Customer Churn Prediction Web App')
    st.markdown(
        '<style>h1{color: #1F4D7A; text-align: center;}</style>', unsafe_allow_html=True
    )

    # Header image with centered alignment
    st.image(r"C:\Users\Administrator\Documents\AIsat\Group_Project\Customer-Churn.png",
             caption="Predict Customer Churn",
             use_column_width=True,
             )

    # Sidebar layout
    st.sidebar.subheader("Legend")
    st.sidebar.markdown('- **Gender**: Select "Female" or "Male" using the radio buttons.')
    st.sidebar.markdown('- **Total Revolving Balance**: Enter the total revolving balance.')
    st.sidebar.markdown('- **Total Transaction Amount**: Enter the total transaction amount.')
    st.sidebar.markdown('- **Total Transaction Count**: Enter the total transaction count.')
    st.sidebar.markdown('- **Total Relationship Count**: Enter the total relationship count.')
    st.sidebar.markdown('- **Months Inactive 12 months**: Enter the number of months inactive in the last 12 months.')

    # Main content layout
    col1, col2 = st.columns(2)

    # First column
    with col1:
        Gender = st.radio('Select gender:', options=['Female', 'Male'])

    # Second column
    with col2:
        Total_Revolving_Bal = st.number_input('Total Revolving Balance', min_value=0.0)
        Total_Trans_Amt = st.number_input('Total Transaction Amount', min_value=0.0)
        Total_Trans_Ct = st.number_input('Total Transaction Count', min_value=0.0)
        Total_Relationship_Count = st.number_input('Total Relationship Count', min_value=0)
        Months_Inactive_12_mon = st.number_input('Months Inactive 12 months', min_value=0)

    # Button for prediction
    if st.button('Predict Customer Churn', key='prediction_button'):
        attrition = churn_prediction(Gender, Total_Revolving_Bal, Total_Trans_Amt, Total_Trans_Ct,
                                     Total_Relationship_Count, Months_Inactive_12_mon)

        # Display prediction result with custom styling
        if prediction[0] == 1:
            st.error('❗ The customer is on the verge of churning.')
        else:
            st.success('🎉 The customer is not on the verge of churning.')


if __name__ == '__main__':
    main()
