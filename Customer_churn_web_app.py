import numpy as np
import pandas as pd
import pickle
import streamlit as st
import requests
import io

# ... (previous code remains unchanged)

def churn_prediction_for_df(df):
    # Convert 'Gender' to numerical value
    gender_mapping = {'Female': 0, 'Male': 1}
    df['Gender'] = df['Gender'].map(gender_mapping)

    predictions = loaded_model.predict(df)

    return predictions

def main(debug=True):
    # ... (previous code remains unchanged)

    # Button for prediction
    if st.button('Predict Customer Churn', key='prediction_button'):
        attrition = churn_prediction(Gender, Total_Revolving_Bal, Total_Trans_Amt, Total_Trans_Ct,
                                     Total_Relationship_Count, Months_Inactive_12_mon)

        # Display prediction result with custom styling
        if attrition[0] == '1':
            st.error('❗ The customer is on the verge of churning.')
        else:
            st.success('🎉 The customer is not on the verge of churning.')

    # Option to upload a file
    uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])
    if uploaded_file is not None:
        # Read the uploaded file
        df_uploaded = pd.read_csv(uploaded_file)

        # Make predictions for the uploaded data
        predictions_df = pd.DataFrame({'Predicted Churn': churn_prediction_for_df(df_uploaded)})

        # Combine the original data with predicted results
        result_df = pd.concat([df_uploaded, predictions_df], axis=1)

        # Download the CSV file
        csv_data = result_df.to_csv(index=False)
        st.download_button(
            label="Download Predicted Results",
            data=io.StringIO(csv_data).read(),
            file_name="predicted_results.csv",
            key='download_button'
        )
