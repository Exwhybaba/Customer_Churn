import numpy as np
import pandas as pd
import pickle
import streamlit as st
import requests
import io

# Loading the model
url = "https://raw.githubusercontent.com/Exwhybaba/Customer_Churn/main/trained6_model.sav"
response = requests.get(url)

if response.status_code == 200:
    loaded_model = pickle.loads(response.content)
else:
    st.error("Failed to retrieve the model file. Status code: {}".format(response.status_code))
    st.stop()

# prediction function
def churn_prediction_for_df(df):
    # Convert 'Gender' to numerical value
    gender_mapping = {'Female': 0, 'Male': 1}
    df['Gender'] = df['Gender'].map(gender_mapping)

    predictions = loaded_model.predict(df)

    return predictions

def main(debug=True):
    # Setting page layout with wide mode
    st.set_page_config(layout="wide")

    # giving a title with custom CSS
    st.title('🚀 Customer Churn Prediction Web App')
    st.markdown(
        '<style>h1{color: #1F4D7A; text-align: center;}</style>', unsafe_allow_html=True
    )

    # Header image with centered alignment
    st.image("https://raw.githubusercontent.com/Exwhybaba/Customer_Churn/main/Customer-Churn.png",
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
        # Getting user input for Total_Revolving_Bal
        Total_Revolving_Bal_min = 0
        Total_Revolving_Bal_max = 2517
        Total_Revolving_Bal = st.number_input('Total Revolving Balance',
                                              min_value=Total_Revolving_Bal_min,
                                              max_value=Total_Revolving_Bal_max,
                                              value=Total_Revolving_Bal_min)  # You can set a default value if needed

        # Getting user input for Total_Trans_Amt
        Total_Trans_Amt_min = 510
        Total_Trans_Amt_max = 8618
        Total_Trans_Amt = st.number_input('Total Transaction Amount',
                                          min_value=Total_Trans_Amt_min,
                                          max_value=Total_Trans_Amt_max,
                                          value=Total_Trans_Amt_min)  # You can set a default value if needed

        # Getting user input for Total_Trans_Ct
        Total_Trans_Ct_min = 10
        Total_Trans_Ct_max = 113
        Total_Trans_Ct = st.number_input('Total Transaction Count',
                                         min_value=Total_Trans_Ct_min,
                                         max_value=Total_Trans_Ct_max,
                                         value=Total_Trans_Ct_min)  # You can set a default value if needed

        # Getting user input for Total_Relationship_Count
        Total_Relationship_Count_min = 1
        Total_Relationship_Count_max = 6
        Total_Relationship_Count = st.number_input('Total Relationship Count',
                                                   min_value=Total_Relationship_Count_min,
                                                   max_value=Total_Relationship_Count_max,
                                                   value=Total_Relationship_Count_min)  # You can set a default value if needed

        # Getting user input for Months_Inactive_12_mon
        Months_Inactive_12_mon_min = 1
        Months_Inactive_12_mon_max = 6
        Months_Inactive_12_mon = st.number_input('Months Inactive 12 months',
                                                 min_value=Months_Inactive_12_mon_min,
                                                 max_value=Months_Inactive_12_mon_max,
                                                 value=Months_Inactive_12_mon_min)  # You can set a default value if needed

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

        # Display the table with predicted results
        st.dataframe(result_df)

        # Download the CSV file
        csv_data = result_df.to_csv(index=False)
        st.download_button(
            label="Download Predicted Results",
            data=io.StringIO(csv_data).read(),
            file_name="predicted_results.csv",
            key='download_button'
        )

if __name__ == '__main__':
    main()
