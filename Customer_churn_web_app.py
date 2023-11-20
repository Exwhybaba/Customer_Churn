import numpy as np
import pandas as pd
import pickle
import streamlit as st
import requests
import io

## Loading the model
url = "https://raw.githubusercontent.com/Exwhybaba/Customer_Churn/main/trained6_model.sav"
response = requests.get(url)

if response.status_code == 200:
    loaded_model = pickle.loads(response.content)
else:
    st.error("Failed to retrieve the model file. Status code: {}".format(response.status_code))
    st.stop()

def churn_prediction(Gender, Total_Revolving_Bal, Total_Trans_Amt, Total_Trans_Ct, Total_Relationship_Count,
                     Months_Inactive_12_mon):
    # Convert 'Gender' to numerical value
    gender_mapping = {'F': 0, 'M': 1}
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

# Main function
def main():
    # Setting page layout with wide mode
    st.set_page_config(layout="wide")

    # Background image and animated header
    st.markdown(
        f"""
        <style>
            body {{
                background-image: url("https://raw.githubusercontent.com/Exwhybaba/Customer_Churn/main/abstract-luxury-gradient-blue-background-smooth-dark-blue-with-black-vignette-studio-banner.jpg");
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('🚀 Customer Churn Prediction Web App')
    st.markdown(
        '<p style="font-size: 24px; color: #1F4D7A; animation: pulse 1s infinite;">Predict Customer Churn</p>',
        unsafe_allow_html=True
    )

    # Header image with centered alignment
    st.image("https://raw.githubusercontent.com/Exwhybaba/Customer_Churn/main/Customer-Churn.png",
             caption="Predict Customer Churn",
             use_column_width=True,
             )

    # Sidebar layout with rounded corners
    st.sidebar.markdown(
        '<style>div.Widget.row-widget.stRadio div[role="radiogroup"] > label {border-radius: 10px;}</style>',
        unsafe_allow_html=True
    )
    st.sidebar.subheader("Legend")
    st.sidebar.markdown('- **Gender**: Select "Female" or "Male" using the radio buttons.')
    st.sidebar.markdown('- **Total Revolving Balance**: Enter the total revolving balance.')
    st.sidebar.markdown('- **Total Transaction Amount**: Enter the total transaction amount.')
    st.sidebar.markdown('- **Total Transaction Count**: Enter the total transaction count.')
    st.sidebar.markdown('- **Total Relationship Count**: Enter the total relationship count.')
    st.sidebar.markdown('- **Months Inactive 12 months**: Enter the number of months inactive in the last 12 months.')

    # Main content layout with rounded corners
    st.markdown(
        '<style>div.Widget.stButton button{border-radius: 10px;}</style>',
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)

    # First column
    with col1:
        Gender = st.radio('Select gender:', options=['Female', 'Male'])

    # Second column
    with col2:
        Total_Revolving_Bal_min = 0
        Total_Revolving_Bal_max = 2517
        Total_Revolving_Bal = st.number_input('Total Revolving Balance',
                                              min_value=Total_Revolving_Bal_min,
                                              max_value=Total_Revolving_Bal_max,
                                              value=Total_Revolving_Bal_min)

        Total_Trans_Amt_min = 510
        Total_Trans_Amt_max = 8618
        Total_Trans_Amt = st.number_input('Total Transaction Amount',
                                          min_value=Total_Trans_Amt_min,
                                          max_value=Total_Trans_Amt_max,
                                          value=Total_Trans_Amt_min)

        Total_Trans_Ct_min = 10
        Total_Trans_Ct_max = 113
        Total_Trans_Ct = st.number_input('Total Transaction Count',
                                         min_value=Total_Trans_Ct_min,
                                         max_value=Total_Trans_Ct_max,
                                         value=Total_Trans_Ct_min)

        Total_Relationship_Count_min = 1
        Total_Relationship_Count_max = 6
        Total_Relationship_Count = st.number_input('Total Relationship Count',
                                                   min_value=Total_Relationship_Count_min,
                                                   max_value=Total_Relationship_Count_max,
                                                   value=Total_Relationship_Count_min)

        Months_Inactive_12_mon_min = 1
        Months_Inactive_12_mon_max = 6
        Months_Inactive_12_mon = st.number_input('Months Inactive 12 months',
                                                 min_value=Months_Inactive_12_mon_min,
                                                 max_value=Months_Inactive_12_mon_max,
                                                 value=Months_Inactive_12_mon_min)

    # Animated button for prediction with a success icon
    if st.button('Predict Customer Churn', key='prediction_button', help="Click to predict customer churn"):
        with st.spinner('Predicting...'):
            # Prediction logic
            attrition = churn_prediction(Gender, Total_Revolving_Bal, Total_Trans_Amt, Total_Trans_Ct,
                                         Total_Relationship_Count, Months_Inactive_12_mon)

        # Display prediction result with custom styling and icons
        result_placeholder = st.empty()
        if attrition[0] == '1':
            result_placeholder.error('❗ The customer is on the verge of churning. 🚨')
        else:
            result_placeholder.success('🎉 The customer is not on the verge of churning. 🌟')

    # Option to upload a file with a file icon
    uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])
    if uploaded_file is not None:
        # Read the uploaded file
        df_uploaded = pd.read_csv(uploaded_file)

    
        # Make predictions for the uploaded data
        predictions_df = pd.DataFrame({
         'Predicted Churn': churn_prediction(
        Gender=df_uploaded['Gender'].values,  # Extract values from the 'Gender' column
        Total_Revolving_Bal=df_uploaded['Total_Revolving_Bal'],
        Total_Trans_Amt=df_uploaded['Total_Trans_Amt'],
        Total_Trans_Ct=df_uploaded['Total_Trans_Ct'],
        Total_Relationship_Count=df_uploaded['Total_Relationship_Count'],
        Months_Inactive_12_mon=df_uploaded['Months_Inactive_12_mon']
    )
})


        # Combine the original data with predicted results
        result_df = pd.concat([df_uploaded, predictions_df], axis=1)

        # Display the table with predicted results
        st.dataframe(result_df)

        # Download the CSV file with a download icon
        csv_data = result_df.to_csv(index=False)
        st.download_button(
            label="Download Predicted Results",
            data=io.StringIO(csv_data).read(),
            file_name="predicted_results.csv",
            key='download_button',
            help="Click to download the predicted results"
        )

    # Real-time updates with placeholder and loading spinner
    result_placeholder = st.empty()
    result_placeholder.text("Waiting for predictions...")


if __name__ == '__main__':
    main()
