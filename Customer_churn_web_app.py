import streamlit as st
import requests
import io
import pickle
import pandas as pd
import numpy as np

# GitHub raw content URL for your model file
model_path = r"C:\Users\Administrator\Documents\AIsat\Group_Project\model_and_transformers2.sav"

# Load the model and transformers
with open(model_path, 'rb') as file:
    loaded_model, scaler, normalizer = pickle.load(file)

# Define global variables with default values
total_relationship_count = 1
total_revolving_bal = 0
total_amt_chng_q4_q1 = 0.275
total_trans_amt = 510
total_trans_ct = 10
total_ct_chng_q4_q1 = 0.206

def churn_prediction(rel_count, revol_bal, amt_chng_q4_q1, trans_amt, trans_ct, ct_chng_q4_q1):
    data = {
        'Total_Relationship_Count': [rel_count],
        'Total_Revolving_Bal': [revol_bal],
        'Total_Amt_Chng_Q4_Q1': [amt_chng_q4_q1],
        'Total_Trans_Amt': [trans_amt],
        'Total_Trans_Ct': [trans_ct],
        'Total_Ct_Chng_Q4_Q1': [ct_chng_q4_q1]
    }
    df = pd.DataFrame(data)
    df2array = np.asarray(df)
    reshape_array = df2array.reshape(1, -1)

    
    def transformation(reshape_array):
        scaler_reshape = scaler.transform(reshape_array)
        normalizer_reshape = normalizer.transform(scaler_reshape)
        return normalizer_reshape

    # Transform data
    transformed_data = transformation(reshape_array)


    # Make prediction
    prediction = loaded_model.predict(transformed_data)
    

    return prediction[0]



# Main function
def main():
    st.title('🚀 Customer Churn Prediction Web App')
    st.markdown(
        '<p style="font-size: 24px; color: #1F4D7A; animation: pulse 1s infinite;">Predict Customer Churn</p>',
        unsafe_allow_html=True
    )

    # Header image with centered alignment
    st.image(r"C:\Users\Administrator\Documents\AIsat\Group_Project\Customer-Churn.png",
             caption="Predict Customer Churn",
             use_column_width=True,
             )

    # Sidebar layout with rounded corners
    st.sidebar.markdown(
        '<style>div.Widget.row-widget.stRadio div[role="radiogroup"] > label {border-radius: 10px;}</style>',
        unsafe_allow_html=True
    )

    st.sidebar.subheader("Legend")
    st.sidebar.markdown('- **Total Relationship Count**: Enter the total relationship count.')
    st.sidebar.markdown('- **Total Revolving Balance**: Enter the total revolving balance.')
    st.sidebar.markdown('- **Total Amount Change Q4-Q1**: Enter the total amount change from Q4 to Q1.')
    st.sidebar.markdown('- **Total Transaction Amount**: Enter the total transaction amount.')
    st.sidebar.markdown('- **Total Transaction Count**: Enter the total transaction count.')
    st.sidebar.markdown('- **Total Count Change Q4-Q1**: Enter the total count change from Q4 to Q1.')

    # Main content layout with rounded corners
    st.markdown(
        '<style>div.Widget.stButton button{border-radius: 10px;}</style>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # First column
    with col1:
        # Use the global variables within this block
        total_relationship_count = st.number_input('Total Relationship Count',
                                                   min_value=1,
                                                   max_value=6,
                                                   value=1)

        total_revolving_bal = st.number_input('Total Revolving Balance',
                                              min_value=0,
                                              max_value=2517,
                                              value=0)

        total_amt_chng_q4_q1 = st.slider('Total Amount Change Q4-Q1',
                                         min_value=0.275,
                                         max_value=1.212,
                                         value=0.275,
                                         step=0.001)

    # Second column
    with col2:
        # Use the global variables within this block
        total_trans_amt = st.number_input('Total Transaction Amount',
                                          min_value=510,
                                          max_value=8618,
                                          value=510)

        total_trans_ct = st.number_input('Total Transaction Count',
                                         min_value=10,
                                         max_value=113,
                                         value=10)

        total_ct_chng_q4_q1 = st.number_input('Total Count Change Q4-Q1',
                                             min_value=0.206,
                                             max_value=1.182,
                                             value=0.206)

    # Animated button for prediction with a success icon
    if st.button('Predict Customer Churn', key='prediction_button', help="Click to predict customer churn"):
        with st.spinner('Predicting ⏳...'):
            # Prediction logic
            attrition = churn_prediction(total_relationship_count, total_revolving_bal, total_amt_chng_q4_q1,
                                         total_trans_amt, total_trans_ct, total_ct_chng_q4_q1)

        # Display prediction result with custom styling and icon
        result_placeholder = st.empty()

        # Check if attrition is not empty and handle the result
        if attrition is not None:      
            if attrition == 1:
                result_placeholder.error(' ❗ The customer is on the verge of churning. 🚨')
            else:
                result_placeholder.success('🎉 The customer is not on the verge of churning. 🌟')
        else:
            # Handle the case where attrition is empty or None
            st.warning('No prediction result. Please check your input values and try again.')

     # Option to upload a file with a file icon
    uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])
    if uploaded_file is not None:
        # Read the uploaded file
        uploaded_df = pd.read_csv(uploaded_file)
    
        
        # Make predictions for the uploaded data
        uploaded_df2np = np.asarray(uploaded_df)
        predicted_value = loaded_model.predict(uploaded_df2np)
        uploaded_df['predicted_churn'] = predicted_value.reshape(-1,1)
    
        # Download the CSV file with a download icon
        csv_data = uploaded_df.to_csv(index=False)
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
