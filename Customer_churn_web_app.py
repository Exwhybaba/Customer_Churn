import streamlit as st
import requests
import io
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, Normalizer, MinMaxScaler, StandardScaler, OneHotEncoder,LabelBinarizer
import pickle
import requests

# GitHub raw content URL for your model file
model_url = 'https://raw.githubusercontent.com/Exwhybaba/Customer_Churn/main/model_and_transformers.sav'


# Download the model file
response = requests.get(model_url)
with open('model_and_transformers.sav', 'wb') as file:
    file.write(response.content)

# Load the model and transformers
with open('model_and_transformers.sav', 'rb') as file:
    loaded_model, scaler, normalizer = pickle.load(file)



def churn_prediction(Total_Relationship_Count, Total_Revolving_Bal,Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Trans_Ct,Total_Ct_Chng_Q4_Q1):
    data = {
        'Total_Relationship_Count': [Total_Relationship_Count],
        'Total_Revolving_Bal' : [Total_Revolving_Bal], 
        'Total_Amt_Chng_Q4_Q1' : [Total_Amt_Chng_Q4_Q1],
        'Total_Trans_Amt' : [Total_Trans_Amt], 
        'Total_Trans_Ct' : [Total_Trans_Ct], 
        'Total_Ct_Chng_Q4_Q1' : [Total_Ct_Chng_Q4_Q1]
        
    }
    df = pd.DataFrame(data)
    df2array = np.asarray(df)
    reshape_array = df2array.reshape(1, -1)

    def transformation(reshape_array):
        scaler_reshape = scaler.transform(reshape_array)
        normalizer_reshape = normalizer.transform(scaler_reshape)
        return normalizer_reshape

    #transform data
    transformed_data = transformation(reshape_array)

    # make prediction
    prediction = loaded_model.predict(transformed_data)

    if prediction[0] == 1:
        print('The customer is on the verge of churning.')
    else:
        print('The customer is not on the verge of churning')

    return df





# Main function
def main():
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
        Total_Relationship_Count_min = 1
        Total_Relationship_Count_max = 6
        Total_Relationship_Count = st.number_input('Total Relationship Count',
                                                   min_value=Total_Relationship_Count_min,
                                                   max_value=Total_Relationship_Count_max,
                                                   value=Total_Relationship_Count_min)
    
        Total_Revolving_Bal_min = 0
        Total_Revolving_Bal_max = 2517
        Total_Revolving_Bal = st.number_input('Total Revolving Balance',
                                              min_value=Total_Revolving_Bal_min,
                                              max_value=Total_Revolving_Bal_max,
                                              value=Total_Revolving_Bal_min, 
                                             )
    
        Total_Amt_Chng_Q4_Q1_min = 0.275
        Total_Amt_Chng_Q4_Q1_max = 1.212
        Total_Amt_Chng_Q4_Q1 = st.slider('Total Amount Change Q4-Q1',
                                               min_value=Total_Amt_Chng_Q4_Q1_min,
                                               max_value=Total_Amt_Chng_Q4_Q1_max,
                                               value=Total_Amt_Chng_Q4_Q1_min,
                                           step=0.001)
    
    # Second column
    with col2:
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
    
        Total_Ct_Chng_Q4_Q1_min = 0.206
        Total_Ct_Chng_Q4_Q1_max = 1.182
        Total_Ct_Chng_Q4_Q1 = st.number_input('Total Count Change Q4-Q1',
                                    min_value=Total_Ct_Chng_Q4_Q1_min,
                                    max_value=Total_Ct_Chng_Q4_Q1_max,
                                    value=Total_Ct_Chng_Q4_Q1_min)  
    
    
    
       
       # Animated button for prediction with a success icon
        if st.button('Predict Customer Churn', key='prediction_button', help="Click to predict customer churn"):
            with st.spinner('Predicting...'):
                # Prediction logic
                attrition = churn_prediction(Total_Relationship_Count, Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1,
                        Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1)
    
            # Display prediction result with custom styling and icons
            result_placeholder = st.empty()
            # Check if the predicted value is 1
            if attrition == 1:
                result_placeholder.error('❗ The customer is on the verge of churning. 🚨')
            else:
                result_placeholder.success('🎉 The customer is not on the verge of churning. 🌟')


        # Option to upload a file with a file icon
        uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])
        if uploaded_file is not None:
            # Read the uploaded file
            uploaded_df = pd.read_csv(uploaded_file)
    
        
            # Make predictions for the uploaded data
            uploaded_df2np = np.asarray(uploaded_df)
            predicted_value = model.predict(uploaded_df2np)
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

