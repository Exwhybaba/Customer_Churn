import streamlit as st
import requests
import io
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import zipfile as zf
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, Normalizer, MinMaxScaler, StandardScaler, OneHotEncoder,LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,f1_score
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE, ADASYN,SMOTENC
from imblearn.under_sampling import RandomUnderSampler
import pickle
from sklearn.metrics import confusion_matrix 


# In[4]:


path = "https://raw.githubusercontent.com/Exwhybaba/Customer_Churn/main/credit_card_churn.csv"

# In[5]:


#with zf.ZipFile(path, 'r') as myfile:
    #myfile.extractall()


# In[6]:


#filePaths = r"C:\Users\Administrator\Documents\AIsat\Group_Project\datasets"


df1 = pd.read_csv(path)

# renaming the long columns
df1.rename(columns={'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1':'NB_Classifier_1',
       'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2': 'NB_Classifier_2'}, 
          inplace=True)



bins = [20, 35, 50, 100]
labels = ['Young Adult', 'Adult', 'Senior']
df1['Age_Group'] = pd.cut(df1['Customer_Age'], bins= bins, labels= labels)


# In[19]:


#mapping the categorical data
catgorical = df1.select_dtypes('object')




number = df1.select_dtypes('number')


# In[31]:



# ### Treating for Outliers

# In[51]:


outliers = []
def detect_outliers_iqr(data):
    data = sorted(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    # print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    # print(lwr_bound, upr_bound)
    for i in data: 
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
    return outliers


# In[53]:


sample_outliers = detect_outliers_iqr(df1['Total_Trans_Amt'])
df1.drop(df1[df1['Total_Trans_Amt'].isin(sample_outliers)].index, inplace=True)


# In[55]:


sample_outliers = detect_outliers_iqr(df1['Total_Amt_Chng_Q4_Q1'])
df1.drop(df1[df1['Total_Amt_Chng_Q4_Q1'].isin(sample_outliers)].index, inplace=True)


# In[56]:



# In[57]:


sample_outliers = detect_outliers_iqr(df1['Total_Ct_Chng_Q4_Q1'])
df1.drop(df1[df1['Total_Ct_Chng_Q4_Q1'].isin(sample_outliers)].index, inplace=True)


# ## Encoding

# In[61]:


cate = df1.select_dtypes('object')



# In[62]:


for i in cate.columns:
    print(i, cate[i].unique())


# In[63]:


df1['Age_Group'].unique()



# In[65]:


df1['Attrition_Flag'].unique()


# In[67]:


feature = [f for f in df1.columns if f != 'Attrition_Flag']
target =  [t for t in df1.columns if t == 'Attrition_Flag']

X = df1[feature]
y = df1[target]

X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size= 0.20, random_state=42)


# In[68]:


y_train['Attrition_Flag'] = y_train['Attrition_Flag'].map(lambda x : 1 if x == 'Attrited Customer' else 0 )
y_test['Attrition_Flag'] = y_test['Attrition_Flag'].map(lambda x : 1 if x == 'Attrited Customer' else 0 )


# In[69]:


# Apply SMOTENC to balance the dataset
categorical_features = [2, 4, 5,6, 7,22]
smotenc = SMOTENC(sampling_strategy='auto', categorical_features=categorical_features, random_state=42)
X_resampled, y_resampled = smotenc.fit_resample(X_train, y_train)
X_retst, y_retst = smotenc.fit_resample(X_test, y_test)


# In[70]:


X_train = X_resampled
y_train =  y_resampled

X_test = X_retst
y_test = y_retst


# In[71]:

# ## Encoding

# In[72]:


#Education Encoding

# Define the order of categories
education_order = ['Unknown','Uneducated', 'College', 'Graduate', 'High School', 'Post-Graduate', 'Doctorate']

# Initialize the OrdinalEncoder with the specified order
edu_ordinal_encoder = OrdinalEncoder(categories=[education_order]).fit(X_train[['Education_Level']])

# Fit and transform on the training data
X_train['Education_Level'] = edu_ordinal_encoder.transform(X_train[['Education_Level']])

# Transform the test data using the same encoder
X_test['Education_Level'] = edu_ordinal_encoder.transform(X_test[['Education_Level']])



# Marital encoding
# Define the order of categories
marital_order = ['Unknown','Single', 'Married', 'Divorced']

# Initialize the OrdinalEncoder with the specified order
mar_ordinal_encoder = OrdinalEncoder(categories=[marital_order]).fit(X_train[['Marital_Status']])

# Fit and transform on the training data
X_train['Marital_Status'] = mar_ordinal_encoder.transform(X_train[['Marital_Status']])

# Transform the test data using the same encoder
X_test['Marital_Status'] = mar_ordinal_encoder.transform(X_test[['Marital_Status']])



#Income encodng
# Define the order of categories
income_order = ['Unknown','Less than $40K', '$40K - $60K', '$60K - $80K', 
       '$80K - $120K', '$120K +']

# Initialize the OrdinalEncoder with the specified order
inc_ordinal_encoder = OrdinalEncoder(categories=[income_order]).fit(X_train[['Income_Category']])

# Fit and transform on the training data
X_train['Income_Category'] = inc_ordinal_encoder.transform(X_train[['Income_Category']])

# Transform the test data using the same encoder
X_test['Income_Category'] = inc_ordinal_encoder.transform(X_test[['Income_Category']])




#Card encoding
# Define the order of categories
card_order = ['Blue', 'Silver', 'Gold', 'Platinum']

# Initialize the OrdinalEncoder with the specified order
card_ordinal_encoder = OrdinalEncoder(categories=[card_order]).fit(X_train[['Card_Category']])

# Fit and transform on the training data
X_train['Card_Category'] = card_ordinal_encoder.transform(X_train[['Card_Category']])

# Transform the test data using the same encoder
X_test['Card_Category'] = card_ordinal_encoder.transform(X_test[['Card_Category']])





#Age Encoding
# Define the order of categories
age_order = ['Adult', 'Senior', 'Young Adult']

# Initialize the OrdinalEncoder with the specified order
age_ordinal_encoder = OrdinalEncoder(categories=[age_order]).fit(X_train[['Age_Group']])

# Fit and transform on the training data
X_train['Age_Group'] = age_ordinal_encoder.transform(X_train[['Age_Group']])

# Transform the test data using the same encoder
X_test['Age_Group'] = age_ordinal_encoder.transform(X_test[['Age_Group']])




#Gender encoding
# Initialize the Binarizer
binarizer = LabelBinarizer().fit(X_train['Gender'])
X_train['Gender']= binarizer.transform(X_train['Gender'])

X_test['Gender'] = binarizer.transform(X_test['Gender'])


# In[75]:


X_train_df = pd.DataFrame(X_train, columns = feature)



# In[76]:


X_test_df = pd.DataFrame(X_test, columns = feature)


# ## Features Selection

# In[77]:


X_train_df['Attrition_Flag'] = y_train




# In[79]:


columntoDrop = ['Avg_Open_To_Buy', 'NB_Classifier_2', 'NB_Classifier_1']
X_test_df.drop(columns=columntoDrop, inplace= True)
X_train_df.drop(columns=columntoDrop, inplace= True)


# In[81]:


columntoDrop = ['Attrition_Flag']
X_train_df.drop(columns=columntoDrop, inplace= True)



RFE_ = ['Total_Relationship_Count', 'Total_Revolving_Bal', 'Total_Amt_Chng_Q4_Q1','Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1']



# In[84]:

# In[85]:


X_train = X_train_df[RFE_].values
X_test = X_test_df[RFE_].values


#Rescaling
scaler = MinMaxScaler(feature_range = (0,1)).fit(X_train)
R_Xtrain = scaler.transform(X_train)
R_Xtest = scaler.transform(X_test)


# In[89]:


R_Xtrain_df = pd.DataFrame(R_Xtrain, columns = X_train_df[RFE_].columns)




# In[91]:


#Normalizer
normalizer = Normalizer().fit(R_Xtrain)
NR_Xtrain = normalizer.transform(R_Xtrain)
NR_Xtest = normalizer.transform(R_Xtest)


# In[92]:


NR_Xtrain_df = pd.DataFrame(NR_Xtrain, columns = X_train_df[RFE_].columns)


# In[93]:



# In[94]:


X_train = NR_Xtrain
X_test = NR_Xtest


# In[95]:






model = model = RandomForestClassifier(n_estimators= 1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)




# In[103]:

## Loading the model
#url = #"https://drive.google.com/file/d/1FFz55ZI78 -PwqmESTaTojfYRitImrVxm/view?usp=drive_link"
#response = requests.get(url)

#if response.status_code == 200:
    #loaded_model = pickle.loads(response.content)
#else:
    #st.error("Failed to retrieve the model file. #Status code: {}".format(response.status_code))
    #st.stop()

# Function for churn prediction
def churn_prediction(Total_Relationship_Count, Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1,
                    Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1):
    data = {
        'Total_Relationship_Count': [Total_Relationship_Count],
        'Total_Revolving_Bal': [Total_Revolving_Bal],
        'Total_Amt_Chng_Q4_Q1': [Total_Amt_Chng_Q4_Q1],
        'Total_Trans_Amt': [Total_Trans_Amt],
        'Total_Trans_Ct': [Total_Trans_Ct],
        'Total_Ct_Chng_Q4_Q1': [Total_Ct_Chng_Q4_Q1]
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
    prediction = model.predict(transformed_data)

    if prediction[0] == 1:
        print('The customer is on the verge of churning.')
    else:
        print('The customer is not on the verge of churning')

    return df




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

    # Helper function to set page config only once
    def set_page_config_once():
        if not hasattr(st, '_page_config_called'):
            st.set_page_config(layout="wide")
            st._page_config_called = True


if __name__ == '__main__':
    main()

