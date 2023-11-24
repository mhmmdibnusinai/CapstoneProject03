import pickle
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Load the model
model = pickle.load(open('model/final_model.sav', 'rb'))

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def main():
    # Load picture
    image_hospital = Image.open('img/hospital.jpg')

    # Add option to select online or offline prediction
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch")
        )

    # Add explanatory text and picture in the sidebar
    st.sidebar.info('This app is created to predict diabetes patient')    
    st.sidebar.image(image_hospital)

    # Add title
    st.title("Diabetes Prediction App")

    if add_selectbox == 'Online':

        # Set up the form to fill in the required data 
        age = st.number_input(
            'age', min_value=0, max_value=17)
        job = st.number_input(
            'job', min_value=0, max_value=200)
        balance = st.number_input(
            'balance', min_value=0, max_value=125)
        housing = st.number_input(
            'housing', min_value=0, max_value=100)
        loan = st.number_input(
            'loan', min_value=0, max_value=900)
        contact = st.number_input(
            'contact', min_value=0, max_value=70)
        month = st.number_input(
            'month', min_value=0, max_value=3)
        campaign = st.number_input(
            'campaign', min_value=0, max_value=100)
        pdays = st.number_input(
            'pdays', min_value=0, max_value=100)
        poutcome = st.number_input(
            'poutcome', min_value=0, max_value=100)
        deposit = st.number_input(
            'deposit', min_value=0, max_value=100)
    
        # Convert form to data frame
        input_df = pd.DataFrame([
            {
                'age': age,
                'job': job,
                'balance': balance,
                'housing': housing,
                'loan': loan,
                'contact': contact,
                'month': month,
                'campaign': campaign,
                'pdays' : pdays,
                'poutcome': poutcome,
                'deposit' : deposit
                }
            ]
        )
        
        # Set a variabel to store the output
        output = ""

        # Make a prediction 
        if st.button("Predict"):
            output = model.predict(input_df)
            if (output[0] == 0):
                output = 'potential'
            else:
                output = 'not potential'

        # Show prediction result
        st.success(output)          

    if add_selectbox == 'Batch':

        # Add a feature to upload the file to be predicted
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            # Convert the file to data frame
            data = pd.read_csv(file_upload)

            # Select only columns required by the model
            data = data[[
                'age',
                'job',
                'balance',
                'housing',
                'loan',
                'contact',
                'month',
                'campaign',
                'pdays',
                'poutcome',
                'deposit'
                ]
            ]

            # Make predictions
            data['Prediction'] = np.where(model.predict(data)==1, 'yes', 'no')

            # Show the result on page
            st.write(data)

            # Add a button to download the prediction result file 
            st.download_button(
                "Press to Download",
                convert_df(data),
                "Prediction Result.csv",
                "text/csv",
                key='download-csv'
            )

if __name__ == '__main__':
    main()