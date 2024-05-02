import streamlit as st
import pandas as pd

def upload_file():
    # Initialize the dataframe and numerical/categorical column variables
    df = None
    num_cols = None
    cat_cols = None

    # Display a subheader for file upload
    st.subheader("1. Select your dataset")

    # Allow user to upload a file
    uploaded_file = st.file_uploader("Choose a file:", type="csv")

    # If a file is uploaded, read it into a pandas dataframe
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Display the dataframe to the user
        st.write(df)

    # If a dataframe was created, display two columns for numerical and categorical column selection
    if df is not None:
        col1, col2 = st.columns(2)

        # In the first column, allow the user to select numerical columns
        with col1:
            num_cols = st.multiselect('Choose the numerical columns', df.select_dtypes(include=["int64", "float64"]).columns, key=1)

        # In the second column, allow the user to select categorical columns
        with col2:
            cat_cols = st.multiselect('Choose categorical columns', [x for x in df.columns if x not in num_cols], key=2)

    # Return the dataframe and selected columns
    return df, num_cols, cat_cols
