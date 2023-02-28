import streamlit as st
import pandas as pd

def upload_file():
    df = None
    num_cols = None
    cat_cols = None

    st.subheader("1. Select your dataset")
    uploaded_file = st.file_uploader("Choose a file:")

    if uploaded_file is not None:
      df = pd.read_csv(uploaded_file)
      st.write(df)

    #add here more things for the mainpage
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            num_cols = st.multiselect('Choose the numerical columns', df.columns, key=1)
        with col2:
            cat_cols = st.multiselect('Choose categorical columns', [x for x in df.columns if x not in num_cols], key=2)

    return df, num_cols, cat_cols

