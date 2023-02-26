import streamlit as st
from ydata_profiling import ProfileReport

from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers.timeseries import TimeGAN
from streamlit_pandas_profiling import st_profile_report

from pages.functions.train import DataType

compare=None
model=None
input_path=None
sample_path=None

st.subheader("Generate synthetic data from a trained model")

col1, col2 = st.columns([4, 2])
with col1:
    input_path = st.text_input("Provide the path to a trained model")
with col2:
    datatype = st.selectbox('Select your data type', (DataType.TABULAR.value, DataType.TIMESERIES.value))
    datatype=DataType(datatype)

if input_path!= '' and input_path is not None:
    if datatype == DataType.TABULAR:
        model = RegularSynthesizer.load(input_path)
    else:
        model = TimeGAN.load(input_path)

if model:
    st.success('Trained model was loaded. You can now generate synthetic samples')

    col1, col2 = st.columns([4,2])
    with col1:
        n_samples = st.number_input("Number of samples to generate", min_value=0, value=1000)
        profile = st.checkbox("Generate synthetic data profiling?", value=False)
    with col2:
        sample_path = st.text_input("Synthetic samples file path", value=None)

    if st.button('Generate samples'):
        synth_data = model.sample(n_samples)
        st.write(synth_data)

        if sample_path is not None:
            synth_data.to_csv(sample_path)

        if profile:
            report = ProfileReport(synth_data, title='Synthetic data profile', interactions=None)
            st_profile_report(report)