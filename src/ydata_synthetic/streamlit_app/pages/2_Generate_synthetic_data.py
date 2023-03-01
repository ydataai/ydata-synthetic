import streamlit as st

from ydata_synthetic.streamlit_app.pages.functions.train import DataType
from ydata_synthetic.streamlit_app.pages.functions.generate import load_model, generate_profile

def run():
    st.subheader("Generate synthetic data from a trained model")

    col1, col2 = st.columns([4, 2])
    with col1:
        input_path = st.text_input("Provide the path to a trained model", value="trained_synth.pkl")
    with col2:
        datatype = st.selectbox('Select your data type', (DataType.TABULAR.value,))
        datatype=DataType(datatype)

    col1, col2 = st.columns([4,2])
    with col1:
        n_samples = st.number_input("Number of samples to generate", min_value=0, value=1000)
        profile = st.checkbox("Generate synthetic data profiling?", value=False)
    with col2:
        sample_path = st.text_input("Synthetic samples file path", value='synthetic.csv')

    if st.button('Generate samples'):
        #load a trained model
        model = load_model(input_path=input_path,
                           datatype=datatype)

        st.success('Trained model was loaded. You can now generate synthetic samples')

        #sample synthetic data
        synth_data = model.sample(n_samples)
        st.write(synth_data)

        #save the synthetic data samples to a given path
        synth_data.to_csv(sample_path)

        if profile:
            generate_profile(df=synth_data)

if __name__ == '__main__':
    run()