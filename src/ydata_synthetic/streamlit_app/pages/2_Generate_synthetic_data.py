import streamlit as st
import json
import os

from ydata.sdk.synthesizers import RegularSynthesizer
from ydata.sdk.common.client import get_client

from ydata_synthetic.streamlit_app.pages.functions.train import DataType
from ydata_synthetic.streamlit_app.pages.functions.generate import load_model, generate_profile

def run():
    st.subheader("Generate synthetic data from a trained model")
    from_SDK = False
    model_data = {}
    valid_token = False
    col1, col2 = st.columns([4, 2])
    with col1:
        input_path = st.text_input("Provide the path to a trained model", value="trained_synth.pkl")
        # Try to load as a JSON as SDK
        try: 
            f = open(input_path)
            model_data = json.load(f)
            from_SDK = True
        except:
            pass

        if from_SDK:
            token = st.text_input("SDK Token", type="password", value=model_data.get('token'))
            os.environ['YDATA_TOKEN'] = token


    with col2:
        datatype = st.selectbox('Select your data type', (DataType.TABULAR.value,))
        datatype=DataType(datatype)

        if from_SDK and 'YDATA_TOKEN' in os.environ:
            st.write("##")
            try:
                get_client()
                st.text('✅ Valid')
                valid_token = True
            except Exception:
                st.text('❌ Invalid')
    
    if from_SDK and 'token' in model_data and not valid_token:
        st.warning("The token used during training is not valid anymore. Please, use a new token.")

    if from_SDK and not valid_token:
        st.error("""**ydata-sdk Synthesizer requires a valid token.**    
        In case you do not have an account, please, create one at https://ydata.ai/ydata-fabric-free-trial.    
        To obtain the token, please, login to https://fabric.ydata.ai.    
        The token is available on the homepage once you are connected.
        """)

    col1, col2 = st.columns([4,2])
    with col1:
        n_samples = st.number_input("Number of samples to generate", min_value=0, value=1000)
        profile = st.checkbox("Generate synthetic data profiling?", value=False)
    with col2:
        sample_path = st.text_input("Synthetic samples file path", value='synthetic.csv')

    if st.button('Generate samples'):
        if from_SDK:
            model = RegularSynthesizer.get(uid=model_data.get('uid'))

        else:
            model = load_model(input_path=input_path, datatype=datatype)

        st.success('The model was properly loaded and is now ready to generate synthetic samples!')


        #sample synthetic data
        with st.spinner('Generating samples... This might take time.'):
            synth_data = model.sample(n_samples)
        st.write(synth_data)

        #save the synthetic data samples to a given path
        synth_data.to_csv(sample_path)

        if profile:
            generate_profile(df=synth_data)

if __name__ == '__main__':
    run()