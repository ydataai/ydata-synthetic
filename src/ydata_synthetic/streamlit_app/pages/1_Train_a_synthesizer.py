from typing import Union
import os
import json
import streamlit as st

from ydata.sdk.synthesizers import RegularSynthesizer
from ydata.sdk.common.client import get_client

from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.regular.model import Model

from ydata_synthetic.streamlit_app.pages.functions.load_data import upload_file
from ydata_synthetic.streamlit_app.pages.functions.train import DataType, __CONDITIONAL_MODELS
from ydata_synthetic.streamlit_app.pages.functions.train import init_synth, advanced_setttings, training_parameters

def get_available_models(type: Union[str, DataType]):

    dtype = DataType(type)
    if dtype == DataType.TABULAR:
        models_list = [e.value.upper() for e in Model if e.value not in ['cgan', 'cwgangp']] + ['ydata-sdk Synthesizer']
    else:
        st.warning('Time-Series models are not yet supported .')
        models_list = ([''])
    return models_list

def run():
    model_name= None

    df, num_cols, cat_cols = upload_file()

    if df is not None:
        st.subheader("2. Select your synthesizer parameters")

        col_type, col_model = st.columns(2)

        with col_type:
            datatype = st.selectbox('Select your data type', (DataType.TABULAR.value, ))
        with col_model:
            if datatype is not None:
                models_list = get_available_models(type=datatype)
                model_name = st.selectbox('Select your model', models_list)

        if model_name not in ['', 'ydata-sdk Synthesizer']:
            st.text("Select your synthesizer model parameters")
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.number_input('Batch size', 0, 500, 500, 1)

            with col2:
                lr = st.number_input('Learning rate', 0.01, 0.1, 0.05, 0.01)

            with st.expander('**More settings**'):
                model_path = st.text_input("Saved trained model to path:", value="trained_synth.pkl")
                noise_dim, layer_dim, beta_1, beta_2 = advanced_setttings()

            # Create the Train parameters
            gan_args = ModelParameters(batch_size=batch_size,
                                        lr=lr,
                                        betas=(beta_1, beta_2),
                                        noise_dim=noise_dim,
                                        layers_dim=layer_dim)

            model = init_synth(datatype=datatype, modelname=model_name, model_parameters=gan_args)

            if model != None:
                st.text("Set your synthesizer training parameters")
                #Get the training parameters
                epochs, label_col = training_parameters(model_name, df.columns)

                train_args = TrainParameters(epochs=epochs)

                st.subheader("3. Train your synthesizer")
                if st.button('Click here to start the training process'):
                    with st.spinner("Please wait while your synthesizer trains..."):
                        if label_col is not None:
                            model.fit(data=df, num_cols=num_cols, cat_cols=cat_cols, train_arguments=train_args, label_cols=label_col)
                        else:
                            model.fit(data=df, num_cols=num_cols, cat_cols=cat_cols, train_arguments=train_args)

                    st.success('Synthesizer was trained succesfully!')
                    st.info(f"The trained model will be saved at {model_path}.")

                    model.save(model_path)



        if model_name == 'ydata-sdk Synthesizer':
            valid_token = False
            st.text("Model parameters")
            col1, col2 = st.columns(2)
            with col1:
                token = st.text_input("SDK Token", type="password")
                os.environ['YDATA_TOKEN'] = token

            with col2:
                st.write("##")
                try:
                    get_client()
                    st.text('✅ Valid')
                    valid_token = True
                except Exception:
                    st.text('❌ Invalid')

            if not valid_token:
                st.error("""**ydata-sdk Synthesizer requires a valid token.**    
                In case you do not have an account, please, create one at https://ydata.ai/ydata-fabric-free-trial.    
                To obtain the token, please, login to https://fabric.ydata.ai.    
                The token is available on the homepage once you are connected.
                """)
                

            with st.expander('**More settings**'):
                model_path = st.text_input("Saved trained model to path:", value="trained_synth.pkl")
            
            st.subheader("3. Train your synthesizer")
            if st.button('Click here to start the training process', disabled=not valid_token):
                model = RegularSynthesizer()
                with st.spinner("Please wait while your synthesizer trains..."):
                    dtypes = {}
                    for c in num_cols:
                        dtypes[c] = 'numerical'
                    for c in cat_cols:
                        dtypes[c] = 'categorical'
                    model.fit(X=df, dtypes=dtypes)

                st.success('Synthesizer was trained succesfully!')
                st.info(f"The trained model will be saved at {model_path}.")

                model_data = {
                    'uid': model.uid,
                    'token': os.environ['YDATA_TOKEN']
                }
                with open(model_path, 'w') as outfile:
                    json.dump(model_data, outfile)

                    
            

if __name__ == '__main__':
    run()