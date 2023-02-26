from typing import Union
import streamlit as st

from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.regular.model import Model

from pages.functions.load_data import upload_file
from pages.functions.train import DataType, __CONDITIONAL_MODELS
from pages.functions.train import init_synth, advanced_setttings, training_parameters

def get_available_models(type: Union[str, DataType]):

    dtype = DataType(type)
    if dtype == DataType.TABULAR:
        models_list = [e.value.upper() for e in Model]
    else:
        models_list = ('', 'TimeGAN', )
    return models_list

def run():
    model_name= None
    model = None
    epochs=None
    model_path = "trained_synth.pkl"

    df, num_cols, cat_cols = upload_file()

    if df is not None:
        st.subheader("2. Select your synthesizer parameters")

        col_type, col_model = st.columns(2)

        with col_type:
            datatype = st.selectbox('Select your data type', (DataType.TABULAR.value, DataType.TIMESERIES.value))
        with col_model:
            if datatype is not None:
                models_list = get_available_models(type=datatype)
                model_name = st.selectbox('Select your model', models_list)

        if model_name !='':
            st.text("Select your synthesizer model parameters")
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.number_input('Batch size', 0, 500, 500, 1)

            with col2:
                lr = st.number_input('Learning rate', 0.01, 0.1, 0.05, 0.01)

            with st.expander('**Advanced settings**'):
                noise_dim, layer_dim, beta_1, beta_2 = advanced_setttings()

            # Create the Train parameters
            gan_args = ModelParameters(batch_size=batch_size,
                                       lr=lr,
                                       betas=(beta_1, beta_2),
                                       noise_dim=noise_dim,
                                       layers_dim=layer_dim)

            model = init_synth(datatype=datatype, modelname=model_name, model_parameters=gan_args)

            if model!=None:
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

                    st.success('Synthesizer was trained succesfully!!')

                    st.info(f"The trained model will be saved at {model_path}.")

                    model.save(model_path)