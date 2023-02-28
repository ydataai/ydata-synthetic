"""
    ydata-synthetic streamlit app landing page
"""
import streamlit as st

def main():
    st.set_page_config(
        page_title="YData Synthetic - Synthetic data generation streamlit_app",
        page_icon="👋",
        layout="wide"
    )
    col1, col2 = st.columns([2, 4])

    with col1:
        st.image("https://assets.ydata.ai/oss/ydata-synthetic-_red.png", width=200)

    with col2:
        st.title("Welcome to YData Synthetic!")
        st.text("Your application for synthetic data generation!")

    st.markdown('[ydata-synthetic](https://github.com/ydataai/ydata-synthetic) is an open-source library and is used to generate synthetic data mimicking the real world data.')
    st.header('What is synthetic data?')
    st.markdown('Synthetic data is artificially generated data that is not collected from real-world events. It replicates the statistical components of real data containing no identifiable information, ensuring an individual’s privacy.')
    st.header('Why Synthetic Data?')
    st.markdown('''
    Synthetic data can be used for many applications:          
    - Privacy
    - Remove bias
    - Balance datasets
    - Augment datasets''')

    # read the instructions in x/
    st.markdown('This *streamlit_app* application can generate synthetic data for your dataset. '
                'Please read all the instructions in the sidebar before you start the process.')

    # read the instructions in x/
    st.subheader('Select & train a synthesizer')
    #Add here the example text for the end users

    st.markdown('''
    `ydata-synthetic` streamlit app enables the training and generation of synthetic data from generative architectures.
    The current app only provides support for the generation tabular data and for the following architectures:
    - GAN
    - WGAN
    - WGANGP
    - CTGAN
    ''')

    #best practives for synthetic data generation
    st.markdown('''
    ##### What you should ensure before training the synthesizer:
    - Make sure your dataset has no missing data. 
        - If missing data is a problem, no worries. Check the article and this article. 
    - Make sure you choose the right number of epochs and batch_size considering your dataset shape. 
        - The choice of these 2 parameters highly affects the results you may get. 
    - Make sure that you've the right data types selected. 
        - Only numerical and categorical values are supported.
        - In case date , datetime, or text is available in the dataset, the columns should be preprocessed before the model training.''')

    st.markdown('The trained synthesizer is saved to `*.trained_synth.pkl*` by default.')

    st.subheader('Generate & compare synthetic samples')

    st.markdown(''' 
    The ydata-synthetic app experience allows you to:
    - Generate as many samples as you want based on the provided input
    - Generate a profile for the generated synthetic samples
    - Save the generated samples to a local directory''')

    # guidelines for sampling and
    st.markdown(''' 
    ##### What you should ensure before generating synthetic samples:
    - If no model file path is provided, the default location `.trained_synth.pkl` is assumed.
    - Always choose the correct type of data, that corresponds to the trained model in order to avoid loading errors.''')

    st.subheader('Coming soon')
    st.markdown('''
    - Support for time-series models: TimeGAN
    - Integrate more advanced settings for CTGAN
    - Side-by-side comparison real vs synthetic data sample with `ydata-profiling`''')

if __name__ == '__main__':
    main()