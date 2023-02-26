import streamlit as st

st.set_page_config(
    page_title="YData Synthetic - Synthetic data generation streamlit",
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
st.markdown('Synthetic data is artificially generated data that is not collected from real world events. It replicates the statistical components of real data without containing any identifiable information, ensuring individuals privacy.')
st.header('Why Synthetic Data?')
st.markdown('''Synthetic data can be used for many applications:          
- Privacy
- Remove bias
- Balance datasets
- Augment datasets''')

# read the instructions in x/
st.markdown('This streamlit application can generate synthetic data for your dataset. '
            'Please read all the instructions in the sidebar before you start the process.')

# read the instructions in x/
st.subheader('Select & train a synthesizer')

st.subheader('Generate & compare synthetic samples')