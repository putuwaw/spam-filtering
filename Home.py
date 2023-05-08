import streamlit as st
from module.cnn import predict

st.header("Spam Filtering")
st.write('''
            Spam filtering using using Deep Learning Algorithms - Convolutional Neural Network (CNN).
        ''')

text = st.text_area("Enter your text here:", height=100,
                    help="English text is recommended")

if st.button('Predict'):
    result = predict(text)
    st.subheader("Result")
    st.write(f"Your text are: **{result}**")
