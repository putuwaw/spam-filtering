import streamlit as st

st.header("Spam Filtering")
st.write('''Spam filtering is a web app that can classify whether an email is spam or not.
            This web app was created to fulfill the mid semester assignment of Network Security subject.''')

st.subheader("Tech Stack")
st.write("The tech stack used in this project are:")
tech1, tech2, tech3, tech4 = st.columns(4)
with tech1:
    st.image("https://github.com/streamlit.png", width=100)
    st.write("[Streamlit](https://streamlit.io/)")
with tech2:
    st.image("https://github.com/keras-team.png", width=100)
    st.write("[Keras](https://keras.io/)")
with tech3:
    st.image("https://github.com/scikit-learn.png", width=100)
    st.write("[Scikit Learn](https://scikit-learn.org/stable/)")
with tech4:
    st.image("https://github.com/tensorflow.png", width=100)
    st.write("[Tensorflow](https://www.tensorflow.org/)")

st.subheader("Resume")
st.write(
    "The group resume can be found [here](https://github.com/putuwaw/spam-filtering/blob/main/docs/PJK_SPAM_FILTERING_E_1.pdf).")

st.subheader("Contributors")
person1, person2, person3, person4 = st.columns(4)
with person1:
    st.image("https://github.com/putuwaw.png", width=100)
    st.write("[Putu Widyantara](https://github.com/putuwaw)")
with person2:
    st.image("https://github.com/Kebelll.png", width=100)
    st.write("[Kenny Belle](https://github.com/Kebelll)")
with person3:
    st.image("https://github.com/madya-dev.png", width=100)
    st.write("[Madya Santosa](https://github.com/madya-dev)")
with person4:
    st.image("https://github.com/kamisama27.png", width=100)
    st.write("[Dheva Surya](https://github.com/kamisama27)")


st.subheader("Source Code")
st.write(
    "The source code can be found [here](https://github.com/putuwaw/spam-filtering).")
