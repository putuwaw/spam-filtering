import re
import string
import streamlit as st
import pandas as pd


@st.cache_data
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv('docs/spam.csv', encoding='latin-1')


@st.cache_data
def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(how="any", axis=1, inplace=True)
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['message'] = df['message'].apply(clean_text)
    return df


@st.cache_data
def get_dataframe():
    df = get_data()
    df = preprocessing(df)
    return df
