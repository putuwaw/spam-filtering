import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import Model
from keras.layers import Input, Embedding, Conv1D, Concatenate, GlobalMaxPooling1D, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from module.preprocessing import get_dataframe


@st.cache_data
def split_data(df: pd.DataFrame) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], random_state=42, test_size=.2)
    return X_train, X_test, y_train, y_test


@st.cache_resource
def create_model(df: pd.DataFrame) -> Model:
    X_train, _, y_train, _ = split_data(df)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    input_dim = len(tokenizer.word_index) + 1
    output_dim = 100
    input_length = 20

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_train_pad_seq = pad_sequences(X_train_seq, maxlen=input_length)

    inputs = Input(shape=(input_length,))

    embedding = Embedding(input_dim=input_dim,
                          output_dim=output_dim, input_length=input_length)(inputs)

    conv1 = Conv1D(filters=2, kernel_size=2, activation='relu')(embedding)
    conv2 = Conv1D(filters=2, kernel_size=3, activation='relu')(embedding)
    conv3 = Conv1D(filters=2, kernel_size=4, activation='relu')(embedding)

    concat = Concatenate()([GlobalMaxPooling1D()(
        conv1), GlobalMaxPooling1D()(conv2), GlobalMaxPooling1D()(conv3)])

    dense = Dense(units=6, activation='relu')(concat)

    outputs = Dense(units=2, activation='softmax')(dense)

    model = Model(inputs=inputs, outputs=outputs)

    one_hot_y_train = to_categorical(y_train)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(X_train_pad_seq, one_hot_y_train, epochs=5)

    return model


@st.cache_data
def predict(text: str) -> str:
    df = get_dataframe()
    model = create_model(df)
    text = [text]

    X_train, _, _, _ = split_data(df)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    seq = tokenizer.texts_to_sequences(text)
    pad = pad_sequences(seq, maxlen=20)

    predictions = model.predict(pad)

    result = "spam" if predictions[0][1] > predictions[0][0] else "not spam (ham)"

    return result


def get_accuracy(model: Model) -> float:
    df = get_dataframe()
    X_train, X_test, _, y_test = split_data(df)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad_seq = pad_sequences(X_test_seq, maxlen=20)

    y_pred = model.predict(X_test_pad_seq)
    y_class = np.argmax(y_pred, axis=1)
    return accuracy_score(y_test, y_class)
