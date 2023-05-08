import streamlit as st
from module.preprocessing import get_data, preprocessing
from module.cnn import create_model, get_accuracy

st.header("Convolutional Neural Network (CNN)")
st.write('''
Convolutional Neural Networks (CNN) can be used for spam filtering by analyzing the content of emails or text messages and detecting patterns that are characteristic of spam messages. The CNN uses a hierarchical approach to processing the input data, starting with low-level features and gradually building up to more complex patterns.
''')

st.subheader("Preprocessing")
st.write(
    "Dataset from this model are taken from UCI Machine Learning at [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).")

st.write("Dataset preview:")

df = get_data()
st.write(df.head())

st.write("Before build model, we need to do some preprocessing on the dataset.")
st.write("After preprocessing, the dataset will be like this:")
df = preprocessing(df)
st.write(df.head())

st.subheader("Model")
st.write(
    "Next we will create model like the image below. The image are taken from [here](https://arxiv.org/abs/1510.03820): ")

st.image('docs/images/cnn.png')

st.write('''
    Illustration of a Convolutional Neural Network (CNN) architecture for sentence classification. 
    Here we depict three filter region sizes: 2, 3 and 4, each of which has 2 filters.
    Every filter performs convolution on the sentence matrix and generates (variable-length) feature maps. 
    Then 1-max pooling is performed over each map, i.e., the largest number from each feature map is recorded. 
    Thus a univariate feature vector is generated from all six maps, and these 6 features are concatenated to form a feature vector for the penultimate layer. 
    The final softmax layer then receives this feature vector as input and uses it to classify the sentence; 
    here we assume binary classification and hence depict two possible output states. 

''')

st.write("Here is the model implementation in Python:")
st.code('''
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
''')

model = create_model(df)

st.write("Model summary:")
model.summary(print_fn=lambda x: st.caption(x))

st.subheader("Evaluation")
st.write("Model evaluation using accuracy score.")
st.write(get_accuracy(model))
