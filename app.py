import pickle

import gdown
import numpy as np
import pandas as pd
import streamlit as st
import torch
from simpletransformers.classification import MultiLabelClassificationModel

st.markdown("# Abstract to categories converter")
st.markdown("### The app generates more suitable categories for your abstract")
st.markdown(
    "<img width=600px src='https://pub.mdpi-res.com/mathematics/mathematics-10-04398/article_deploy/html/images/mathematics-10-04398-g003.png'>",
    unsafe_allow_html=True)

st.markdown("")

text = st.text_area("PUT YOUR ABSTRACT HERE", height=200)


# @st.cache_resource
@st.cache_data
def load_model():
    model_loaded = MultiLabelClassificationModel('roberta', "roberta-base",
                                                 num_labels=93,
                                                 use_cuda=False)

    url = 'https://drive.google.com/uc?export=download&id=17DK_VEDnwhtawSozyhvqbKSTm-PQqt89'
    output = 'pytorch_model.bin'
    gdown.download(url, output, quiet=False)

    # torch.hub.download_url_to_file('https://drive.google.com/uc?export=download&id=17DK_VEDnwhtawSozyhvqbKSTm-PQqt89',
    #                                'pytorch_model.bin')
    model_loaded.model.load_state_dict(
        torch.load("pytorch_model.bin", map_location=torch.device('cpu')))
    return model_loaded


# @st.cache_resource
@st.cache_data
def load_encoder():
    with open('model/multi_label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return encoder


@st.cache_data
def get_predictions(model, text):
    predicted_categories_encoded, raw_outputs = model.predict([text])
    return predicted_categories_encoded, raw_outputs


@st.cache_data
def get_top(raw_outputs, _encoder):
    top_indices = np.argsort(raw_outputs[0])[::-1]
    cur_ind = 0
    categories = []
    confs = []
    while cur_ind < len(raw_outputs[0]):
        encoded = np.zeros_like(raw_outputs)
        encoded[0][top_indices[cur_ind]] = 1
        category = _encoder.inverse_transform(encoded)[0]
        categories.append(category[0])
        confs.append(raw_outputs[0][top_indices[cur_ind]])

        cur_ind += 1

    return categories, confs


model = load_model()
multi_label_encoder = load_encoder()

if text != "":
    predicted_categories_encoded, raw_outputs = get_predictions(model, text)
#     raw_outputs = np.random.random((1, 93))
    predicted_categories, confs = get_top(raw_outputs, multi_label_encoder)

    data = np.concatenate([[predicted_categories], np.round([confs], 3)], axis=0).T

    dataframe = pd.DataFrame(data, columns=["Category", "Confidence"])

    conf_thresh = st.slider('Confidence threshold', 0.0, 1.0, 0.5)
    dataframe_filtered = dataframe[dataframe["Confidence"].astype('float32') > conf_thresh]
    st.table(dataframe_filtered)
