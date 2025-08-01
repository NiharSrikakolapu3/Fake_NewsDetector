import streamlit as st
from liveNews import fetch_political_headlines
from lime.lime_text import LimeTextExplainer
import numpy as np

from FakeNews import (
    load_glove_embeddings,
    load_model,
    clean_text,
    embed_text,
    predict_news,
)

# Load model and embeddings
@st.cache_resource
def load_resources():
    model = load_model("model_glove.pkl")
    embeddings_index = load_glove_embeddings("../data/glove.6B.100d.txt")
    return model, embeddings_index

model, embeddings_index = load_resources()

# LIME helper functions
def predict_proba_lime(texts, model, embeddings_index):
    probs = []
    for txt in texts:
        cleaned = clean_text(txt)
        vector = embed_text(cleaned, embeddings_index)
        prob = model.predict_proba([vector])[0]
        probs.append(prob)
    return np.array(probs)

def explain_with_lime(text, model, embeddings_index, num_features=10):
    class_names = ['Real', 'Fake']
    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda x: predict_proba_lime(x, model, embeddings_index),
        num_features=num_features
    )
    return explanation.as_list()

# --- UI ---
st.title("📰 Fake News Detector (GloVe + Logistic Regression)")
st.write("Enter a news headline or article below, or select a live headline for detection.")

# --- User Input Section ---
st.markdown("### ✍️ Analyze Your Own Text")
user_input = st.text_area("Enter News Text", height=200)

if st.button("Predict My Text"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        prediction = predict_news(user_input, model, embeddings_index)
        cleaned = clean_text(user_input)
        vector = embed_text(cleaned, embeddings_index)
        prob = model.predict_proba([vector])[0]

        st.markdown("#### 🔎 Prediction for Your Text")
        st.markdown(f"**Prediction**: {'🟥 Fake' if prediction == 'Fake' else '🟩 Real'}")
        st.markdown(f"**Probability Real**: `{prob[0]:.3f}`")
        st.markdown(f"**Probability Fake**: `{prob[1]:.3f}`")

        # Show LIME explanation
        st.markdown("#### 🧩 Word-level Explanation (LIME)")
        lime_exp = explain_with_lime(user_input, model, embeddings_index)
        for word, weight in lime_exp:
            color = "red" if weight > 0 else "green"
            st.markdown(f"<span style='color:{color};'>{word}: {weight:.3f}</span>", unsafe_allow_html=True)

# --- Live Headlines Section ---
st.markdown("---")
st.markdown("### 📰 Analyze a Recent Political Headline")

headlines_df = fetch_political_headlines()
headlines = headlines_df["text"].tolist()

if headlines:
    selected_headline = st.selectbox("Choose a headline", headlines)
    if st.button("Predict Selected Headline"):
        prediction = predict_news(selected_headline, model, embeddings_index)
        cleaned = clean_text(selected_headline)
        vector = embed_text(cleaned, embeddings_index)
        prob = model.predict_proba([vector])[0]

        st.markdown("#### 🔎 Prediction for Selected Headline")
        st.markdown(f"**Prediction**: {'🟥 Fake' if prediction == 'Fake' else '🟩 Real'}")
        st.markdown(f"**Probability Real**: `{prob[0]:.3f}`")
        st.markdown(f"**Probability Fake**: `{prob[1]:.3f}`")

        # Show LIME explanation for selected headline
        st.markdown("#### 🧩 Word-level Explanation (LIME)")
        lime_exp = explain_with_lime(selected_headline, model, embeddings_index)
        for word, weight in lime_exp:
            color = "red" if weight > 0 else "green"
            st.markdown(f"<span style='color:{color};'>{word}: {weight:.3f}</span>", unsafe_allow_html=True)
else:
    st.warning("No political headlines found. Try again later.")
