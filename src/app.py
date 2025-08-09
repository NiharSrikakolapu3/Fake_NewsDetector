import streamlit as st
from liveNews import fetch_political_headlines
from lime.lime_text import LimeTextExplainer
import numpy as np
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
from PIL import Image

from FakeNews import (
    load_glove_embeddings,
    load_model,
    clean_text,
    embed_text,
    predict_news,
    predict_proba_lime,
    explain_with_lime
)

FEEDBACK_LOG = "./feedback_log.csv"

@st.cache_resource
def load_resources():
    logistic_model = load_model("model_logistic.pkl")
    rf_model = load_model("model_random_forest.pkl")
    embeddings_index = load_glove_embeddings("../data/glove.6B.100d.txt")
    return {"Logistic Regression": logistic_model, "Random Forest": rf_model}, embeddings_index

models, embeddings_index = load_resources()

def log_feedback(row):
    df = pd.DataFrame([row])

    # Write header only if the file is empty
    write_header = os.path.getsize(FEEDBACK_LOG) == 0

    try:
        df.to_csv(FEEDBACK_LOG, mode='a', header=write_header, index=False)
        st.success("Feedback saved!")
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

if "feedback_list" not in st.session_state:
    st.session_state.feedback_list = []

st.title("Fake Political News Detector")
st.write("Enter a news headline or article below, or select a live headline for detection.")

selected_model_name = st.selectbox("Choose Model", list(models.keys()))
model = models[selected_model_name]

confidence_threshold = st.slider(
    "Set prediction confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.01,
    help="Adjust the threshold above which news is classified as Fake."
)

#This method is created to analyze personal text, to see if your text can pass as a real or fake news
st.markdown("Analyze Your Own Text and Submit Feedback")

with st.form("analysis_form"):
    user_input = st.text_area("Enter News Text", height=200)
    analyze_btn = st.form_submit_button("Analyze")

if analyze_btn and user_input.strip():
    cleaned = clean_text(user_input)
    vector = embed_text(cleaned, embeddings_index)
    prob = model.predict_proba([vector])[0]
    prediction = 'Fake' if prob[1] > confidence_threshold else 'Real'

    # Store feedback but don't show message yet
    st.session_state.current_feedback = {
        "timestamp": datetime.now().isoformat(),
        "text": user_input,
        "model": selected_model_name,
        "prediction": prediction,
        "confidence_real": prob[0],
        "confidence_fake": prob[1],
        "threshold": confidence_threshold,
        "feedback": "agree",  # default- but will change because the user should get to pick what they think of the current prediction
    }

# Show prediction and feedback submission UI only if analysis done
if "current_feedback" in st.session_state:
    cf = st.session_state.current_feedback

    st.markdown("Prediction for Your Text")
    st.markdown(f"**Model Used**: `{cf['model']}`")
    st.markdown(f"**Prediction**: {'🟥 Fake' if cf['prediction'] == 'Fake' else '🟩 Real'}")
    st.markdown(f"**Probability Real**: `{cf['confidence_real']:.3f}`")
    st.markdown(f"**Probability Fake**: `{cf['confidence_fake']:.3f}`")
    st.markdown(f"**Threshold**: {cf['threshold']:.2f}")

    st.markdown("**LIME Explanation (word impact):**")
    lime_exp = explain_with_lime(cf["text"], model, embeddings_index)
    for word, weight in lime_exp:
        color = "red" if weight > 0 else "green"
        st.markdown(f"<span style='color:{color};'>{word}: {weight:.3f}</span>", unsafe_allow_html=True)

    # Feedback radio and submit
    fb_choice = st.radio(
        "Do you agree with this prediction?",
        ["Agree", "Disagree"],
        index=0 if cf["feedback"] == "agree" else 1,
        key="feedback_radio_user"
    )

    # Update session state feedback
    if fb_choice.lower() != cf["feedback"]:
        st.session_state.current_feedback["feedback"] = fb_choice.lower()

    if st.button("Submit Feedback"):
        st.session_state.feedback_list.append(st.session_state.current_feedback)
        st.success("✅ Feedback recorded in session.")
        del st.session_state.current_feedback  
# This is the code that gets the live news
st.markdown("---")
st.markdown("Analyze a Recent Political Headline")

headlines_df = fetch_political_headlines()
headlines = headlines_df["text"].tolist() if not headlines_df.empty else []

if "current_live_feedback" not in st.session_state:
    st.session_state.current_live_feedback = None

if headlines:
    with st.form("live_headline_form"):
        selected_headline = st.selectbox("Choose a headline", headlines)
        analyze_live = st.form_submit_button("Analyze Selected Headline")

    if analyze_live and selected_headline:
        cleaned = clean_text(selected_headline)
        vector = embed_text(cleaned, embeddings_index)
        prob = model.predict_proba([vector])[0]
        prediction = 'Fake' if prob[1] > confidence_threshold else 'Real'

        st.session_state.current_live_feedback = {
            "timestamp": datetime.now().isoformat(),
            "text": selected_headline,
            "model": selected_model_name,
            "prediction": prediction,
            "confidence_real": prob[0],
            "confidence_fake": prob[1],
            "threshold": confidence_threshold,
            "feedback": "agree"
        }

if st.session_state.current_live_feedback:
    cf = st.session_state.current_live_feedback

    st.markdown("Prediction for Selected Headline")
    st.markdown(f"**Model Used**: `{cf['model']}`")
    st.markdown(f"**Prediction**: {'🟥 Fake' if cf['prediction'] == 'Fake' else '🟩 Real'}")
    st.markdown(f"**Probability Real**: `{cf['confidence_real']:.3f}`")
    st.markdown(f"**Probability Fake**: `{cf['confidence_fake']:.3f}`")
    st.markdown(f"**Threshold**: {cf['threshold']:.2f}")

    st.markdown("**LIME Explanation (word impact):**")
    lime_exp = explain_with_lime(cf["text"], model, embeddings_index)
    for word, weight in lime_exp:
        color = "red" if weight > 0 else "green"
        st.markdown(f"<span style='color:{color};'>{word}: {weight:.3f}</span>", unsafe_allow_html=True)

    fb_choice = st.radio(
        "Do you agree with this prediction?",
        ["Agree", "Disagree"],
        index=0 if cf["feedback"] == "agree" else 1,
        key="feedback_radio_headline"
    )

    if fb_choice.lower() != cf["feedback"]:
        st.session_state.current_live_feedback["feedback"] = fb_choice.lower()

    if st.button("Submit Feedback for Headline"):
        st.session_state.feedback_list.append(st.session_state.current_live_feedback)
        st.success("✅ Feedback recorded in session.")
        del st.session_state.current_live_feedback
else:
    if not headlines:
        st.warning("No political headlines found.")

# Feedback store button
st.markdown("---")
st.markdown("Log Stored Feedback to File")

if st.session_state.feedback_list:
    st.write(f"{len(st.session_state.feedback_list)} feedback entries stored in session:")
    st.json(st.session_state.feedback_list)
    if st.button("Log All Stored Feedback"):
        for fb in st.session_state.feedback_list:
            log_feedback(fb)
        st.session_state.feedback_list = []
        st.success("🗑️ Cleared stored feedback after logging.")
else:
    st.info("No feedback stored in session. Use the Submit Feedback button above.")

# Created this to visualize yhour graph in 3d pca 
st.markdown("---")
st.markdown(" Visualize How the Model Separates Real vs Fake News (3D PCA)")

if st.button("Show 3D Plot of Training Data"):
    st.info("Loading precomputed 3D embeddings...")
    embeddings_file = "3d_embeddings.npz"
    data = np.load(embeddings_file)
    X_3d = data["X"]
    y = data["y"]

    df_plot = pd.DataFrame({
                "PC1": X_3d[:, 0],
                "PC2": X_3d[:, 1],
                "PC3": X_3d[:, 2],
                "Label": y
            })
    df_plot["LabelName"] = df_plot["Label"].map({0: "Real", 1: "Fake"})

    fig = px.scatter_3d(
            df_plot, x="PC1", y="PC2", z="PC3",
            color="LabelName",
            color_discrete_map={"Real": "blue", "Fake": "red"},
            title="3D PCA of News Embeddings",
            labels={"PC1": "PC 1", "PC2": "PC 2", "PC3": "PC 3"}
        )
    fig.update_layout(legend_title_text='Class')
    st.plotly_chart(fig, use_container_width=True)

# Loads what was previously stored in FakeNews.py so user can visualize the models
st.markdown("---")
st.markdown("Model Performance Visualizations")

model_file_map = {
    "Logistic Regression": {
        "acc": "metrics/logistic_accuracy.png",
        "cm": "metrics/logistic_confusion_matrix.png",
        "report": "metrics/logistic_classification_report.csv",
    },
    "Random Forest": {
        "acc": "metrics/random_forest_accuracy.png",
        "cm": "metrics/random_forest_confusion_matrix.png",
        "report": "metrics/random_forest_classification_report.csv",
    }
}

acc_path = model_file_map[selected_model_name]["acc"]
cm_path = model_file_map[selected_model_name]["cm"]
report_path = model_file_map[selected_model_name]["report"]

try:
    acc_img = Image.open(acc_path)
    st.image(acc_img, caption=f"{selected_model_name} Accuracy Bar Plot", use_container_width=True)
except:
    st.warning(f"Accuracy plot not found at {acc_path}")

try:
    cm_img = Image.open(cm_path)
    st.image(cm_img, caption=f"{selected_model_name} Confusion Matrix", use_container_width=True)
except:
    st.warning(f"Confusion matrix image not found at {cm_path}")

try:
    report_df = pd.read_csv(report_path)
    st.dataframe(report_df)
except:
    st.warning(f"Classification report CSV not found at {report_path}")
st.markdown("---")
st.markdown(" Retrain Model")

if st.button("Retrain Model"):
   st.info(
    "This is a placeholder for now, but the main idea is to use the feedback you provide—whether you agree or disagree with the predictions—"
    "to retrain and improve the model, making it better tailored to your needs."
)