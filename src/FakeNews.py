import os
import re
import string
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from lime.lime_text import LimeTextExplainer

# ----------- Text Preprocessing -----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_data(df):
    df = df.dropna(subset=["text"])
    df["clean_text"] = df["text"].apply(clean_text)
    return df

# ----------- Load GloVe embeddings -----------
def load_glove_embeddings(glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, encoding="utf8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings_index[word] = vector
    print(f"Loaded {len(embeddings_index)} word vectors from GloVe.")
    return embeddings_index

def embed_text(text, embeddings_index, embedding_dim=100):
    words = text.split()
    valid_vectors = [embeddings_index[word] for word in words if word in embeddings_index]
    if not valid_vectors:
        return np.zeros(embedding_dim)
    return np.mean(valid_vectors, axis=0)

# ----------- Data Loaders -----------
def load_and_label_data(real_path, fake_path):
    try:
        real_df = pd.read_csv(real_path)
        fake_df = pd.read_csv(fake_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        return pd.DataFrame(columns=["text", "label"])

    real_df["text"] = real_df.get("title", real_df.iloc[:, 0])
    fake_df["text"] = fake_df.get("title", fake_df.iloc[:, 0])
    real_df["label"] = 0
    fake_df["label"] = 1

    return pd.concat([real_df[["text", "label"]], fake_df[["text", "label"]]], ignore_index=True)

def load_liar_data():
    liar_files = ['../data/train.tsv', '../data/valid.tsv', '../data/test.tsv']
    liar_data = []

    for path in liar_files:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        try:
            df = pd.read_csv(path, sep='\t', header=None)
            if len(df.columns) > 2:
                df = df[[1, 2]]
                df.columns = ['label', 'text']
                df = df[df['label'].isin(['true', 'false'])]
                df['label'] = df['label'].map({'true': 0, 'false': 1})
                liar_data.append(df)
        except Exception as e:
            print(f"Error reading LIAR file {path}: {e}")

    return pd.concat(liar_data, ignore_index=True) if liar_data else pd.DataFrame(columns=["text", "label"])

# ----------- Model Functions -----------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model

def save_model(model, model_path="model_glove.pkl"):
    joblib.dump(model, model_path)

def load_model(model_path="model_glove.pkl"):
    return joblib.load(model_path)

def predict_news(text, model, embeddings_index, threshold=0.6):
    cleaned = clean_text(text)
    vector = embed_text(cleaned, embeddings_index)
    prob = model.predict_proba([vector])[0]
    prediction = 1 if prob[1] > threshold else 0
    return "Fake" if prediction == 1 else "Real"

# ----------- LIME Explainer Functions -----------
def predict_proba_lime(texts, model, embeddings_index):
    # LIME expects a list of texts and returns probability predictions
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
    # Show explanation in notebook or print to console
    print("LIME explanation (word contributions):")
    for word, weight in explanation.as_list():
        print(f"{word}: {weight}")
    # If running in Jupyter Notebook, you can uncomment below:
    # explanation.show_in_notebook(text=True)

# ----------- Main Execution -----------
if __name__ == "__main__":
    kaggle_real_path = "../data/True.csv"
    kaggle_fake_path = "../data/Fake.csv"

    print("Loading datasets...")
    kaggle_data = load_and_label_data(kaggle_real_path, kaggle_fake_path)
    liar_data = load_liar_data()

    print("Merging datasets...")
    combined = pd.concat([kaggle_data, liar_data], ignore_index=True)
    print("Class distribution:", Counter(combined["label"]))

    print("Preprocessing...")
    combined = preprocess_data(combined)

    print("Loading GloVe embeddings...")
    glove_path = "../data/glove.6B.100d.txt"
    embeddings_index = load_glove_embeddings(glove_path)

    print("Creating document embeddings...")
    X = np.vstack(combined["clean_text"].apply(lambda x: embed_text(x, embeddings_index)))
    y = combined["label"].values

    print("Training model...")
    model = train_model(X, y)

    print("Saving model...")
    save_model(model)

    print("\nRunning prediction and LIME explanation sample:")
    sample = "Aliens have landed in New York, claims anonymous source"
    result = predict_news(sample, model, embeddings_index)
    print(f"Sample Prediction: {result}")

    # Show LIME explanation (simple word-level contribution)
    explain_with_lime(sample, model, embeddings_index)
