import os
import re
import string
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import joblib
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import seaborn as sns

# I created the following 2 functions to preprocess data
def clean_text(text):
    # Converted the string to lower case, got rid of punctuation, digits, and trailing white space
    try:
        text = str(text).lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except:
        return " "

def preprocess_data(df):
    # Uses panda to get rid of na rows in text, then creates a new collumn in the df called clean text which uses the above function,
    # and then returns the new df for further functions
    df = df.dropna(subset=["text"])
    df["clean_text"] = df["text"].apply(clean_text)
    return df

# Takes the glove file,reads it, splits the lines
def load_glove_embeddings(glove_file_path):
    # So when looking at the glove file it looks something like this- the 0.418 0.24968 -0.41242 0.1217 0.34527 ... 0.1238
    glove_model = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # I want to seperate the line
            split_line = line.split()
            # Store the first word of the line this will be the key
            word = split_line[0]
            # everything after the first word will be be stored as embeddings 
            embedding = np.array(split_line[1:], dtype=np.float64)
            # Storing it as a key value pair
            glove_model[word] = embedding
        # Just printing how many total words i got from the glove file
    print(f"Total words loaded: {len(glove_model)}")
    return glove_model

# The main goal of this is to compare your text, to the glove embedding and give you an average of the words in the sentence
def embed_text(text, embeddings_index, embedding_dim=100):
    words = text.split()
    valid_vectors = []
    for word in words:
        if word in embeddings_index:
            valid_vectors.append(embeddings_index[word])
    if not valid_vectors:
        # Returns a vector of 0's if it isnt found
        return np.zeros(embedding_dim)
    return np.mean(valid_vectors, axis=0)

# Main goal of these 2 functions is to load data from the liar dataset and kaggle dataset mainly used for training the models
# Stack the dfs using pandas to create one df so its easier with text and label columns only because for me those are the only ones important
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

# The following functions were created for visualization purpouses so I want people to see the differences between the 2 models that I created
def visualize_model_performance(y_test, y_pred, model_name, output_dir="metrics"):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_confusion_matrix.png")
    plt.close()

    # Classification Report CSV
    report = classification_report(y_test, y_pred, target_names=["Real", "Fake"], output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f"{output_dir}/{model_name}_classification_report.csv")

    # Accuracy Bar Plot
    acc = accuracy_score(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    plt.bar([model_name], [acc], color='green')
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy: {acc:.2f}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_accuracy.png")
    plt.close()

# The following functions were actually created to make the models which can later be interacted with when Im actually creating predictios
def train_model(X, y, model_type='logistic'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Invalid model type. Use 'logistic' or 'random_forest'.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    visualize_model_performance(y_test, y_pred, model_type)

    return model

def save_model(model, model_path):
    joblib.dump(model, model_path)

def load_model(model_path):
    return joblib.load(model_path)

def load_models():
    logistic_model = load_model("model_logistic.pkl")
    rf_model = load_model("model_random_forest.pkl")
    return logistic_model, rf_model


# This is the main function to use preprocessing and predict the user input news
def predict_news(text, model, embeddings_index, threshold=0.6):
    # Step 1: Clean the input text (remove noise, lowercase, etc.)
    cleaned_text = clean_text(text)

    # Step 2: Convert the cleaned text into a numerical vector using word embeddings
    embedded_vector = embed_text(cleaned_text, embeddings_index)

    # Step 3: Get the predicted probabilities from the model
    # The model returns probabilities for both classes: [Real, Fake]
    probabilities = model.predict_proba([embedded_vector])[0]
    fake_prob = probabilities[1]  # Probability that the news is fake

    # Step 4: Compare with threshold to classify as Fake or Real
    if fake_prob > threshold:
        return "Fake"
    else:
        return "Real"


# Using lime to figure out how words influenced my model
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

# ----------- PCA + LDA 3D Embedding Export Function ----------- #
def save_3d_embeddings_with_lda(X, y, output_path="3d_embeddings.npz"):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    X_pca_scaled = MinMaxScaler((0, 1)).fit_transform(X_pca)

    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X_scaled, y)
    X_lda_scaled = MinMaxScaler((0, 1)).fit_transform(X_lda)

    X_3d = np.hstack((X_pca_scaled, X_lda_scaled))
    np.savez(output_path, X=X_3d, y=y)
    print(f"3D PCA+LDA embeddings saved to {output_path}")

# ----------- Main Execution ----------- #
if __name__ == "__main__":
    kaggle_real_path = "../data/trueOne.csv"
    kaggle_fake_path = "../data/fakeOne.csv"

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

    save_3d_embeddings_with_lda(X, y)

    print("\nTraining Logistic Regression Model...")
    logistic_model = train_model(X, y, model_type='logistic')
    save_model(logistic_model, "model_logistic.pkl")

    print("\nTraining Random Forest Model...")
    rf_model = train_model(X, y, model_type='random_forest')
    save_model(rf_model, "model_random_forest.pkl")

    print("\nRunning prediction and LIME explanation sample (Logistic Regression):")
    sample = "Aliens have landed in New York, claims anonymous source"
    result = predict_news(sample, logistic_model, embeddings_index)
    print(f"Sample Prediction (Logistic): {result}")
    explain_with_lime(sample, logistic_model, embeddings_index)