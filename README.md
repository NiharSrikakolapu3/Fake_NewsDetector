# Fake Political News Detection App

## Overview
This project is  an end-to-end Fake News Detection system focusing on political headlines and articles. It uses machine learning models—Logistic Regression and Random Forest—with GloVe word embeddings to classify news as **Real** or **Fake**. 

## Datasets
This project uses the following datasets for training and evaluation:

- **LIAR dataset:** A benchmark dataset for fake news detection focused on political statements.  
  [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)

- **Fake and Real News Dataset from Kaggle:** Contains labeled fake and real news articles, supplementing the training data.  
  [Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)


## Features
- **Two ML models:** Logistic Regression and Random Forest using pre-trained embeddings  
- **Live updates:** Fetches current political headlines through NewsAPI  
- **Explainability:** Word-level explanation of predictions using LIME  
- **Feedback:** Logs user feedback to help improve the model over time  
- **Visual insights:** PCA plots and accuracy/confusion matrix charts to understand model behavior  
- **Easy to run:** Built with Streamlit for quick and straightforward web deployment

## Workflow / How It Works
The app follows a structured workflow from raw data to predictions and explainability:

### 1. Data Loading
- Loads and merges multiple datasets: **LIAR** (political statements) and **Kaggle** (fake/real news headlines)
- Labels are standardized: **0 = Real, 1 = Fake**

### 2. Preprocessing
- Removes missing entries and cleans text by:
  - Converting to lowercase
  - Removing punctuation, digits, and extra whitespace
- Preprocessed text is stored in a new `clean_text` column

### 3. Embedding with GloVe
- Loads pre-trained **GloVe word embeddings** (`glove.6B.100d.txt`)
- Converts each headline into an averaged vector representation of its words
- Headlines without embeddings are represented as zero vectors

### 4. 3D Visualization Preparation
- Uses **PCA + LDA** to reduce embeddings to 3 dimensions
- Saves the 3D vectors for visual exploration of data clusters in the app

### 5. Model Training
- Trains **Logistic Regression** and **Random Forest** models
- Splits data into train/test sets (**80/20**) for evaluation
- Saves trained models as `model_logistic.pkl` and `model_random_forest.pkl`

### 6. Model Evaluation & Visualization
- Generates **accuracy bar charts**, **confusion matrices**, and **classification reports**
- Visualizations are saved to the `metrics/` folder for later reference

### 7. Prediction & Explainability
- User-input or live news headlines are cleaned and embedded
- Models predict **Real vs Fake** with confidence scores
- **LIME** provides word-level explanations to show which words influenced predictions

### 8. Feedback Logging
- User feedback (**Agree/Disagree**) is logged for retraining and model improvement

## Installation

1. Clone the repo:
    ```bash
    git clone git@github.com:NiharSrikakolapu3/Fake_NewsDetector.git
    cd Fake_NewsDetector
    ```

2. (Optional) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download GloVe embeddings** (these are large files):  
    ```bash
    cd data
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip glove.6B.100d.txt
    cd ..
    ```

5. Run the preprocessing/model script (Must Run prior to starting the app):
    ```bash
    cd src
    python FakeNews.py
    ```

## Usage

Start the app by running the following command:
```bash
streamlit run app.py

Usually can be accessed on  http://localhost:8501 but can be different so check your terminal

