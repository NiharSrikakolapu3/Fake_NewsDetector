# Fake Political News Detection App

## Overview
This project is  an end-to-end Fake News Detection system focusing on political headlines and articles. It uses machine learning models—Logistic Regression and Random Forest—with GloVe word embeddings to classify news as **Real** or **Fake**. The app is user-friendly and comes with:

- Analyzing user-input headlines or live political news
- Showing predictions with confidence scores
- Explaining predictions with LIME to make the model transparent
- Collecting user feedback for future improvements
- Visualizing model performance and data in interactive plots

## Datasets
This project uses the following datasets for training and evaluation:

- **LIAR dataset:** A benchmark dataset for fake news detection focused on political statements.  
  [LIAR Dataset info can be found at](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)

- **Fake and Real News Dataset from Kaggle:** Contains labeled fake and real news articles, supplementing the training data.  
  [Kaggle Dataset info can be found at ](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)


## Features
- **Two ML models:** Logistic Regression and Random Forest using pre-trained embeddings  
- **Live updates:** Fetches current political headlines through NewsAPI  
- **Explainability:** Word-level explanation of predictions using LIME  
- **Feedback:** Logs user feedback to help improve the model over time  
- **Visual insights:** PCA plots and accuracy/confusion matrix charts to understand model behavior  
- **Easy to run:** Built with Streamlit for quick and straightforward web deployment  

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
    unzip glove.6B.zip
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

