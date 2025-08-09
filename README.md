# Fake Political News Detection App

## Overview
This project is a simple and effective Fake News Detection system focusing on political headlines and articles. It uses machine learning models—Logistic Regression and Random Forest—with GloVe word embeddings to classify news as **Real** or **Fake**. The app is user-friendly and comes with:

- Analyzing user-input headlines or live political news
- Showing predictions with confidence scores
- Explaining predictions with LIME to make the model transparent
- Collecting user feedback for future improvements
- Visualizing model performance and data in interactive plots

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

4. **Download GloVe embeddings** (these are large files and are NOT stored in the repo):  
    ```bash
    mkdir -p data
    cd data
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
    cd ..
    ```

5. Run the preprocessing/model script (optional before using the app):
    ```bash
    python3 FakeNews.py
    ```

## Usage

Start the app by running:
```bash
streamlit run app.py