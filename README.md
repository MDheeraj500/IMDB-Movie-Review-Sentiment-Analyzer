# 🎬 IMDB Movie Review Sentiment Analyzer 🎥

The **IMDB Movie Review Sentiment Analyzer** is a full-stack application that utilizes Natural Language Processing (NLP) and Machine Learning (ML) techniques to classify user-submitted movie reviews as either positive or negative. This project showcases an end-to-end pipeline involving data preprocessing, feature extraction, sentiment model training, and user interaction through a visually appealing frontend.

## 🌟 Features

- **Real-Time Sentiment Analysis**: Processes user input and classifies the sentiment of each review with a pre-trained ML model.
- **Interactive, User-Friendly UI**: The frontend provides users with an immersive experience to interactively submit reviews and receive visual feedback on the sentiment.
- **Dynamic Genre Insights**: Users can explore movie genres and ratings in an intuitive, responsive carousel.
- **Immediate Feedback Mechanism**: Sentiment results feature expressive icons that reflect positive or negative sentiments based on the analysis.

## 🚀 Quickstart

To run the IMDB Movie Review Sentiment Analyzer locally, please follow the steps below.

### Prerequisites
- Remember to check the requirements and feel free to use the most updated versions which are compatible
- **Python 3.x**
- **Node.js**
- **npm**

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/IMDB-Movie-Review-Sentiment-Analyzer.git
   cd IMDB-Movie-Review-Sentiment-Analyzer
   ```
2. **Backend Setup:**
   ```bash
   cd Backend
   python3 -m venv venv
   source venv/bin/activate # For Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend Setup:**
  ```bash
  cd ../Frontend
  npm install
  ```
4. **Run the Application**
   - **Backend**: Start the Flask server
     ```bash
       python main.py
     ```
   - **Frontend**: Start the React app
     ```bash
       npm run dev
     ```
5. Open your browser and navigate to localhost to experience the IMDB Movie Review Sentiment Analyzer.

## 📁 Project Structure

```plaintext
IMDB-Movie-Review-Sentiment-Analyzer/
├── Backend/                 # Contains Flask API and ML model
│   ├── main.py              # Main backend code
│   ├── requirements.txt     # Backend dependencies
│   ├── IMDB_review_predictor.pkl  # Pre-trained sentiment analysis model
│   └── tfidfvect.pkl        # TFIDF vectorizer
└── Frontend/                # Contains React app files
    ├── src/                 # Main source directory for React components
    │   ├── App.jsx          # Homepage layout
    │   ├── Prediction.jsx   # Sentiment prediction page
    │   └── assets/          # Images and icons
    └── public/              # Static files and assets
```  

## 💻 Machine Learning and NLP Workflow

This project implements a comprehensive NLP pipeline designed to preprocess and analyze natural language data, focusing on sentiment analysis for movie reviews. Below is an overview of the machine learning and NLP techniques used:

### 💻 NLP Techniques in Sentiment Analysis

This project leverages various Natural Language Processing (NLP) techniques to create a robust sentiment analysis pipeline. Below is a summary of the core methods used:

1. **Data Preprocessing**
   - **Tokenization and Lemmatization:** Using NLTK, reviews are split into individual tokens and reduced to their base forms. This normalization helps generalize words (e.g., "running" to "run") and enhances model accuracy.
   - **Contraction Expansion:** Informal contractions are expanded (e.g., "won't" to "will not") to provide clearer context, crucial for analyzing sentiment in informal text.
   - **Stopword Removal:** High-frequency but low-meaning words are filtered out, allowing the model to focus on sentiment-bearing terms.

2. **Feature Extraction**
   - **TF-IDF Vectorization:** Text data is converted into numerical features based on term frequency and inverse document frequency, capturing word importance across the dataset.
   - **N-gram Approach (Unigrams, Bigrams, Trigrams):** This project uses n-grams to capture sequences of one, two, and three words. This technique is particularly useful in detecting sentiment nuances and identifying sarcastic phrases (e.g., "oh great, just what I needed").

3. **Sentiment Classification and Model Selection**
   - **Logistic Regression, Decision Trees, and Random Forests:** These models are trained on n-gram features to classify sentiment effectively. Logistic Regression provides interpretable feature importance, highlighting specific words and phrases that indicate sentiment.
   - **AdaBoost Ensemble Learning:** An ensemble of weak learners is combined to create a more robust model, enhancing the classifier’s ability to handle complex language patterns in reviews.

4. **Model Evaluation**
   - **Metrics (Precision, Recall, F1 Score, and AUC-ROC):** These metrics offer a comprehensive performance view, especially important in sentiment analysis where language nuances, including sarcasm, play a significant role.

5. **Deployment Preparation**
   - **Pickling Model and Vectorizer:** The TF-IDF vectorizer and trained model are serialized with Pickle for efficient loading and real-time prediction in the Flask API.

This combination of preprocessing, feature extraction, and model selection provides a powerful foundation for analyzing movie review sentiments, capable of capturing language subtleties and handling sarcastic expressions.


## 🛠️ Technologies Used

- **Backend**: Python, Flask, NLTK, Scikit-Learn, Pickle
- **Frontend**: React, React Router, PapaParse, Slick Carousel
- **Styling**: CSS with dark theme customization


## 💬 Feedback

Feel free to open an issue or contact me directly at mdheeraj2000@gmail.com
