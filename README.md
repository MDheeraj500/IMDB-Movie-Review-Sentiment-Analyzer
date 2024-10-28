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
  ``
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

### Data Preprocessing

- **Tokenization and Lemmatization**: Using NLTK, raw text is split into tokens and converted to their base forms, reducing inflections and improving model accuracy.
- **Contraction Expansion**: Expands common contractions to full forms, enhancing the model's ability to interpret informal text.
- **Stopword Removal**: Removes common but insignificant words while preserving essential semantic content for analysis.

### Feature Extraction

- **TF-IDF Vectorization**: Converts text into numerical representations that highlight the importance of words across the dataset while down-weighting commonly used words.
- **Count Vectorizer (Unigram, Bigram, Trigram)**: Extracts top unigrams, bigrams, and trigrams as features to enhance model training.

### Model Training and Selection

- **Logistic Regression**: A linear model that classifies reviews by analyzing feature importance and predicting the likelihood of positive or negative sentiment.
- **Decision Tree and Random Forest Classifiers**: Used to evaluate feature importance, with hyperparameter tuning to mitigate overfitting.
- **AdaBoost Classifier**: Aggregates weak learners to enhance prediction stability and accuracy.

### Model Evaluation

The trained models are evaluated on **precision**, **recall**, **F1 score**, and **AUC-ROC** score, providing robust performance insights. Fine-tuning and feature selection are employed to achieve optimal accuracy.

### Deployment

The final model is serialized using **Pickle** for efficient loading and real-time prediction in the Flask backend.

## 🛠️ Technologies Used

- **Backend**: Python, Flask, NLTK, Scikit-Learn, Pickle
- **Frontend**: React, React Router, PapaParse, Slick Carousel
- **Styling**: CSS with dark theme customization


## 💬 Feedback

Feel free to open an issue or contact me directly at mdheeraj2000@gmail.com
