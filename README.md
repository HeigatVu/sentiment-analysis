# Sentiment Analysis on IMDB Movie Reviews
This project implements and evaluates classical machine learning models for sentiment analysis on the IMDB movie review dataset. The goal is to classify movie reviews as either 'positive' or 'negative' based on their text content.

## Project Overview
This repository provides a step-by-step implementation of a text classification pipeline, including:

  - Data Loading and Cleaning: Ingesting the dataset and removing duplicate entries.

  - Exploratory Data Analysis (EDA): Analyzing the distribution of sentiment labels and the length of reviews.

  - Text Preprocessing: A comprehensive workflow to clean and normalize the raw text data.

  - Feature Engineering: Converting text into numerical vectors using TF-IDF.

  - Model Training and Evaluation: Building, training, and evaluating Decision Tree and Random Forest models.

## Dataset
The project uses the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). This dataset contains 50,000 reviews, evenly split between positive and negative sentiments.

## Methodology
The project follows a structured machine learning pipeline:

1. Preprocessing & EDA: 
    - Removed HTML tags, punctuation, numbers, and emojis.
  
    - Expanded contractions and converted text to lowercase.
  
    - Performed lemmatization and removed English stopwords.
  
    - Visualized the class balance and distribution of word counts per review.

2. Text Representation:

    - Used `TfidfVectorizer` from Scikit-learn to convert preprocessed text into a matrix of TF-IDF features. A vocabulary size of 10,000 features was used.

3. Modeling and Evaluation:

    - The data was split into an 80% training set and a 20% testing set.

    - Two models were trained and compared:

        - `DecisionTreeClassifier`
        
        - `RandomForestClassifier`

    - Model performance was evaluated using the accuracy metric.

## Technology Stack
  - Data Manipulation & Analysis: Pandas, NumPy

  - Text Processing: NLTK, BeautifulSoup4, contractions

  - Machine Learning: Scikit-learn

  - Data Visualization: Matplotlib, Seaborn

## How to Run

1. Clone the repository:
```
git clone https://github.com/HeigatVu/sentiment-analysis.git
cd sentiment-analysis
```

2. Set up the environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install pandas notebook nltk scikit-learn matplotlib seaborn beautifulsoup4 contractions
```

4. Download NLTK data:
Run the following Python script to download the necessary NLTK packages:
```
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

5. Run the notebooks:
Launch Jupyter Notebook and run the notebooks in the following order for the best experience:
   1. `preprocessing.ipynb`
   2. `EDA.ipynb`
   3. `model.ipynb`
