import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import sys 

print("Checking for NLTK data files...")
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    print("NLTK data files found.")
except LookupError:
    print("NLTK data files not found. Attempting to download them automatically...")
    try:
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('punkt')
        print("NLTK data downloaded successfully.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        print("Please check your internet connection and try again.")
        sys.exit(1) 

try:
    train_df = pd.read_csv('train.csv', header=None, names=['polarity', 'title', 'text'], nrows=100)
    test_df = pd.read_csv('test.csv', header=None, names=['polarity', 'title', 'text'], nrows=100)
    print("Datasets loaded successfully.")
    print("Training data shape:", train_df.shape)
    print("Testing data shape:", test_df.shape)
except FileNotFoundError:
    print("Error: train.csv or test.csv not found. Please make sure the files are in the same directory as this script.")
    sys.exit(1) 
print("\nFirst 5 rows of the training data:")
print(train_df.head())

def clean_text(text):
    """
    Cleans and preprocesses the text for sentiment analysis.
    This function handles:
    1. Converting text to lowercase.
    2. Handling escaped characters (e.g., '""', '\n').
    3. Removing non-alphabetic characters and numbers.
    4. Tokenizing words.
    5. Removing English stopwords.
    6. Lemmatizing words to their base form.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = text.replace('""', '"').replace('\\n', ' ')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]

    return ' '.join(tokens)

train_df['full_review'] = train_df['title'].astype(str) + ' ' + train_df['text'].astype(str)
test_df['full_review'] = test_df['title'].astype(str) + ' ' + test_df['text'].astype(str)

print("\nStarting text cleaning on training data...")
train_df['cleaned_review'] = train_df['full_review'].apply(clean_text)
print("Text cleaning on training data complete.")

print("\nStarting text cleaning on testing data...")
test_df['cleaned_review'] = test_df['full_review'].apply(clean_text)
print("Text cleaning on testing data complete.")

print("\nFirst 5 rows of cleaned data:")
print(train_df[['full_review', 'cleaned_review', 'polarity']].head())

X_train = train_df['cleaned_review']
y_train = train_df['polarity']
X_test = test_df['cleaned_review']
y_test = test_df['polarity']

print("\nCreating TF-IDF vectors...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("TF-IDF vectorization complete.")

print("Training the Multinomial Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train_vec, y_train)
print("Model training complete.")

print("\nMaking predictions on the test data...")
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

def predict_sentiment(text, model, vectorizer):
    cleaned_text = clean_text(text)
    text_vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vec)
    sentiment = "Positive" if prediction[0] == 2 else "Negative"
    return sentiment

new_review_1 = "This product is absolutely amazing! I love it and would highly recommend it."
prediction_1 = predict_sentiment(new_review_1, model, vectorizer)
print(f'\nReview: "{new_review_1}" -> Predicted Sentiment: {prediction_1}')

new_review_2 = "The quality is terrible and it broke after only one use. I am very disappointed."
prediction_2 = predict_sentiment(new_review_2, model, vectorizer)
print(f'Review: "{new_review_2}" -> Predicted Sentiment: {prediction_2}')
