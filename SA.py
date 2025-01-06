import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already available
nltk.download('stopwords')

# Load the dataset
dataset_path = "C:/Users/Venka/Downloads/training.1600000.processed.noemoticon.csv/training.1600000.processed.noemoticon.csv.csv"
columns = ["sentiment", "id", "date", "query", "user", "text"]
data = pd.read_csv(dataset_path, encoding="latin-1", names=columns)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

# Apply preprocessing
data["text"] = data["text"].apply(preprocess_text)

# Map sentiment to binary (positive: 4 -> 1, negative: 0 -> 0)
data = data[data["sentiment"].isin([0, 4])]  # Filter out rows with irrelevant sentiments
data["sentiment"] = data["sentiment"].map({0: 0, 4: 1})

# Splitting the dataset
X = data["text"]
y = data["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Sentiment prediction function
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_vect = vectorizer.transform([processed_text])
    prediction = model.predict(text_vect)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment

# Example usage
sample_text = input("Enter a text to analyze sentiment: ")
print(f"The sentiment is: {predict_sentiment(sample_text)}")
