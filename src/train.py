import pandas as pd
import pickle
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from preprocess import clean_text

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/enron_spam_data.csv")

# Combine text
df["text"] = df["Subject"].fillna('') + " " + df["Message"].fillna('')

# Clean labels
df["Spam/Ham"] = df["Spam/Ham"].str.lower().str.strip()
df["label"] = df["Spam/Ham"].map({"spam":1, "ham":0})

df = df.dropna()

# Clean text
df["clean_text"] = df["text"].apply(clean_text)

# Split
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

# Accuracy
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))