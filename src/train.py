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
# Load dataset
df = pd.read_csv("data/enron_spam_data.csv")

# Combine Subject + Message into one column FIRST
df["text"] = df["Subject"].fillna('') + " " + df["Message"].fillna('')

# Now apply category
def assign_category(text):
    text = str(text).lower()

    if any(word in text for word in ["offer","win","free","prize","money"]):
        return "Spam"
    elif any(word in text for word in ["meeting","project","client","report"]):
        return "Work"
    elif any(word in text for word in ["deadline","urgent","asap"]):
        return "Important"
    else:
        return "Personal"

df["category"] = df["text"].apply(assign_category)
# Combine Subject + Message into one column
df["text"] = df["Subject"].fillna('') + " " + df["Message"].fillna('')

df["category"] = df["text"].apply(assign_category)

# Combine text
df["text"] = df["Subject"].fillna('') + " " + df["Message"].fillna('')

# Clean labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["label"] = le.fit_transform(df["category"])

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
import pickle
pickle.dump(le, open("models/label_encoder.pkl", "wb"))

pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

# Accuracy
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))