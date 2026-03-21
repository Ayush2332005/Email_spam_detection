import pickle

# Load
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
le = pickle.load(open("models/label_encoder.pkl", "rb"))

# Input
text = input("Enter email text: ")

# Transform
X = vectorizer.transform([text])

# Predict
pred = model.predict(X)[0]

print("Prediction:", le.inverse_transform([pred])[0])