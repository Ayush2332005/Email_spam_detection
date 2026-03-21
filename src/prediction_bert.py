from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle

# Load
tokenizer = BertTokenizer.from_pretrained("models/bert_model")
model = BertForSequenceClassification.from_pretrained("models/bert_model")

# Load encoder
le = pickle.load(open("models/label_encoder.pkl", "rb"))

# Input
text = input("Enter email text: ")

# Tokenize
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Predict
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
pred = torch.argmax(logits, dim=1).item()

print("Prediction:", le.inverse_transform([pred])[0])