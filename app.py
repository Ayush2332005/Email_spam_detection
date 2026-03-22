from src.predict import predict_email
from src.predict_bert import predict_bert

print("Choose Model:")
print("1. TF-IDF")
print("2. BERT")

choice = input("Enter choice: ")

text = input("Enter Email: ")

if choice == "1":
    print("Prediction:", predict_email(text))
else:
    print("Prediction:", predict_bert(text))