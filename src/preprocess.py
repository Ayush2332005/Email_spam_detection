import pandas as pd

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


def load_data(path):
    df = pd.read_csv(path)

    # Combine Subject + Message
    df["text"] = df["Subject"].fillna('') + " " + df["Message"].fillna('')

    # Assign category
    df["category"] = df["text"].apply(assign_category)

    return df