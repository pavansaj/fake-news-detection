import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    'text': [
        "Breaking news: Earthquake in city",
        "You won a lottery of $1 million",
        "Government passes new law",
        "Click here to earn money fast"
    ],
    'label': [1, 0, 1, 0]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.25, random_state=42
)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

def predict_news(news):
    news_vec = vectorizer.transform([news])
    prediction = model.predict(news_vec)
    return "Real News" if prediction[0] == 1 else "Fake News"

print(predict_news("Breaking: New policy announced"))
