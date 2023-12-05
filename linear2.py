import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, learning_rate=0.05, epochs=1000):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]  
    # 편향(bias)을 위해 X에 1 추가
    W = np.random.rand(n + 1)
    b = np.random.rand()

    for i in range(epochs):
        z = np.dot(X, W) + b
        h = sigmoid(z)
        error = np.abs(h - y).mean()

        if error < 0.001:
            break

        W_grad = learning_rate * np.dot((h - y), X) / m
        b_grad = learning_rate * (h - y).mean()

        W = W - W_grad
        b = b - b_grad

    return W, b

def predict(X, W, b):
    X = np.c_[np.ones((X.shape[0], 1)), X]
    predictions = sigmoid(np.dot(X, W) + b)
    return predictions

data = pd.read_csv("weatherReports.csv")

selected_events = ['Thunderstorm Wind', 'Hail']
selected_data = data.loc[data['event_type'].isin(selected_events), ['event_narrative', 'event_type']].copy()

selected_data['event_narrative'].fillna("", inplace=True)

X = selected_data['event_narrative']
y = (selected_data['event_type'] == 'Thunderstorm Wind').astype(int)  # 이진 분류

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

W, b = gradient_descent(X_train.toarray(), y_train)

predictions = predict(X_test.toarray(), W, b)
y_pred = (predictions >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")