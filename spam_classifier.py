import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam_dataset.csv")

# Features and labels
X = df['text']
y = df['label']

# Convert text to vectors
cv = CountVectorizer()
X = cv.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Testing with user input
email = input("Enter email text: ")
email_vector = cv.transform([email]).toarray()
prediction = model.predict(email_vector)

print("\nPrediction:", "Spam" if prediction[0] == 1 else "Not Spam")
