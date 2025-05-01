import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
df = pd.read_csv('dream_dataset.csv')

# Prepare features and labels
X = df['dream']
y = df['label']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a model pipeline with TF-IDF vectorizer and Logistic Regression classifier
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Save the trained model
with open('dream_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as 'dream_model.pkl'")
