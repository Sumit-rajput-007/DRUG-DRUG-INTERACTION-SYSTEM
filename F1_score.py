import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Load dataset
df = pd.read_csv("drug_interactions.csv")

X = df["Description"]
y = df["Severity"]

# TF-IDF
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vector, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# F1 Score
f1 = f1_score(y_test, y_pred, average='weighted')

print("F1 Score:", f1)