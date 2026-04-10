import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("drug_interactions.csv")

X = df["Description"]
y = df["Severity"]

# TF-IDF
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_vector, y)

# Feature importance
feature_names = vectorizer.get_feature_names_out()
importances = model.feature_importances_

# Top 10 features
indices = np.argsort(importances)[-10:]

plt.figure()
sns.heatmap(importances[indices].reshape(1, -1),
            annot=True,
            cmap="coolwarm",
            xticklabels=feature_names[indices])

plt.title("Top Feature Importance Heatmap")
plt.yticks([])

plt.show()