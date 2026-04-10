import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load data
df = pd.read_csv("drug_interactions.csv")

X = df["Description"]
y = df["Severity"]

# Convert labels
classes = y.unique()

# TF-IDF
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_vector, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Probabilities
y_score = model.predict_proba(X_test)

# Binarize labels
y_test_bin = label_binarize(y_test, classes=classes)

# ROC Plot
plt.figure()

for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{classes[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()