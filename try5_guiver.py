#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import string
import re
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
nltk.download('stopwords')
nltk.download('wordnet')

#loading and cleaning dataset
dataset = pd.read_csv('email.csv')
print("Dataset loaded successfully!")
print(dataset.head())

#handling missing data
dataset = dataset[dataset['Category'].isin(['ham', 'spam'])]
dataset = dataset.dropna(subset=['Category', 'Message'])

#class distribution plot
sns.set_style("whitegrid")
sns.countplot(x='Category', data=dataset, palette='pastel')
plt.xlabel("Email Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Distribution of Spam vs Ham Emails", fontsize=14, color='purple')
plt.savefig("label_distribution.png", bbox_inches='tight')
plt.show()

#email preprocessing
def preprocess_email(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

dataset['Message'] = dataset['Message'].apply(preprocess_email)

print("\nSample after preprocessing:")
print(dataset['Message'].head())

#preparing features
X = dataset['Message']
y = dataset['Category'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

#TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF Vectorization completed!")

#model definition
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine (Linear)": SVC(kernel='linear'),
    "Gradient Boosting": GradientBoostingClassifier()  # not in paper
}

os.makedirs("confusion_matrices", exist_ok=True)

#training data and confusion matrix plotting
rows = 3
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
plt.subplots_adjust(hspace=0.6, wspace=0.4)

for idx, (model_name, model) in enumerate(models.items()):
    print("\n" + "=" * 60)
    print(f"Training and Evaluating: {model_name}")

    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    # Print confusion matrix in terminal
    print(f"\nConfusion Matrix for {model_name}:")
    print(cm)

    # Save individual confusion matrix
    plt_cm = plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', square=True, cbar=False,
                annot_kws={"size": 12})
    plt.title(f"🧾 Confusion Matrix - {model_name}", fontsize=13)
    plt.xlabel("Predicted", fontsize=11)
    plt.ylabel("Actual", fontsize=11)
    cm_path = f"confusion_matrices/{model_name.replace(' ', '_')}.png"
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close(plt_cm)

    # Plot on combined grid
    ax = axes[idx // cols][idx % cols]
    sns.heatmap(cm, annot=True, fmt='d', cmap='rocket_r', square=True, ax=ax,
                cbar=False, linewidths=1.0, linecolor='gray',
                annot_kws={"size": 11})
    ax.set_title(model_name, fontsize=13, weight='bold', pad=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.tick_params(axis='both', labelsize=10)

# Remove empty subplots
total_models = len(models)
for idx in range(total_models, rows * cols):
    fig.delaxes(axes.flatten()[idx])

plt.savefig("all_confusion_matrices_grid.png", bbox_inches='tight')
plt.show()

print("\nAll models evaluated successfully")
print("Individual confusion matrices saved in: confusion_matrices/")
print("Combined grid image saved as: all_confusion_matrices_grid.png")
