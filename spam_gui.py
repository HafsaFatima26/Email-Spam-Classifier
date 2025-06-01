import sys
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy

nltk.download('stopwords')
nltk.download('wordnet')

# Load the saved model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def preprocess_email(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

def predict():
    msg = text_input.toPlainText()
    if not msg.strip():
        result_label.setText("Please enter a message!")
        result_label.setStyleSheet("color: red; font-size: 14px; font-weight: bold;")
        return
    processed = preprocess_email(msg)
    vect_msg = vectorizer.transform([processed])
    prediction = model.predict(vect_msg)[0]
    result = "Spam 🚫" if prediction == 1 else "Ham ✅"
    result_label.setText(f"Prediction: {result}")
    result_label.setStyleSheet("color: green; font-size: 18px; font-weight: bold;" if prediction == 0 else "color: red; font-size: 18px; font-weight: bold;")

# PyQt Application Setup
app = QApplication(sys.argv)

# Main Window
window = QWidget()
window.setWindowTitle("Email Spam Detector")
window.setFixedSize(500, 300)  # Fixed window size

# Layout Setup
layout = QVBoxLayout()

# Title Label
title_label = QLabel("Email Spam Detector")
title_label.setAlignment(Qt.AlignCenter)
title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #2a2a2a;")
layout.addWidget(title_label)

# Spacer between title and input box
layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

# Input Text Area
text_input = QTextEdit()
text_input.setPlaceholderText("Enter your email message here...")
text_input.setStyleSheet("""
    QTextEdit {
        font-size: 14px;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        background-color: #f9f9f9;
    }
""")
layout.addWidget(text_input)

# Spacer between input and button
layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

# Detect Spam Button
detect_button = QPushButton("Detect Spam")
detect_button.setStyleSheet("""
    QPushButton {
        font-size: 16px;
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
    }
    QPushButton:hover {
        background-color: #45a049;
    }
""")
detect_button.clicked.connect(predict)
layout.addWidget(detect_button)

# Spacer between button and output
layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

# Output Label
result_label = QLabel("")
result_label.setAlignment(Qt.AlignCenter)
result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2a2a2a;")
layout.addWidget(result_label)

# Set the layout and show the window
window.setLayout(layout)
window.show()

# Start the application
sys.exit(app.exec_())
