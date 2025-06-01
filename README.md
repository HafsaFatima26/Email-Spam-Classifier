Email Spam Classifier - README
Project Overview
This project is a machine learning-based spam classifier that detects spam emails using NLP techniques. It includes a simple GUI for interaction and visual performance metrics such as confusion matrices and label distribution.
Features
- Cleaned and preprocessed dataset (email.csv)
- Text vectorization using CountVectorizer / TF-IDF
- Trained classification models (e.g., Naive Bayes, Logistic Regression)
- GUI interface for real-time spam prediction (spam_gui.py)
- Visualizations including confusion matrices and label distribution plots
Project Structure

ds_project/
├── email.csv                      # Labeled email dataset taken from kaggle
├── spam_gui.py                    # GUI for spam prediction
├── try5_guiver.py                 # Model training and grid search
├── confusion_matrices/           # Saved confusion matrix images
├── all_confusion_matrices_grid.png
├── label_distribution.png
├── README.docx                    # This file

How to Run
1. Clone the repository:
   git clone https://github.com/HafsaFatima26/Email-Spam-Classifier.git
   cd Email-Spam-Classifier
2. Install dependencies:
   pip install -r requirements.txt
3. Run the GUI:
   python spam_gui.py
Results
- Accuracy: ~XX% (based on best model)
- Precision/Recall available in confusion matrix images
- Visual plots for model evaluation
Tech Stack
- Python (Pandas, NumPy, scikit-learn, Matplotlib)
- Tkinter (for GUI)
- Seaborn (for plots)
Contributors
Hafsa Fatima,Syeda Haneesh,Waniya Syed
