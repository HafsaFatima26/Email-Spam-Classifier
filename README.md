# Email Spam Classifier

## Project Overview
This project is a machine learning–based spam classifier that detects spam emails using NLP techniques. It features a simple GUI for interaction and visual performance metrics (confusion matrices and label distribution plots).

## Features
- Cleaned and preprocessed dataset (`email.csv`)
- Text vectorization using CountVectorizer and TF-IDF
- Trained classification models (e.g., Naive Bayes, Logistic Regression)
- GUI interface for real-time spam prediction (`spam_gui.py`)
- Visualizations including confusion matrices and label distribution plots

## Project Structure

<pre>
ds_project/
├── email.csv                      # Labeled email dataset taken from Kaggle
├── spam_gui.py                    # GUI for spam prediction
├── try5_guiver.py                 # Model training and grid search
├── confusion_matrices/           # Saved confusion matrix images
├── all_confusion_matrices_grid.png
├── label_distribution.png
├── README.docx                    # This file
</pre>



## How to Run
1. **Clone the repository:**  
   ```bash
   git clone https://github.com/HafsaFatima26/Email-Spam-Classifier.git
   cd Email-Spam-Classifier
2. **Install Dependencies**
   pip install -r requirements.txt
3. **Run the GUI**
    python spam_gui.py

## Results
- **Accuracy:** ~XX % (based on the best model)  
- **Precision / Recall:** Available in the saved confusion-matrix images  
- **Visual Plots:** Confusion matrices and label-distribution charts  

## Tech Stack
- **Python:** Pandas, NumPy, scikit-learn, Matplotlib  
- **Tkinter:** Graphical User Interface  
- **Seaborn:** Advanced plotting and visualization  

## Contributors
- Hafsa Fatima  
- Syeda Haneesh  
- Waniya Syed


