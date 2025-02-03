Disease Prediction System

📌 Introduction

The Disease Prediction System is a Machine Learning (ML)-based application that helps in predicting potential diseases based on the selected symptoms. The system uses a Random Forest Classifier trained on a medical dataset, which contains different diseases and their associated symptoms.

This project provides a Graphical User Interface (GUI) built using Tkinter, enabling users to conveniently input their symptoms and receive disease predictions in real-time.

🎯 Objectives

Develop an ML-based model that predicts diseases based on user-input symptoms.

Build an interactive GUI for easy usage.

Implement cross-validation to ensure accurate predictions.

Make the system scalable and flexible for future improvements.

🚀 Features

✅ Machine Learning Model: Uses Random Forest Classifier for accurate predictions.✅ Graphical User Interface (GUI): Built using Tkinter with an enhanced, scrollable symptom selection panel.✅ Symptom Selection Panel: Users can choose multiple symptoms from a well-organized, scrollable list.✅ Submit Button: Click to process selected symptoms and display predicted disease.✅ Popup Alert for Prediction: The predicted disease is shown in a message box.✅ Cross-Validation: Uses Stratified K-Fold Cross-Validation for better model evaluation.✅ Easy-to-Use: No deep technical knowledge required to operate.

🏗 Tech Stack

Python - Core programming language

Scikit-Learn - ML library for model training and prediction

Pandas & NumPy - Data manipulation and preprocessing

Tkinter - GUI development

📂 Project Structure

Disease-Prediction-System/
│── data/                     # Dataset Folder
│   ├── Training.csv           # Training dataset
│── src/                       # Source Code
│   ├── main.py                # Main program file
│── README.md                  # Project documentation
│── requirements.txt           # Required dependencies

📥 Installation

🔹 Prerequisites

Make sure you have Python 3.7+ installed.

🔹 Install Dependencies

Run the following command in the terminal:

pip install pandas numpy scikit-learn tk

🏃‍♂️ Usage

🔹 Step 1: Clone the Repository

git clone https://github.com/your-username/disease-prediction.git
cd disease-prediction

🔹 Step 2: Run the Application

python src/main.py

📝 How It Works

User selects symptoms from a list.

Clicks the Submit button to process symptoms.

The system applies a Machine Learning model to predict the disease.

The predicted disease is displayed in a popup box.

🎨 GUI Interface

The Disease Prediction System features an enhanced Tkinter-based GUI:

Framed Layout: Symptoms appear inside a scrollable frame.

Scrollable Symptom List: Users can scroll through symptoms.

Styled Submit Button: Large, clear Submit button.

Popup Alert: Displays the predicted disease.

🏆 Machine Learning Model

✅ Data Preprocessing

Dataset: The model is trained on a structured medical dataset containing diseases and their corresponding symptoms.

Feature Selection: Symptoms are used as input features, while the disease name is the output label.

Label Encoding: Disease names are converted to numeric values using LabelEncoder.

✅ Model Training

The system utilizes a Random Forest Classifier:

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

✅ Cross-Validation

To ensure robust performance, the model is evaluated using Stratified K-Fold Cross-Validation:

from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"Mean Accuracy: {np.mean(scores):.4f}")

⚙ Future Enhancements

🔹 Additional ML Models: Implement models like SVM, XGBoost for comparison.🔹 Web-Based UI: Convert Tkinter GUI into a Flask or Streamlit Web App.🔹 Larger Dataset: Improve accuracy by adding more symptoms and diseases.🔹 Chatbot Integration: Enable users to interact via a chat-based interface.

👨‍💻 Contributing

Contributions are welcome! Follow these steps:

Fork the repository

Create a new branch

Commit changes

Push to GitHub

Create a pull request

🛡 License

This project is licensed under the MIT License. You are free to use and modify the code.

🤝 Contact

📧 Email: [Your Email Here]🔗 LinkedIn: [Your LinkedIn Profile]📂 GitHub: [Your GitHub Profile]

⭐ Acknowledgments

A big thank you to the Scikit-Learn and Tkinter community for their fantastic libraries! 🎉
