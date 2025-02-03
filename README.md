# Disease Prediction System

## Overview (📌 Introduction)
The Disease Prediction System is a machine learning-based web application that helps users predict potential diseases based on their symptoms. It uses a trained model to predict possible health conditions, assisting users in early diagnosis and preventive healthcare measures.

This project provides a Graphical User Interface (GUI) built using Tkinter, enabling users to conveniently input their symptoms and receive disease predictions in real-time.

## 🎯 Objectives

Develop an ML-based model that predicts diseases based on user-input symptoms.
Build an interactive GUI for easy usage.
Implement cross-validation to ensure accurate predictions.
Make the system scalable and flexible for future improvements.

## 🚀 Features
✅ Machine Learning Model: Uses Random Forest Classifier for accurate predictions.✅ Graphical User Interface (GUI): Built using Tkinter with an enhanced, scrollable symptom selection panel.✅ Symptom Selection Panel: Users can choose multiple symptoms from a well-organized, scrollable list.✅ Submit Button: Click to process selected symptoms and display predicted disease.✅ Popup Alert for Prediction: The predicted disease is shown in a message box.✅ Cross-Validation: Uses Stratified K-Fold Cross-Validation for better model evaluation.✅ Easy-to-Use: No deep technical knowledge required to operate.


## Features
- **Symptom-based prediction**: The system takes input symptoms from the user and provides a list of potential diseases.
- **User-friendly interface**: Simple and intuitive design for easy interaction.
- **Real-time predictions**: Provides instant predictions based on user input.
- **Data-driven insights**: Leverages machine learning models trained on healthcare data for accurate predictions.

## Technologies Used
- **Programming Language**: Python
- **Machine Learning Library**: Scikit-learn, TensorFlow
- **Web Framework**: Flask/Django (depending on the implementation)
- **Database**: MySQL/SQLite (for storing symptoms and diseases)
- **Frontend**: HTML, CSS, JavaScript (for user interaction)

## Installation

### Prerequisites
- Python 3.x
- Flask/Django
- scikit-learn
- pandas
- numpy
- jinja2
- HTML/CSS/JS libraries (if using frontend)

### Clone the repository
```bash
  git clone https://github.com/yourusername/Disease-Prediction-System.git
  cd Disease-Prediction-System
```


## 🏃‍♂️ Usage
🔹 Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/disease-prediction.git
cd disease-prediction
```

🔹 Step 2: Run the Application
```bash
python src/main.py
```

## 📝 How It Works

- User selects symptoms from a list.
- Clicks the Submit button to process symptoms.
- The system applies a Machine Learning model to predict the disease.
- The predicted disease is displayed in a popup box.

## 🎨 GUI Interface

The Disease Prediction System features an enhanced Tkinter-based GUI:
- Framed Layout: Symptoms appear inside a scrollable frame.
- Scrollable Symptom List: Users can scroll through symptoms.
- Styled Submit Button: Large, clear Submit button.
- Popup Alert: Displays the predicted disease.

##🏆 Machine Learning Model
✅ **Data Preprocessing**
- Dataset: The model is trained on a structured medical dataset containing diseases and their corresponding symptoms.
- Feature Selection: Symptoms are used as input features, while the disease name is the output label.
- Label Encoding: Disease names are converted to numeric values using LabelEncoder.
✅ **Model Training**

The system utilizes a Random Forest Classifier:

```bash
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

✅ **Cross-Validation**

To ensure robust performance, the model is evaluated using Stratified K-Fold Cross-Validation:

```bash
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"Mean Accuracy: {np.mean(scores):.4f}")
```

## ⚙ Future Enhancements

🔹 Additional ML Models: Implement models like SVM, XGBoost for comparison.🔹 Web-Based UI: Convert Tkinter GUI into a Flask or Streamlit Web App.🔹 Larger Dataset: Improve accuracy by adding more symptoms and diseases.🔹 Chatbot Integration: Enable users to interact via a chat-based interface.

## 👨‍💻 Contributing
- Contributions are welcome! Follow these steps:
- Fork the repository
- Create a new branch
- Commit changes
- Push to GitHub
- Create a pull request

## 🛡 License

This project is licensed under the MIT License. You are free to use and modify the code.

## 🤝 Contact

📧 Email: [amaltrivedi3904stella@gmail.com]🔗 LinkedIn: [[Your LinkedIn Profile](https://www.linkedin.com/in/amalprasadtrivedi-aiml-engineer/)]📂 GitHub: [[Your GitHub Profile](https://github.com/amalprasadtrivedi)]

## ⭐ Acknowledgments

A big thank you to the Scikit-Learn and Tkinter community for their fantastic libraries! 🎉
