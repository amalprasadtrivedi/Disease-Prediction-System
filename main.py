import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load Dataset
file_path = "D:/Project Copy/Basic Machine Learning Projects/Disease Prediction System/Training.csv"
data = pd.read_csv(file_path)

# Drop empty columns if any
data.dropna(axis=1, how="all", inplace=True)

# Extract features (symptoms) and target (disease)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]  # Last column (disease labels)

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Perform Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Mean Accuracy: {np.mean(scores):.4f}")


# GUI for Disease Prediction
class DiseasePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Disease Prediction System")
        self.root.geometry("600x500")
        self.root.configure(bg="#f2f2f2")

        # Title Label
        ttk.Label(root, text="Disease Prediction System", font=("Arial", 18, "bold"), background="#f2f2f2").pack(
            pady=10)

        # Symptom Selection Frame
        symptom_frame = tk.Frame(root, bg="white", relief="ridge", bd=2)
        symptom_frame.pack(pady=10, padx=20, fill="both", expand=True)

        ttk.Label(symptom_frame, text="Select Symptoms:", font=("Arial", 12, "bold"), background="white").pack(pady=5)

        # Symptoms List
        self.symptoms = list(X.columns)
        self.selected_symptoms = []

        # Scrollable Symptom Selection
        canvas = tk.Canvas(symptom_frame, bg="white")
        scrollbar = ttk.Scrollbar(symptom_frame, orient="vertical", command=canvas.yview)
        symptom_list_frame = tk.Frame(canvas, bg="white")

        symptom_list_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=symptom_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.symptom_vars = {symptom: tk.IntVar() for symptom in self.symptoms}
        for symptom in self.symptoms:
            chk = ttk.Checkbutton(symptom_list_frame, text=symptom, variable=self.symptom_vars[symptom])
            chk.pack(anchor="w", padx=10)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Predict Button
        submit_btn = ttk.Button(root, text="Submit", command=self.predict_disease, style="Submit.TButton")
        submit_btn.pack(pady=20)

        # Result Label
        self.result_label = ttk.Label(root, text="", font=("Arial", 14, "bold"), foreground="blue",
                                      background="#f2f2f2")
        self.result_label.pack()

    def predict_disease(self):
        # Get selected symptoms
        input_data = np.zeros(len(self.symptoms))
        for i, symptom in enumerate(self.symptoms):
            if self.symptom_vars[symptom].get() == 1:
                input_data[i] = 1

        # Predict Disease
        input_data = input_data.reshape(1, -1)
        prediction = model.predict(input_data)
        predicted_disease = label_encoder.inverse_transform(prediction)[0]

        # Show Result in Popup
        messagebox.showinfo("Prediction Result", f"The Predicted Disease is: {predicted_disease}")
        self.result_label.config(text=f"Predicted Disease: {predicted_disease}")


# Run GUI
root = tk.Tk()
app = DiseasePredictionApp(root)
root.mainloop()
