import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import csv
import pandas as pd

model_data = joblib.load("income_ensemble_model.pkl")
model = model_data[0]
preprocessor = model_data[1]
kmeans = model_data[2]

root = tk.Tk()
root.title("Income Predictor")

fields = [
    'age', 'education', 'workclass', 'occupation', 'marital-status', 'relationship',
    'race', 'gender', 'hours-per-week', 'native-country', 'capital-gain',
    'capital-loss', 'fnlwgt', 'educational-num'
]

options_dict = {
    'education': ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college'],
    'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov'],
    'occupation': ['Tech-support', 'Craft-repair', 'Machine-op-inspct', 'Sales'],
    'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced'],
    'relationship': ['Husband', 'Not-in-family', 'Own-child', 'Unmarried'],
    'race': ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo'],
    'gender': ['Male', 'Female'],
    'native-country': ['United-States', 'India', 'Mexico', 'Philippines']
}

widgets = {}

def predict():
    input_data = {}
    try:
        for field in fields:
            value = widgets[field].get()
            if value.strip() == "":
                messagebox.showerror("Input Error", f"Please fill in {field}")
                return
            input_data[field] = value

        df = pd.DataFrame([input_data])
        for col in df.columns:
            if col not in options_dict:
                df[col] = df[col].astype(float)

        x_preprocessed = preprocessor.transform(df)
        if hasattr(x_preprocessed, 'toarray'):
            x_preprocessed = x_preprocessed.toarray()

        cluster_label = kmeans.predict(x_preprocessed).reshape(-1, 1)
        final_input = np.hstack([x_preprocessed, cluster_label])

        pred = model.predict(final_input)
        proba = model.predict_proba(final_input)[0]
        confidence = round(proba[pred[0]] * 100, 2)
        income = ">50K" if pred[0] == 1 else "<=50K"

        messagebox.showinfo("Prediction", f"Predicted Income: {income}\nConfidence: {confidence}%")

        save_prediction(input_data, income, confidence)

    except Exception as e:
        messagebox.showerror("Error", str(e))

def save_prediction(data, income, confidence):
    with open("prediction_history.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(data.values()) + [income, f"{confidence}%"])

def show_history():
    try:
        df = pd.read_csv("prediction_history.csv", header=None)
        win = tk.Toplevel()
        win.title("Prediction History")
        text = tk.Text(win)
        text.insert(tk.END, df.to_string(index=False, header=False))
        text.pack()
    except:
        messagebox.showinfo("No Data", "No predictions yet.")

def clear_form():
    for field in fields:
        if isinstance(widgets[field], ttk.Combobox):
            widgets[field].set('')
        else:
            widgets[field].delete(0, tk.END)

for i, field in enumerate(fields):
    tk.Label(root, text=field).grid(row=i, column=0, padx=5, pady=5)
    if field in options_dict:
        cb = ttk.Combobox(root, values=options_dict[field], state="readonly")
        cb.grid(row=i, column=1, padx=5, pady=5)
        widgets[field] = cb
    else:
        ent = tk.Entry(root)
        ent.grid(row=i, column=1, padx=5, pady=5)
        widgets[field] = ent

tk.Button(root, text="Predict Income", command=predict).grid(row=len(fields), column=0, pady=10)
tk.Button(root, text="Clear", command=clear_form).grid(row=len(fields), column=1, pady=10)
tk.Button(root, text="View History", command=show_history).grid(row=len(fields)+1, column=0, columnspan=2)

root.mainloop()
