import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import json

# Load the model, scaler, and feature names as before
loaded_model = load_model("MPOXSBD.h5")
scaler = joblib.load("scaler.save")
with open("model_features.json", "r") as f:
    model_features = json.load(f)

# Function for prediction
def predict_monkeypox():
    user_data = {
        "Rectal Pain": var_rectal_pain.get(),
        "Sore Throat": var_sore_throat.get(),
        "Penile Oedema": var_penile_oedema.get(),
        "Oral Lesions": var_oral_lesions.get(),
        "Solitary Lesion": var_solitary_lesion.get(),
        "Swollen Tonsils": var_swollen_tonsils.get(),
        "HIV Infection": var_hiv_infection.get(),
        "Sexually Transmitted Infection": var_st_infection.get(),
        "Systemic Illness_Fever": var_sys_illness1.get(),
        "Systemic Illness_Muscle Aches and Pain": var_sys_illness2.get(),
        "Systemic Illness_Swollen Lymph Nodes": var_sys_illness3.get(),
        "Systemic Illness_[Specify Illness]": var_sys_illness4.get(),
    }
    user_df = pd.DataFrame([user_data])
    user_df['sum'] = user_df.sum(axis=1)
    user_df = user_df.reindex(columns=model_features, fill_value=0)
    user_scaled = scaler.transform(user_df)
    probability = loaded_model.predict(user_scaled)[0][0]
    result = "Monkeypox Positive" if probability > 0.5 else "Monkeypox Negative"
    messagebox.showinfo("Prediction Result", f"Prediction Probability: {probability:.4f}\nResult: {result}")

# GUI setup
root = tk.Tk()
root.title("Monkeypox Prediction")

# Define font style (Bold and larger size)
font_style = ("Helvetica", 14, "bold")  # Adjust font size and make it bold

# Define Tkinter variables for checkboxes
var_rectal_pain = tk.IntVar()
var_sore_throat = tk.IntVar()
var_penile_oedema = tk.IntVar()
var_oral_lesions = tk.IntVar()
var_solitary_lesion = tk.IntVar()
var_swollen_tonsils = tk.IntVar()
var_hiv_infection = tk.IntVar()
var_st_infection = tk.IntVar()
var_sys_illness1 = tk.IntVar()
var_sys_illness2 = tk.IntVar()
var_sys_illness3 = tk.IntVar()
var_sys_illness4 = tk.IntVar()

# Create a Frame to structure the layout
frame = tk.Frame(root)
frame.pack(fill="both", expand=True, padx=20, pady=20)

# Create checkboxes for each symptom
tk.Checkbutton(frame, text="Rectal Pain", variable=var_rectal_pain, font=font_style).pack(fill="x", pady=5)
tk.Checkbutton(frame, text="Sore Throat", variable=var_sore_throat, font=font_style).pack(fill="x", pady=5)
tk.Checkbutton(frame, text="Penile Oedema", variable=var_penile_oedema, font=font_style).pack(fill="x", pady=5)
tk.Checkbutton(frame, text="Oral Lesions", variable=var_oral_lesions, font=font_style).pack(fill="x", pady=5)
tk.Checkbutton(frame, text="Solitary Lesion", variable=var_solitary_lesion, font=font_style).pack(fill="x", pady=5)
tk.Checkbutton(frame, text="Swollen Tonsils", variable=var_swollen_tonsils, font=font_style).pack(fill="x", pady=5)
tk.Checkbutton(frame, text="HIV Infection", variable=var_hiv_infection, font=font_style).pack(fill="x", pady=5)
tk.Checkbutton(frame, text="Sexually Transmitted Infection", variable=var_st_infection, font=font_style).pack(fill="x", pady=5)
tk.Checkbutton(frame, text="Systemic Illness - Fever", variable=var_sys_illness1, font=font_style).pack(fill="x", pady=5)
tk.Checkbutton(frame, text="Systemic Illness - Muscle Aches and Pain", variable=var_sys_illness2, font=font_style).pack(fill="x", pady=5)
tk.Checkbutton(frame, text="Systemic Illness - Swollen Lymph Nodes", variable=var_sys_illness3, font=font_style).pack(fill="x", pady=5)
tk.Checkbutton(frame, text="Systemic Illness - [Specify Illness]", variable=var_sys_illness4, font=font_style).pack(fill="x", pady=5)

# Smaller prediction button with bold text
predict_button = tk.Button(frame, text="Predict", command=predict_monkeypox, font=font_style, height=2, width=10)  # Adjust width here
predict_button.pack(pady=20)

# Make the window fill the available space
root.geometry("500x600")  # Adjust window size to better fit the content

root.mainloop()
