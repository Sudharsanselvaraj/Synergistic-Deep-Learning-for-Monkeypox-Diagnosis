import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import json

# Load the saved model
loaded_model = load_model("MPOXSBD.h5")


# Load the scaler
scaler = joblib.load("scaler.save")


# Load the feature names
with open("model_features.json", "r") as f:
    model_features = json.load(f)


# Define a mapping from user input to numerical values
def yes_no_to_binary(response):
    response = response.strip().lower()
    if response in ['yes', 'y']:
        return 1
    elif response in ['no', 'n']:
        return 0
    else:
        raise ValueError("Invalid input. Please enter 'Yes' or 'No'.")

# Define the function to get user symptoms and make a prediction
def predict_monkeypox():
    print("\nPlease enter the following symptoms (Yes/No):\n")

    # Collect user inputs for the features excluding 'sum'
    try:
        rectal_pain = yes_no_to_binary(input("Rectal Pain : "))
        sore_throat = yes_no_to_binary(input("Sore Throat : "))
        penile_oedema = yes_no_to_binary(input("Penile Oedema : "))
        oral_lesions = yes_no_to_binary(input("Oral Lesions : "))
        solitary_lesion = yes_no_to_binary(input("Solitary Lesion : "))
        swollen_tonsils = yes_no_to_binary(input("Swollen Tonsils : "))
        hiv_infection = yes_no_to_binary(input("HIV Infection : "))
        st_infection = yes_no_to_binary(input("Sexually Transmitted Infection : "))
        systemic_illness1 = yes_no_to_binary(input("Systemic Illness - Fever : "))
        systemic_illness2 = yes_no_to_binary(input("Systemic Illness - Muscle Aches and Pain : "))
        systemic_illness3 = yes_no_to_binary(input("Systemic Illness - Swollen Lymph Nodes : "))
        systemic_illness4 = yes_no_to_binary(input("Systemic Illness - [Specify Illness] : "))
    except ValueError as e:
        print(e)
        return

    # Create a dictionary with the exact feature names used during training
    user_data = {
        "Rectal Pain": rectal_pain,
        "Sore Throat": sore_throat,
        "Penile Oedema": penile_oedema,
        "Oral Lesions": oral_lesions,
        "Solitary Lesion": solitary_lesion,
        "Swollen Tonsils": swollen_tonsils,
        "HIV Infection": hiv_infection,
        "Sexually Transmitted Infection": st_infection,
        "Systemic Illness_Fever": systemic_illness1,
        "Systemic Illness_Muscle Aches and Pain": systemic_illness2,
        "Systemic Illness_Swollen Lymph Nodes": systemic_illness3,
        "Systemic Illness_[Specify Illness]": systemic_illness4
    }

    # Convert to DataFrame
    user_df = pd.DataFrame([user_data])

    # Compute 'sum' if it's part of the features
    if 'sum' in model_features:
        user_df['sum'] = user_df.sum(axis=1)
    else:
        user_df['sum'] = user_df.sum(axis=1)  # Adjust as per your training data

    # Ensure the DataFrame has exactly the features used during training
    # Any missing features are filled with 0
    user_df = user_df.reindex(columns=model_features, fill_value=0)

    # Scale the input
    user_scaled = scaler.transform(user_df)

    # Make a prediction
    prediction = loaded_model.predict(user_scaled)

    # Extract the probability
    probability = prediction[0][0]

    # Output the result
    print(f"\nPrediction Probability: {probability:.4f}")
    if probability > 0.5:
        print("The model predicts: **Monkeypox Positive**")
    else:
        print("The model predicts: **Monkeypox Negative**")

# Run the prediction function
if __name__ == "__main__":
    predict_monkeypox()