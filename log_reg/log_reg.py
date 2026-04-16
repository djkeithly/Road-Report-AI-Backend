import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# 1. Define the Model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# --- Define paths for saving/loading ---
MODEL_PATH = "logistic_regression_model.pth"
COLS_PATH = "feature_columns.pkl"

# 2. Load and Preprocess Data
data_file = "TrainingData.csv"
try:
    data = pd.read_csv(data_file)
    print("Data file read successfully.")
except FileNotFoundError:
    print(f"Error: {data_file} not found. Please ensure it's in the same directory.")
    exit()

drop_cols = [
    "Crash ID",
    "Crash Date",
    "Hour of Day",
    "Crash Year",
    "Day of Week",
    "Rural Urban Type",
]
data = data.drop(columns=drop_cols, errors="ignore")

# One-Hot Encoding
data = pd.get_dummies(data, drop_first=True)

# Save feature columns and separate features/labels
feature_columns = data.columns[:-1]
features_array = data.iloc[:, :-1].to_numpy(dtype=np.float32)
labels_array = data.iloc[:, -1].to_numpy(dtype=np.float32).reshape(-1, 1)

# 3. Convert Numpy arrays to PyTorch Tensors
X_tensor = torch.from_numpy(features_array)
y_tensor = torch.from_numpy(labels_array)

# Get the number of features to dynamically set the model's input dimension
input_dim = X_tensor.shape[1]

# 4. Initialize Model, Loss Function, and Optimizer
model = LogisticRegression(input_dim)
criterion = nn.BCEWithLogitsLoss()
learning_rate = 0.0075
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ==========================================
# 5. Check, Load, or Train Logic
# ==========================================
if os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH):
    print(f"\nFound saved files! Loading model from '{MODEL_PATH}'...")

    # Load the feature columns mapping
    with open(COLS_PATH, "rb") as f:
        feature_columns = pickle.load(f)

    # Load the model weights (weights_only=True is a PyTorch security standard)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()  # Set to evaluation mode immediately after loading
    print("Model and feature columns loaded successfully!")

else:
    print("\nNo saved model found. Starting training...")
    epochs = 500

    for epoch in range(epochs):
        # --- Forward Pass ---
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        # --- Backward Pass ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print("Training complete!")

    # Save the model weights
    torch.save(model.state_dict(), MODEL_PATH)

    # Save the feature columns using pickle
    with open(COLS_PATH, "wb") as f:
        pickle.dump(feature_columns, f)

    print(f"Model saved to '{MODEL_PATH}'")
    print(f"Feature columns saved to '{COLS_PATH}'")

# ==========================================
# Testing Custom Values
# ==========================================

banding_dataset = [
    {
        # "Crash ID": [30000283],
        "City": ["DALLAS"],
        "County": ["DALLAS"],
        # "Crash Date": ["2025-01-01"],
        "Crash Month": ["1"],
        "Crash Time": ["0"],
        # "Crash Year": ["2025"],
        # "Day of Week": ["WEDNESDAY"],
        # "Hour of Day": ["00:00 - 00:59"],
        # "Road Class": ["CITY STREET"],
        "Road Class": ["CITY STREET"],
        "Street Name": ["S I 35E S"],
        "Surface Condition": ["1 - DRY"],
        "Weather Condition": ["1 - CLEAR"],
    },
    {
        # "Crash ID": [30000283],
        "City": ["DALLAS"],
        "County": ["DALLAS"],
        # "Crash Date": ["2025-01-01"],
        "Crash Month": ["1"],
        "Crash Time": ["0"],
        # "Crash Year": ["2025"],
        # "Day of Week": ["WEDNESDAY"],
        # "Hour of Day": ["00:00 - 00:59"],
        # "Road Class": ["CITY STREET"],
        "Road Class": ["INTERSTATE"],
        "Street Name": ["S I 35E S"],
        "Surface Condition": ["1 - DRY"],
        "Weather Condition": ["1 - CLEAR"],
    },
    {
        # "Crash ID": [30000283],
        "City": ["DALLAS"],
        "County": ["DALLAS"],
        # "Crash Date": ["2025-01-01"],
        "Crash Month": ["1"],
        "Crash Time": ["0"],
        # "Crash Year": ["2025"],
        # "Day of Week": ["WEDNESDAY"],
        # "Hour of Day": ["00:00 - 00:59"],
        # "Road Class": ["CITY STREET"],
        "Road Class": ["INTERSTATE"],
        "Street Name": ["S I 35E S"],
        "Surface Condition": ["2 - WET"],
        "Weather Condition": ["3 - RAIN"],
    },
]
my_custom_data_set = [
    {
        # "Crash ID": [30000283],
        "City": ["FARMERS BRANCH"],
        "County": ["DALLAS"],
        # "Crash Date": ["2025-01-01"],
        "Crash Month": ["1"],
        "Crash Time": ["0"],
        # "Crash Year": ["2025"],
        # "Day of Week": ["WEDNESDAY"],
        # "Hour of Day": ["00:00 - 00:59"],
        # "Road Class": ["CITY STREET"],
        "Road Class": ["CITY STREET"],
        "Street Name": ["S I 35E S"],
        "Surface Condition": ["1 - DRY"],
        "Weather Condition": ["1 - CLEAR"],
    },
    {
        # "Crash ID": [30000283],
        "City": ["DALLAS"],
        "County": ["DALLAS"],
        # "Crash Date": ["2025-01-01"],
        "Crash Month": ["1"],
        "Crash Time": ["0"],
        # "Crash Year": ["2025"],
        # "Day of Week": ["WEDNESDAY"],
        # "Hour of Day": ["00:00 - 00:59"],
        # "Road Class": ["CITY STREET"],
        "Road Class": ["CITY STREET"],
        "Street Name": ["S I 35E S"],
        "Surface Condition": ["1 - DRY"],
        "Weather Condition": ["1 - CLEAR"],
    },
    {
        # "Crash ID": [30000283],
        "City": ["DALLAS"],
        "County": ["DALLAS"],
        # "Crash Date": ["2025-01-01"],
        "Crash Month": ["1"],
        "Crash Time": ["0"],
        # "Crash Year": ["2025"],
        # "Day of Week": ["WEDNESDAY"],
        # "Hour of Day": ["00:00 - 00:59"],
        # "Road Class": ["CITY STREET"],
        "Road Class": ["INTERSTATE"],
        "Street Name": ["S I 35E S"],
        "Surface Condition": ["1 - DRY"],
        "Weather Condition": ["1 - CLEAR"],
    },
    {
        # "Crash ID": [30000283],
        "City": ["DALLAS"],
        "County": ["DALLAS"],
        # "Crash Date": ["2025-01-01"],
        "Crash Month": ["1"],
        "Crash Time": ["0"],
        # "Crash Year": ["2025"],
        # "Day of Week": ["WEDNESDAY"],
        # "Hour of Day": ["00:00 - 00:59"],
        # "Road Class": ["CITY STREET"],
        "Road Class": ["CITY STREET"],
        "Street Name": ["S I 35E S"],
        "Surface Condition": ["2 - WET"],
        "Weather Condition": ["3 - RAIN"],
    },
    {
        # "Crash ID": [30000283],
        "City": ["ARLINGTON"],
        "County": ["ARLINGTON"],
        # "Crash Date": ["2025-01-01"],
        "Crash Month": ["1"],
        "Crash Time": ["0"],
        # "Crash Year": ["2025"],
        # "Day of Week": ["WEDNESDAY"],
        # "Hour of Day": ["00:00 - 00:59"],
        # "Road Class": ["CITY STREET"],
        "Road Class": ["CITY STREET"],
        "Street Name": ["S I 35E S"],
        "Surface Condition": ["2 - WET"],
        "Weather Condition": ["3 - RAIN"],
    },
    {
        # "Crash ID": [30000283],
        "City": ["DALLAS"],
        "County": ["DALLAS"],
        # "Crash Date": ["2025-01-01"],
        "Crash Month": ["1"],
        "Crash Time": ["0"],
        # "Crash Year": ["2025"],
        # "Day of Week": ["WEDNESDAY"],
        # "Hour of Day": ["00:00 - 00:59"],
        # "Road Class": ["CITY STREET"],
        "Road Class": ["CITY STREET"],
        "Street Name": ["BUCKINGHAM RD"],
        "Surface Condition": ["1 - DRY"],
        "Weather Condition": ["1 - CLEAR"],
    },
    {
        # "Crash ID": [30000283],
        "City": ["ARLINGTON"],
        "County": ["ARLINGTON"],
        # "Crash Date": ["2025-01-01"],
        "Crash Month": ["1"],
        "Crash Time": ["0"],
        # "Crash Year": ["2025"],
        # "Day of Week": ["WEDNESDAY"],
        # "Hour of Day": ["00:00 - 00:59"],
        # "Road Class": ["CITY STREET"],
        "Road Class": ["INTERSTATE"],
        "Street Name": ["S I 35E S"],
        "Surface Condition": ["1 - DRY"],
        "Weather Condition": ["1 - CLEAR"],
    },
]

banding_cutoff = [0, 0, 0]
iter = 0
for band_item in banding_dataset:
    # Convert your custom data into a Pandas DataFrame
    custom_df = pd.DataFrame(band_item)

    # Drop the same columns you dropped during training
    custom_df = custom_df.drop(columns=drop_cols, errors="ignore")

    # One-Hot Encode your custom data
    custom_encoded = pd.get_dummies(custom_df)

    # Align columns using the feature_columns (either newly created or loaded from pickle!)
    custom_aligned = custom_encoded.reindex(columns=feature_columns, fill_value=0)

    # Convert to a PyTorch Tensor
    custom_tensor = torch.tensor(custom_aligned.to_numpy(dtype=np.float32))

    # Run the Prediction
    model.eval()
    with torch.no_grad():
        raw_logits = model(custom_tensor)
        probability = torch.sigmoid(raw_logits)
        predicted_class = (probability >= 0.5).float()

    print("\nBanding cutoff for: ", iter)
    print(f"Predicted Probability: {probability.item():.4f}")
    banding_cutoff[iter] = probability.item()
    iter += 1

for my_custom_data in my_custom_data_set:
    # Convert your custom data into a Pandas DataFrame
    custom_df = pd.DataFrame(my_custom_data)

    # Drop the same columns you dropped during training
    custom_df = custom_df.drop(columns=drop_cols, errors="ignore")

    # One-Hot Encode your custom data
    custom_encoded = pd.get_dummies(custom_df)

    # Align columns using the feature_columns (either newly created or loaded from pickle!)
    custom_aligned = custom_encoded.reindex(columns=feature_columns, fill_value=0)

    # Convert to a PyTorch Tensor
    custom_tensor = torch.tensor(custom_aligned.to_numpy(dtype=np.float32))

    # Run the Prediction
    model.eval()
    with torch.no_grad():
        raw_logits = model(custom_tensor)
        probability = torch.sigmoid(raw_logits)
        predicted_class = (probability >= 0.5).float()

    print("\n--- Custom Value Prediction ---")
    print(f"Predicted Probability: {probability.item():.4f}")
    score = 3
    for i in range(3):
        if probability.item() > banding_cutoff[i]:
            score = i
            break
    print("Risk Score: ", i)
