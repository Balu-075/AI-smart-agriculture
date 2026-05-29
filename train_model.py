import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Create model folder
os.makedirs("model", exist_ok=True)

# Load Dataset from correct sheet
data = pd.read_excel(
    "data set/Crop_Recommendation.csv.xlsx",
    sheet_name="Crop Data"
)

# Features and Target
X = data.drop("Crop Label", axis=1)
y = data["Crop Label"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100)

# Save Model
pickle.dump(model, open("model/crop_model.pkl", "wb"))

print("Model Saved Successfully!")