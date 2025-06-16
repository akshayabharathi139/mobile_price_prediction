import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("mobile_price_data.csv")  # Ensure this CSV file exists in the 'data/' folder

# Split features and target
X = df.drop(columns=["price_range"])
y = df["price_range"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model inside the 'model' folder
import os
os.makedirs("model", exist_ok=True)  # Create 'model' folder if it doesn't exist
with open("model/price_predictor.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully as 'model/price_predictor.pkl'")
