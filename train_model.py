import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("Customer-Churn.csv")

# Drop customer ID and rows with missing values
df.drop(["customerID"], axis=1, inplace=True)
df.dropna(inplace=True)

# Convert 'TotalCharges' to numeric if necessary
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Convert target to binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df)

# Separate features and target
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# Save the column order
feature_names = X.columns.tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the feature names
with open("features.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("✅ Model and features saved successfully!")
