import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. Load Data ---
# Adjust path if using Google Drive
DATA_PATH = 'patient_data.csv'
print(f"Loading dataset from: {DATA_PATH}")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find {DATA_PATH}. Please ensure the data file is present.")

# --- 2. Exploratory Data Analysis (EDA) ---
print("\n=== Sample Data ===")
print(df.head())

print("\n=== Data Description ===")
print(df.describe(include='all').transpose())

# Plot distributions for numerical features
def plot_numeric_distribution(df, numeric_cols):
    for col in numeric_cols:
        plt.figure()
        plt.hist(df[col].dropna(), bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

# Plot counts for categorical features
def plot_categorical_counts(df, categorical_cols):
    for col in categorical_cols:
        counts = df[col].value_counts()
        plt.figure()
        plt.bar(counts.index.astype(str), counts.values)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.show()

# Identify feature types
target_col = 'disease_outcome'
all_cols = df.columns.tolist()
feature_cols = [c for c in all_cols if c != target_col]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

plot_numeric_distribution(df, [c for c in numeric_cols if c != target_col])
plot_categorical_counts(df, categorical_cols)

# --- 3. Preprocessing & Encoding ---
df_processed = df.copy()
encoders = {}
# Encode categorical features
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    encoders[col] = le

# Scale numeric features
scaler = StandardScaler()
df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

# Encode target
le_target = LabelEncoder()
df_processed[target_col] = le_target.fit_transform(df_processed[target_col].astype(str))
encoders[target_col] = le_target

# Features and target arrays
X = df_processed[feature_cols]
y = df_processed[target_col]

# --- 4. Train/Test Split & Model Training ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 5. Evaluation ---
y_pred = model.predict(X_test)
print('\n=== Classification Report ===')
print(classification_report(y_test, y_pred,
      target_names=le_target.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(np.arange(len(le_target.classes_)), le_target.classes_, rotation=45, ha='right')
plt.yticks(np.arange(len(le_target.classes_)), le_target.classes_)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Feature Importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure()
plt.barh([feature_cols[i] for i in indices], importances[indices])
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

# --- 6. Save Model, Scaler & Encoders ---
with open('disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print("Saved model to 'disease_model.pkl', scaler to 'scaler.pkl', and encoders to 'encoders.pkl'.")

# --- 7. Prediction Utility ---
def predict_disease(input_dict: dict) -> str:
    """
    Predict disease outcome from a dict of patient attributes.
    Example:
      input_dict = {
        'age': 54,
        'blood_pressure': 130,
        'cholesterol': 220,
        'gender': 'Male',
        ...
      }
    Returns:
      Predicted disease outcome label (string).
    """
    # Prepare row
    row = []
    for col in feature_cols:
        if col not in input_dict:
            raise KeyError(f"Missing feature '{col}'")
        val = input_dict[col]
        if col in categorical_cols:
            row.append(encoders[col].transform([str(val)])[0])
        else:
            # numeric, scale
            scaled = scaler.transform([[input_dict[col] if col not in categorical_cols else 0]])[0][numeric_cols.index(col)]
            row.append(scaled)
    code = model.predict([row])[0]
    return le_target.inverse_transform([code])[0]

# Example usage
aif __name__ == '__main__':
    # Use median/mode for example
    sample = {}
    for col in feature_cols:
        if col in categorical_cols:
            sample[col] = df[col].mode()[0]
        else:
            sample[col] = df[col].median()
    prediction = predict_disease(sample)
    print("Example prediction:", prediction)
