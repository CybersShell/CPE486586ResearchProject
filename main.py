import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from whitepossum import model as wpModel
import seaborn as sns

# -------------------------
# LOAD DATA
# -------------------------

column_names = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label"
]

train_df = pd.read_csv("nsl-kdd/KDDTrain+.txt", names=column_names)
test_df = pd.read_csv("nsl-kdd/KDDTest+.txt", names=column_names)

full_df = pd.concat([train_df, test_df])
full_df = shuffle(full_df)

from whitepossum import LogisticRegression

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
def load_nslkdd_data(train_path, test_path):
    """Load NSL-KDD dataset."""
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
    ]

    train_df = pd.read_csv(train_path, names=columns)
    test_df = pd.read_csv(test_path, names=columns)

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, test_df


def preprocess_data(train_df, test_df):
    """Preprocess NSL-KDD data."""
    # Create binary labels
    train_df['label'] = (train_df['attack_type'] != 'normal').astype(int)
    test_df['label'] = (test_df['attack_type'] != 'normal').astype(int)

    # Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    combined_df = pd.concat([train_df, test_df], axis=0)

    for col in categorical_cols:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col].astype(str))

    train_df_encoded = combined_df.iloc[:len(train_df)].copy()
    test_df_encoded = combined_df.iloc[len(train_df):].copy()

    # Separate features and labels
    drop_cols = ['attack_type', 'difficulty', 'label']
    X_train = train_df_encoded.drop(drop_cols, axis=1).values.astype(np.float32)
    y_train = train_df['label'].values.astype(np.float32)
    X_test = test_df_encoded.drop(drop_cols, axis=1).values.astype(np.float32)
    y_test = test_df['label'].values.astype(np.float32)

    # Handle any NaN/Inf before scaling
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Final safety check
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    print(f"\nFeatures: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Check predictions
    unique, counts = np.unique(y_pred, return_counts=True)
    print("\nPrediction Distribution:")
    for u, c in zip(unique, counts):
        label = 'Normal' if u == 0 else 'Attack'
        print(f"  {label}: {c} ({c/len(y_pred)*100:.1f}%)")

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*50)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack'], zero_division=0))

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


print("="*70)
print("LOGISTIC REGRESSION FOR NETWORK INTRUSION DETECTION")
print("Dataset: NSL-KDD")
print("="*70)

# Load data
print("\n[1/4] Loading NSL-KDD dataset...")
train_df, test_df = load_nslkdd_data('KDDTrain+.txt', 'KDDTest+.txt')

# Preprocess
print("\n[2/4] Preprocessing data...")
X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)

# Split for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\nClass distribution (training):")
print(f"  Normal: {(y_train_split==0).sum()}")
print(f"  Attack: {(y_train_split==1).sum()}")

# Train
print("\n[3/4] Training Logistic Regression model...")
model = LogisticRegression(
n_features=X_train.shape[1],
learning_rate=0.01,
n_epochs=100,
batch_size=256
)

model.fit(X_train_split, y_train_split, X_val, y_val)

# Plot
model.plot_training_history()

# Evaluate
print("\n[4/4] Evaluating model on test set...")
metrics = evaluate_model(model, X_test, y_test)

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)

full_df["binary_label"] = full_df["label"].apply(lambda x: "normal" if x == "normal" else "attack")

train_size = len(train_df)
train_df = full_df.iloc[:train_size]
test_df  = full_df.iloc[train_size:]

X_train = train_df.drop(["label", "binary_label"], axis=1)
y_train = train_df["binary_label"]

X_test = test_df.drop(["label", "binary_label"], axis=1)
y_test = test_df["binary_label"]

# -------------------------
# FIX: FORCE NUMERIC COLUMNS TO FLOAT
# -------------------------

categorical_cols = ["protocol_type", "service", "flag"]
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

for col in numeric_cols:
    le = LabelEncoder()
    if X_train[col].dtype == type(object):
        X_train[col] = le.fit_transform(X_train[col])
    if X_test[col].dtype == type(object):
        X_test[col] = le.fit_transform(X_test[col])
        
# -------------------------
# PREPROCESSING PIPELINE
# -------------------------

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)

model = Pipeline([
    ("preprocess", preprocess),
    ("rf", RandomForestClassifier(n_estimators=20000, random_state=46, class_weight="balanced"))
])

# -------------------------
# TRAIN + TEST
# -------------------------

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------
# CONFUSION MATRIX
# -------------------------

cm = confusion_matrix(y_test, y_pred, labels=["normal", "attack"])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



