import os
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from flask import Flask, render_template, request
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder

app = Flask(__name__)

# Paths
train_csv = "train.csv"
img_folder = "static/images"

# Load dataset
def load_data(train_csv, img_folder, img_size=(224, 224, 3)):
    df_train = pd.read_csv(train_csv)
    print("CSV loaded. Total rows:", len(df_train))

    X, y, img_paths = [], [], []
    for _, row in df_train.iterrows():
        img_path = os.path.join(img_folder, f"{row['id']}.jpg")
        if os.path.exists(img_path):
            img = imread(img_path)
            img_resized = resize(img, img_size, anti_aliasing=True, mode='reflect')
            X.append(img_resized.reshape(-1))
            y.append(row['species'])
            img_paths.append(img_path)
        else:
            print("Missing image:", img_path)

    print("Images loaded:", len(X))
    return np.array(X), np.array(y), img_paths

# Train Random Forest model
def train_random_forest(X, y, n_estimators=100, test_size=0.4, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Label encode the target labels (y)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train_encoded)
    
    return rf_model, label_encoder, X_test, y_test_encoded

# Evaluation using Majority Voting
def evaluate_random_forest(rf_model, X_test, y_test):
    tree_preds = np.array([tree.predict(X_test) for tree in rf_model.estimators_])
    y_pred_majority = mode(tree_preds, axis=0, keepdims=False).mode

    accuracy = accuracy_score(y_test, y_pred_majority)
    precision = precision_score(y_test, y_pred_majority, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred_majority, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred_majority, average='macro', zero_division=0)

    cm = confusion_matrix(y_test, y_pred_majority)
    specificity = np.diag(cm) / (np.sum(cm, axis=1) + 1e-6)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(np.mean(specificity), 4)
    }

# Prediction for a single image
def classify_leaf(rf_model, label_encoder, img_folder, img_num):
    local_img_path = os.path.join(img_folder, f"{img_num}.jpg")
    if not os.path.exists(local_img_path):
        return None
    img = imread(local_img_path)
    img_resized = resize(img, (224, 224, 3), anti_aliasing=True, mode='reflect').reshape(1, -1)

    # Get predictions from each tree
    tree_preds = np.array([tree.predict(img_resized) for tree in rf_model.estimators_])
    predicted_species_encoded = mode(tree_preds, axis=0, keepdims=False).mode[0]

    # Ensure predicted species is an integer before inverse transforming
    predicted_species_encoded = int(predicted_species_encoded)  # Casting to int

    # Decode the predicted label
    predicted_species = label_encoder.inverse_transform([predicted_species_encoded])[0]
    
    web_img_path = f"/static/images/{img_num}.jpg"
    return predicted_species, web_img_path

# Load and train model once
X, y, img_paths = load_data(train_csv, img_folder)
rf_model, label_encoder, X_test, y_test = train_random_forest(X, y)
evaluation_results = evaluate_random_forest(rf_model, X_test, y_test)

# Flask routes
@app.route('/')
def index():
    return render_template("index.html", evaluation_results=evaluation_results, predicted_species=None, img_path=None)

@app.route('/classify', methods=['POST'])
def classify():
    img_num = request.form['img_num']
    result = classify_leaf(rf_model, label_encoder, img_folder, img_num)
    if result:
        predicted_species, img_path = result
        return render_template("index.html", evaluation_results=evaluation_results, predicted_species=predicted_species, img_path=img_path)
    else:
        return render_template("index.html", evaluation_results=evaluation_results, predicted_species="Image not found!", img_path=None)

if __name__ == '__main__':
    app.run(debug=True)
