import os
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from flask import Flask, render_template, request
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# Paths
train_csv = "train.csv"
img_folder = "static/images"

# Load dataset
def load_data(train_csv, img_folder, img_size=(64, 64, 3)):
    df_train = pd.read_csv(train_csv)
    print("CSV loaded. Total rows:", len(df_train))

    X, y, img_paths = [], [], []
    for _, row in df_train.iterrows():
        img_path = os.path.join(img_folder, f"{row['id']}.jpg")
        if os.path.exists(img_path):
            img = imread(img_path)
            img_resized = resize(img, img_size, anti_aliasing=True, mode='reflect')
            X.append(img_resized.reshape(-1))  # Flatten
            y.append(row['species'])
            img_paths.append(img_path)
        else:
            print("Missing image:", img_path)

    print("Images loaded:", len(X))
    return np.array(X), np.array(y), img_paths

# Train MLP model
def train_mlp(X, y, test_size=0.4, random_state=42):
    # Normalize
    X = X / 255.0

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Balance data with SMOTE
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

    # Optional: Dimensionality reduction (helps MLP performance)
    pca = PCA(n_components=100)  # reduce to 100 components
    X_resampled = pca.fit_transform(X_resampled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_size, random_state=random_state, stratify=y_resampled
    )

    mlp_model = MLPClassifier(hidden_layer_sizes=(150,), max_iter=500, random_state=random_state)
    mlp_model.fit(X_train, y_train)

    return mlp_model, label_encoder, X_test, y_test, pca

# Evaluation
def evaluate_mlp(mlp_model, X_test, y_test):
    y_pred = mlp_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    specificity = np.diag(cm) / (np.sum(cm, axis=1) + 1e-6)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(np.mean(specificity), 4)
    }

# Prediction for a single image
def classify_leaf(mlp_model, label_encoder, pca, img_folder, img_num):
    local_img_path = os.path.join(img_folder, f"{img_num}.jpg")
    if not os.path.exists(local_img_path):
        return None
    img = imread(local_img_path)
    img_resized = resize(img, (64, 64, 3), anti_aliasing=True, mode='reflect').reshape(1, -1)
    img_resized = img_resized / 255.0

    img_pca = pca.transform(img_resized)
    y_pred = mlp_model.predict(img_pca)

    predicted_species = label_encoder.inverse_transform(y_pred)[0]

    web_img_path = f"/static/images/{img_num}.jpg"
    return predicted_species, web_img_path

# Load and train model once
X, y, img_paths = load_data(train_csv, img_folder)
mlp_model, label_encoder, X_test, y_test, pca = train_mlp(X, y)
evaluation_results = evaluate_mlp(mlp_model, X_test, y_test)

# Flask routes
@app.route('/')
def index():
    return render_template("index.html", evaluation_results=evaluation_results, predicted_species=None, img_path=None)

@app.route('/classify', methods=['POST'])
def classify():
    img_num = request.form['img_num']
    result = classify_leaf(mlp_model, label_encoder, pca, img_folder, img_num)
    if result:
        predicted_species, img_path = result
        return render_template("index.html", evaluation_results=evaluation_results, predicted_species=predicted_species, img_path=img_path)
    else:
        return render_template("index.html", evaluation_results=evaluation_results, predicted_species="Image not found!", img_path=None)

if __name__ == '__main__':
    app.run(debug=True)
