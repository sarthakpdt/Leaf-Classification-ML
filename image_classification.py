import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread  # Read images
from skimage.transform import resize  # Resize images
from sklearn.model_selection import train_test_split  # Split dataset
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # Metrics

def load_data(train_csv, img_folder, img_size=(224, 224, 3)):
    # Load images and labels from CSV
    df_train = pd.read_csv(train_csv)
    X, y, img_paths = [], [], []
    for _, row in df_train.iterrows():  # Iterate through CSV rows
        img_path = os.path.join(img_folder, f"{row['id']}.jpg")  # Construct image path
        if os.path.exists(img_path):  # Check if image exists
            img = imread(img_path)  # Read image
            img_resized = resize(img, img_size, anti_aliasing=True, mode='reflect')  # Resize image
            X.append(img_resized.reshape(-1))  # Flatten and store image
            y.append(row['species'])  # Store label
            img_paths.append(img_path)  # Store image path
    return np.array(X), np.array(y), img_paths  # Convert to arrays

def train_knn(X, y, n_neighbors=5, test_size=0.4, random_state=42):
    # Split dataset and train KNN model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)  # Initialize KNN classifier
    knn.fit(X_train, y_train)  # Train model
    return knn, X_test, y_test  # Return trained model and test data

def evaluate_knn(knn, X_test, y_test):
    # Evaluate KNN model
    y_pred = knn.predict(X_test)  # Predict labels
    accuracy = accuracy_score(y_test, y_pred)  # Compute accuracy
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)  # Compute precision
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)  # Compute recall
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)  # Compute F1 score
    cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix
    specificity = np.diag(cm) / (np.sum(cm, axis=1) + 1e-6)  # Compute specificity

    # Print evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Specificity:", np.mean(specificity))

def classify_leaf(knn, img_folder):
    # Classify an image based on user input
    img_num = input("Enter image number: ")  # Get user input
    img_path = os.path.join(img_folder, f"{img_num}.jpg")  # Construct image path
    if not os.path.exists(img_path):  # Check if image exists
        print("Image not found!")
        return
    img = imread(img_path)  # Read image
    img_resized = resize(img, (224, 224, 3), anti_aliasing=True, mode='reflect').reshape(1, -1)  # Resize and flatten
    predicted_species = knn.predict(img_resized)[0]  # Predict species

    print("Predicted Species:", predicted_species)  # Print prediction
    plt.imshow(imread(img_path))  # Display image
    plt.axis("off")  # Hide axes
    plt.title("Predicted: " + predicted_species)  # Set title
    plt.show()  # Show image

# Define paths
train_csv = "train.csv"  # CSV file path
img_folder = "images"  # Image folder path

# Load dataset
X, y, img_paths = load_data(train_csv, img_folder)

# Train model
knn_model, X_test, y_test = train_knn(X, y)

# Evaluate model
evaluate_knn(knn_model, X_test, y_test)

# Classify an image
classify_leaf(knn_model, img_folder)
