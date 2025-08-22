# 🍃 Leaf Prediction System  

## 📌 Project Overview  
This project is a **Leaf Classification System** that predicts the species of a leaf from its image.  
It uses **Machine Learning models (Random Forest, MLP with PCA & SMOTE)** to classify images and provides a simple **Flask-based web interface** for user interaction.  

Users can:  
- View model evaluation metrics (Accuracy, Precision, Recall, F1-Score, Specificity).  
- Upload or choose a leaf image by ID to classify its species.  
- See the predicted species and corresponding image directly on the web app.  

---

## 🎯 Objectives  
- Preprocess and load leaf images from the dataset.  
- Train and evaluate ML models:  
  - **Random Forest Classifier** (tree-based ensemble method).  
  - **MLP Classifier** with PCA dimensionality reduction & SMOTE oversampling.  
- Integrate models into a **Flask application** with an interactive web interface.  
- Provide accurate predictions along with model performance statistics.  

---

## ⚙️ Methodology  

### 🔹 Data Preprocessing  
- Images resized to `(224x224x3)` for Random Forest.  
- Images resized to `(64x64x3)` for MLP.  
- Labels encoded using **LabelEncoder**.  
- **SMOTE applied** to balance the dataset (for MLP model).  
- **PCA** reduces dimensionality to 100 components (improves MLP performance).  

### 🔹 Models  
1. **Random Forest Classifier**  
   - Trained on image features.  
   - Predictions via **majority voting** across decision trees.  

2. **MLP Classifier** (CNN-like with PCA + SMOTE)  
   - Normalized pixel values.  
   - PCA reduced input dimensionality.  
   - Balanced dataset using SMOTE.  
   - Fully connected MLP trained with 150 hidden neurons.  

### 🔹 Flask Web App  
- Displays evaluation metrics.  
- Lets users input an image number (e.g., `1`, `2`, `3...`).  
- Shows predicted species and the image on the webpage.  

---
## 📂 Project Structure  
Leaf-Prediction/
│── static/
│ └── images/ # Leaf dataset images
│── templates/
│ └── index.html # Web interface
│── train.csv # CSV file with image IDs and labels
│── app.py # Flask app with Random Forest model
│── cnn.py # Flask app with MLP model
│── README.md # Documentation


---

## 📦 Requirements  

Install dependencies using:  
```bash
pip install pandas numpy scikit-learn scikit-image flask imbalanced-learn
```

## 🚀 How to Run

Clone the repository:

git clone https://github.com/your-username/Leaf-Prediction.git
cd Leaf-Prediction


Start the Flask app with Random Forest:

python app.py


Or run with MLP model:

python cnn.py


Open the browser at:

http://127.0.0.1:5000


Use the input box to enter an image number (e.g., 1, 2, 3).

## 📊 Sample Web Interface

Evaluation Metrics Displayed:

Accuracy: 0.87
Precision: 0.85
Recall: 0.83
F1 Score: 0.84
Specificity: 0.88


Classification Result:

Predicted Species: Acer Campestre


Leaf Image Example:


## 🙌 Acknowledgment

This project was inspired by the Leaf Classification Challenge and implemented as part of my learning in Computer Vision + Machine Learning.

## 📝 License

This project is licensed for educational purposes.
## 📂 Project Structure  

