# 🏡 California Housing ML Project

A comprehensive machine learning project on the **California Housing Dataset**, covering the full ML pipeline — from data preprocessing and EDA to regression, classification, SVM, neural networks, clustering, and a live Flask web app for predictions.

---

## 📌 Project Overview

This project explores multiple supervised and unsupervised learning techniques to predict California housing prices and classify them into price tiers (Low / Medium / High). A Flask web application ties it all together, serving real-time predictions from the trained models.

---

## 📁 Project Structure

```
California_Housing_ML_Project/
├── notebook/
│   ├── 01_data_preprocessing_eda.ipynb       # EDA, feature scaling, train/val/test split
│   ├── 02_regression_models.ipynb            # Linear & Multiple Linear Regression
│   ├── 03_classification_models.ipynb        # Logistic Regression, Decision Tree, Random Forest
│   ├── 04_svm_model.ipynb                    # SVM with multiple kernels
│   ├── 05_neural_network.ipynb               # Neural Network classifier
│   └── 06_clustering_pca.ipynb               # KMeans Clustering & PCA
├── models/
│   ├── regression_model.pkl                  # Best regression model
│   ├── classifier_model.pkl                  # Best classification model (Random Forest)
│   ├── svm_model.pkl                         # Best SVM model
│   ├── neural_network_model.keras            # Trained neural network
│   ├── scaler.pkl                            # Feature scaler
│   └── processed_data.pkl                    # Preprocessed dataset
├── web_app/
│   ├── app.py                                # Flask application
│   ├── templates/
│   │   └── index.html                        # Frontend UI
│   └── static/
│       └── style.css                         # Stylesheet
└── requirements.txt
```

---

## 🧠 ML Techniques Covered

| Notebook | Technique | Goal |
|---|---|---|
| 01 | Data Preprocessing & EDA | 70/15/15 split, feature scaling, visualizations |
| 02 | Regression | Predict house price (continuous) using Linear & Multiple Linear Regression |
| 03 | Classification | Classify price tier (Low/Medium/High) using Logistic Regression, Decision Tree, Random Forest |
| 04 | SVM | SVM classification with multiple kernels; compared against Random Forest |
| 05 | Neural Network | 3-class neural network classifier with early stopping |
| 06 | Clustering & PCA | KMeans clustering with elbow method; PCA for dimensionality reduction & visualization |

---

## 🌐 Web Application

The Flask app takes 8 housing features as input and returns three simultaneous predictions:

- **Estimated Market Value** — from the regression model (continuous price)
- **Price Tier (Random Forest)** — Low / Medium / High classification
- **Price Tier (SVM)** — Low / Medium / High classification

### Input Features

| Feature | Description |
|---|---|
| `MedInc` | Median income of the block group |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/California_Housing_ML_Project.git
cd California_Housing_ML_Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebooks

Open the notebooks in order inside the `notebook/` directory using Jupyter:

```bash
jupyter notebook
```

### 4. Launch the web app

```bash
cd web_app
python app.py
```

Then visit `http://localhost:5000` in your browser.

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Scikit-learn** — ML models (Regression, Classification, SVM, KMeans, PCA)
- **TensorFlow / Keras** — Neural network
- **Flask** — Web application framework
- **Pandas & NumPy** — Data manipulation
- **Matplotlib & Seaborn** — Visualizations
- **Joblib** — Model serialization

---

## 📊 Dataset

The **California Housing Dataset** is sourced from `sklearn.datasets.fetch_california_housing`. It contains block-group-level data from the 1990 California census, with ~20,000 samples and 8 features. The target variable is the median house value (in units of $100,000).

---

## 📈 Results Summary

| Model | Task | Metric |
|---|---|---|
| Multiple Linear Regression | Price prediction | MSE & R² on test set |
| Random Forest Classifier | Price tier | Accuracy on test set |
| SVM (best kernel) | Price tier | Accuracy on test set |
| Neural Network | Price tier | Accuracy with early stopping |
| KMeans + PCA | Clustering | Elbow method + visual cluster analysis |

> Detailed results and plots are available inside each notebook.

---

## 👤 Author

**Satwik Gupta**  
Roll Number: 2305075 | Section: CSE - 6
