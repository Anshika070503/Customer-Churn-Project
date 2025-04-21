# 📉 Customer Churn Prediction Web App

This is a machine learning web application that predicts whether a customer is likely to **churn** (leave a service) based on various input features like contract type, internet service, tenure, etc. The model provides a prediction along with a confidence score.

---

## 🚀 Features

- Predict customer churn in real-time using a trained ML model
- Streamlit-powered interactive web interface
- Confidence score displayed with prediction
- Handles categorical and numerical features seamlessly
- Ready for deployment and scalable

---

## 🛠 Tech Stack

- **Language:** Python
- **Libraries:** pandas, scikit-learn, Streamlit, pickle
- **Version Control:** Git & GitHub

---

## 🧠 Machine Learning Model

- Model Used: `Random Forest Classifier` (can be swapped with any other model)
- Preprocessing: One-hot encoding for categorical variables
- Training: Trained on a clean dataset with target label `Churn`

---

## 📦 Files in This Repository

| File | Description |
|------|-------------|
| `app.py` | Streamlit code for the web application |
| `train_model.py` | Script to preprocess data and train the model |
| `model.pkl` | Serialized trained ML model |
| `features.pkl` | List of feature columns used in training |
| `requirements.txt` | All Python dependencies for easy setup |

---

## ▶️ Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/customer-churn-predictor.git
   cd customer-churn-predictor
