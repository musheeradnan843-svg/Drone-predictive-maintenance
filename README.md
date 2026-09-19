# Drone Predictive Maintenance

A machine learning based predictive maintenance system that predicts machine failure and detects unusual machine behavior using industrial sensor data.

The project uses XGBoost for machine failure prediction and Isolation Forest for anomaly detection. A FastAPI backend is used to serve the trained models through an API.

## Project Overview

Predictive maintenance helps identify potential machine problems before they result in major failures.

This project uses machine sensor information such as:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

The system performs two tasks:

1. Predicts whether a machine failure is likely to occur.
2. Detects unusual machine behavior using anomaly detection.

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

Dataset features used in this project:

- Type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

Target:

- Machine failure

The original dataset also contains failure-mode related columns that were not used as model input.

## Machine Learning

### Failure Prediction

Several classification models were evaluated, including:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine

XGBoost was selected as the final failure prediction model and further tuned using GridSearchCV.

### Anomaly Detection

Isolation Forest was used for unsupervised anomaly detection.

Unlike the failure prediction model, Isolation Forest does not use the machine failure target during training. It identifies observations that have unusual combinations of sensor values.

The anomaly detection output includes:

- Normal
- Anomaly
- Anomaly Score

## Data Preprocessing

The preprocessing pipeline includes:

- One-hot encoding for the machine Type feature
- StandardScaler for numerical features
- Train-test split with stratification
- Reusing the same preprocessing objects during API prediction

The trained preprocessing objects are saved using Joblib.

## FastAPI

The trained models are integrated into a FastAPI application.

The API accepts machine sensor values and returns:

- Failure prediction
- Failure status
- Anomaly prediction
- Anomaly status
- Anomaly score

### API Flow

```text
Client
   ↓
FastAPI /predict
   ↓
Input Validation
   ↓
Data Preprocessing
   ↓
XGBoost + Isolation Forest
   ↓
Prediction Results
   ↓
JSON Response
