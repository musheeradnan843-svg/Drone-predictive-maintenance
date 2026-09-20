# Drone Predictive Maintenance

A machine learning based predictive maintenance system that predicts machine failure and detects unusual machine behavior using industrial sensor data.

The project uses **XGBoost** for machine failure prediction and **Isolation Forest** for anomaly detection. A **FastAPI** backend serves the trained models through an API, **PostgreSQL** stores prediction history for monitoring, and an interactive frontend built with **HTML, CSS, and JavaScript** allows users to interact with the system.

---

## Project Context

The project is designed around a **drone-assisted predictive maintenance** concept for monitoring industrial machines.

The current prototype uses **industrial machine sensor data** rather than drone imagery or drone telemetry. The "drone-assisted" concept represents the intended application context for machine monitoring.

The current system focuses on sensor-based predictive maintenance using machine operating data such as temperature, rotational speed, torque, and tool wear.

In a future version, drone telemetry or inspection data could be integrated with the maintenance system.

---

## Project Overview

Predictive maintenance aims to identify potential machine problems before they result in major failures.

This project uses machine sensor information such as:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

The system performs two main tasks:

1. **Machine Failure Prediction**
   - Predicts whether a machine is likely to fail.
   - Uses XGBoost classification.

2. **Anomaly Detection**
   - Detects unusual machine operating behavior.
   - Uses Isolation Forest.
   - Provides an anomaly score to indicate how unusual the machine's operating condition is.

---

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

### Features Used

- Type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

### Target

- Machine failure

The original dataset also contains failure-mode related columns such as:

- TWF
- HDF
- PWF
- OSF
- RNF

These failure-mode columns were not used as model input features.

---

## Machine Learning

### Failure Prediction

Several classification models were evaluated during the project:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine

XGBoost was selected as the final machine failure prediction model and further tuned using **GridSearchCV**.

The final XGBoost model predicts:

```text
0 → No Failure
1 → Failure

                         USER
                           |
                           v
                HTML + CSS + JavaScript
                           |
                           | Sensor Input
                           v
                     JavaScript
                           |
                           | POST /predict
                           v
                   FASTAPI BACKEND
                           |
                           v
                  INPUT VALIDATION
                           |
                           v
                 DATA PREPROCESSING
                           |
              +------------+------------+
              |                         |
              v                         v
       OneHotEncoder              StandardScaler
              |                         |
              +------------+------------+
                           |
                           v
                    PROCESSED INPUT
                           |
              +------------+------------+
              |                         |
              v                         v
           XGBoost              Isolation Forest
              |                         |
              v                         v
      Machine Failure            Anomaly Detection
       Prediction                      |
              |                         v
              |                   Anomaly Score
              |                         |
              +------------+------------+
                           |
                           v
                  PREDICTION RESULTS
                           |
                           v
                     PostgreSQL
                  Prediction History
                           |
                           v
                     JSON Response
                           |
                           v
                      JavaScript
                           |
                           v
                   FRONTEND DISPLAY
                           |
                           v
                         USER
