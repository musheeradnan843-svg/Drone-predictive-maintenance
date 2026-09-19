from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# -----------------------------
# Load saved models and objects
# -----------------------------

xgb_model = joblib.load("xgb_failure_model.joblib")
iso_model = joblib.load("isolation_forest_model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("onehot_encoder.joblib")


# -----------------------------
# Create FastAPI app
# -----------------------------

app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine failure and anomaly detection API",
    version="1.0"
)


# -----------------------------
# Input schema
# -----------------------------

class MachineInput(BaseModel):

    Type: str
    Air_Temperature: float
    Process_Temperature: float
    Rotational_Speed: float
    Torque: float
    Tool_Wear: float


# -----------------------------
# Prediction endpoint
# -----------------------------

@app.post("/predict")
def predict_machine(data: MachineInput):

    # Convert input into DataFrame
    input_df = pd.DataFrame([{
        "Type": data.Type,
        "Air temperature [K]": data.Air_Temperature,
        "Process temperature [K]": data.Process_Temperature,
        "Rotational speed [rpm]": data.Rotational_Speed,
        "Torque [Nm]": data.Torque,
        "Tool wear [min]": data.Tool_Wear
    }])


    # -----------------------------
    # One-Hot Encoding
    # -----------------------------

    type_encoded = encoder.transform(
        input_df[["Type"]]
    )

    type_encoded_df = pd.DataFrame(
        type_encoded,
        columns=encoder.get_feature_names_out(["Type"])
    )


    # -----------------------------
    # Scale numerical features
    # -----------------------------

    numerical_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    numerical_scaled = scaler.transform(
        input_df[numerical_columns]
    )

    numerical_scaled_df = pd.DataFrame(
        numerical_scaled,
        columns=[
        "Air temperature K",
        "Process temperature K",
        "Rotational speed rpm",
        "Torque Nm",
        "Tool wear min"
    ]
)
    


    # -----------------------------
    # Final input for models
    # -----------------------------

    final_input = pd.concat(
        [type_encoded_df, numerical_scaled_df],
        axis=1
    )


    # -----------------------------
    # XGBoost prediction
    # -----------------------------

    failure_prediction = xgb_model.predict(final_input)[0]


    # -----------------------------
    # Isolation Forest prediction
    # -----------------------------

    anomaly_prediction = iso_model.predict(final_input)[0]

    anomaly_score = iso_model.decision_function(final_input)[0]


    # -----------------------------
    # Convert results
    # -----------------------------

    failure_result = "Failure" if failure_prediction == 1 else "No Failure"

    anomaly_result = "Anomaly" if anomaly_prediction == -1 else "Normal"


    # -----------------------------
    # API response
    # -----------------------------

    return {
        "failure_prediction": int(failure_prediction),
        "failure_status": failure_result,
        "anomaly_prediction": int(anomaly_prediction),
        "anomaly_status": anomaly_result,
        "anomaly_score": float(anomaly_score)
    }
