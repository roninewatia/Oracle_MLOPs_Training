from fastapi import FastAPI, HTTPException
from typing import Any
import mlflow
import mlflow.sklearn
import pandas as pd
import os

app = FastAPI(
        title = "Customer Churn Prediction Fast API",
        description = "Customer Churn Prediction Fast API",
        version = "1.0.0"
)

# Connect to mlflow tracking server
#mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load registered model version
#MODEL_URI = "models:/LogisticRegressionModelRegistered./1"

# 1. Connect to MLflow Tracking Server

mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )
)

# 2. Load Registered ML Model

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/LogisticRegressionModelRegistered./4"
)
model = mlflow.sklearn.load_model(MODEL_URI)

# get the feature names
FEATURE_NAMES = list(model.feature_names_in_)

@app.get("/")
def home():
    return {
        "message" : "Customer CHurn Prediction FASTAPI is running" 
    }
    
@app.get("/health")
def health():
    return {
        "status" : "healthy"
    }
    
@app.post("/predict")
def predict(customer_data: dict[str, Any]):

    try:
        
        # Check for missing and unexpected col
        missing = [
            col for col in FEATURE_NAMES
            if col not in customer_data
        ]

        extra = [
            col for col in customer_data
            if col not in FEATURE_NAMES
        ]

        if missing or extra:
            raise HTTPException(
                status_code = 422,
                detail = {
                     "missing_features" : missing,
                     "extra_features" : extra
             })

        # convert JSON into DF
        input_df = pd.DataFrame(
            [customer_data],
            columns = FEATURE_NAMES
        )

        # Generate prediction
        prediction = model.predict(input_df)[0]

        # Generate probability
        prob = model.predict_proba(input_df)[0]

        classes = list(model.classes_)
        churn_index = classes.index(1)
        churn_prob = float(prob[churn_index])

        # return result
        return {
                       "prediction" : str(prediction),
                       "churn_prob" : round(churn_prob, 4) 
        }

    except Exception as e:
        return HTTPException (
            status_code = 500,
            detail = str(e)
        )


