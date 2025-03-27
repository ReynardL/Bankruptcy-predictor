from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import io
from xgboost import XGBClassifier
from pydantic import BaseModel
from typing import List
import logging

class PredictReturnModel(BaseModel):
    predictions: List[int]

app = FastAPI()

final_features = [
    'Quick Ratio',
    'Fixed Assets to Assets',
    'Interest-bearing debt interest rate',
    'Total debt/Total net worth',
    'Borrowing dependency',
    'ROA(C) before interest and depreciation before interest',
    'Continuous Net Profit Growth Rate',
    'Research and development expense rate',
    'Allocation rate per person',
    'Revenue per person',
    'Cash/Current Liability',
    'Accounts Receivable Turnover',
    'Quick Assets/Total Assets',
    'Total income/Total expense',
    'Net Value Per Share (B)',
    'Cash Flow to Equity',
    'Non-industry income and expenditure/revenue',
    'After-tax Net Profit Growth Rate',
    'Inventory Turnover Rate (times)',
    'Total expense/Assets',
    'Net Value Growth Rate',
    'Operating Expense Rate',
    'Total Asset Growth Rate',
    'Cash Turnover Rate',
    'Current Liabilities/Liability',
    'Interest Expense Ratio',
    'Operating Profit Growth Rate',
    'Long-term fund suitability ratio (A)',
    'Cash Flow Per Share',
    'Average Collection Days'
]

model = XGBClassifier()
model.load_model("model.json")

logger = logging.getLogger("uvicorn.error")

@app.get("/")
async def get_root():
    return {"message": "Welcome to Cloud Run"}

@app.post("/predict", response_model=PredictReturnModel)
async def predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.StringIO((await file.read()).decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid CSV format. Expected a comma-delimited CSV with header row.")
    
    missing = set(final_features) - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV is missing the following columns: {', '.join(missing)}")
    
    df_final = df[final_features]
    
    if df_final.isnull().values.any():
        raise HTTPException(status_code=400, detail="Input value cannot be null.")
    
    try:
        df_final = df_final.astype(float)
    except ValueError:
        raise HTTPException(status_code=400, detail="All input values must be floats.")
    
    if (df_final < 0).any().any():
        raise HTTPException(status_code=400, detail="All values must be non-negative.")

    try:
        predictions = model.predict(df_final)
    except Exception as e:
        logger.exception("Internal error during prediction.")
        raise HTTPException(status_code=500, detail="Internal error during prediction.")
    
    return {"predictions": predictions.tolist()}
