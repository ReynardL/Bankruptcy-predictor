from fastapi import FastAPI, File, UploadFile
import pandas as pd
import io
from xgboost import XGBClassifier
from pydantic import BaseModel
from typing import List

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

@app.get("/")
async def get_root():
    return {"message": "Welcome to Cloud Run"}

@app.post("/predict", response_model=PredictReturnModel)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    missing = set(final_features) - set(df.columns)
    if missing:
        return {"error": f"Missing columns in CSV: {missing}"}

    df_final = df[final_features]
    predictions = model.predict(df_final)
    return {"predictions": predictions.tolist()}
