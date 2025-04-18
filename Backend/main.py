from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import io
from xgboost import XGBClassifier
from pydantic import BaseModel
import great_expectations as ge
from great_expectations.core.expectation_suite import ExpectationSuite
from typing import List, Dict
import logging
import shap

# schemas / data contracts
class PredictReturnModel(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    shap_values: List[Dict[str, float]]

# initialize FastAPI
app = FastAPI()

# list of features for model input
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

# initialize model
model = XGBClassifier()
model.load_model("model.json")

# Create a SHAP explainer once at startup
explainer = shap.TreeExplainer(model)

# initialize logger
logger = logging.getLogger("uvicorn.error")

# validation functions
def validate_input(df: pd.DataFrame) -> bool:
    context = ge.get_context()
    expectation_suite_name = "input_suite"

    try:
        suite = context.get_expectation_suite(expectation_suite_name=expectation_suite_name)
        logger.info(f"Loaded ExpectationSuite '{suite.name}' containing {len(suite.expectations)} expectations.")
    except ge.exceptions.DataContextError:
        suite = ExpectationSuite(expectation_suite_name=expectation_suite_name)
        logger.info(f"Created empty ExpectationSuite '{suite.expectation_suite_name}'.")

    batch = ge.from_pandas(df)
    batch.expectation_suite_name = expectation_suite_name

    features_0_1 = [
        'Borrowing dependency',
        'ROA(C) before interest and depreciation before interest',
        'Continuous Net Profit Growth Rate',
        'Current Liabilities/Liability',
        'Interest Expense Ratio',
        'Operating Profit Growth Rate',
        'Long-term fund suitability ratio (A)',
        'Cash Flow Per Share'
    ]
    features_0_1e10 = [
        'Quick Ratio',
        'Fixed Assets to Assets',
        'Interest-bearing debt interest rate',
        'Total debt/Total net worth',
        'Research and development expense rate',
        'Allocation rate per person',
        'Revenue per person',
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
        'Average Collection Days'
    ]

    for feature in final_features:
        batch.expect_column_to_exist(column=feature)
        batch.expect_column_values_to_not_be_null(column=feature)
        batch.expect_column_values_to_be_of_type(column=feature, type_="float")

        if feature in features_0_1:
            batch.expect_column_values_to_be_between(column=feature, min_value=0, max_value=1)
        elif feature in features_0_1e10:
            batch.expect_column_values_to_be_between(column=feature, min_value=0, max_value=1e10)
        else:
            logger.warning(f"Feature {feature} not assigned to a validation group.")

    results = batch.validate(result_format="SUMMARY")
    return results["success"]

def validate_output(predictions: List[int], probabilities: List[float]) -> bool:
    context = ge.get_context()
    expectation_suite_name = "output_suite"

    try:
        suite = context.get_expectation_suite(expectation_suite_name=expectation_suite_name)
        logger.info(f"Loaded ExpectationSuite '{suite.name}' containing {len(suite.expectations)} expectations.")
    except ge.exceptions.DataContextError:
        suite = ExpectationSuite(expectation_suite_name=expectation_suite_name)
        logger.info(f"Created empty ExpectationSuite '{suite.expectation_suite_name}'.")

    df = pd.DataFrame({
        'predictions': predictions,
        'probabilities': probabilities
    })
    
    batch = ge.from_pandas(df)
    batch.expectation_suite_name = expectation_suite_name
    batch.expect_column_to_exist(column="predictions")
    batch.expect_column_values_to_be_in_set(column="predictions", value_set=[0, 1])
    batch.expect_column_to_exist(column="probabilities")
    batch.expect_column_values_to_be_between(column="probabilities", min_value=0, max_value=1)

    results = batch.validate(result_format="SUMMARY")
    return results["success"]

def get_predictions_with_explanations(df: pd.DataFrame):
    try:
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1] 
        
        shap_values = explainer(df)
        
        shap_values_list = []
        for i in range(len(df)):
            sample_dict = {}
            for j, feature in enumerate(final_features):
                sample_dict[feature] = float(shap_values.values[i, j])
            shap_values_list.append(sample_dict)
            
        return predictions.tolist(), probabilities.tolist(), shap_values_list
    except Exception as e:
        logger.exception(f"Internal error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal error during prediction.")

# helper functions
async def preprocess_input(file: UploadFile) -> pd.DataFrame:
    try:
        file_content = await file.read()
        file_io = io.StringIO(file_content.decode("utf-8"))
        df = pd.read_csv(file_io)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format. Please upload a valid CSV file.")
    except Exception as e:
        logger.exception("Error reading CSV file.")
        raise HTTPException(status_code=500, detail="Internal server error while reading the file.")

    df_final = df[final_features]
    try:
        df_final = df_final.astype(float)
    except ValueError:
        raise HTTPException(status_code=400, detail="All input values must be floats.")
    return df_final

# endpoints
@app.post("/predict", response_model=PredictReturnModel)
async def predict(file: UploadFile = File(...)):
    # file and datatype validation
    df_final = await preprocess_input(file)
    
    # ge input values validation
    if not validate_input(df_final):
        raise HTTPException(status_code=400, detail="Input data validation failed.")

    # model prediction with explanations
    predictions_list, probabilities_list, shap_values_list = get_predictions_with_explanations(df_final)

    # ge output values validation
    if not validate_output(predictions_list, probabilities_list):
        raise HTTPException(status_code=500, detail="Output data validation failed.")
    
    return {
        "predictions": predictions_list, 
        "probabilities": probabilities_list,
        "shap_values": shap_values_list
    }
