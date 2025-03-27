# Bankruptcy Predictor

This project is a FastAPI-based web service that predicts financial returns based on input CSV files. The service uses a pre-trained XGBoost model to make predictions.

## Features

- Upload a CSV file containing financial data.
- Validate the CSV file for required columns.
- Predict financial returns using the XGBoost model.
- Return predictions in JSON format.

## Requirements

- Python 3.10+
- FastAPI
- Pandas
- XGBoost

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ReynardL/Bankruptcy-predictor.git
    cd Bankruptcy-predictor
    ```

2. Install the required packages:
    ```bash
    uv sync
    ```

3. Place the pre-trained XGBoost model (`model.json`) in the root directory of the project.

## Usage

1. Start the FastAPI server:
    ```bash
    uvicorn App.main:app --reload
    ```

2. Open your browser and navigate to `http://127.0.0.1:8000` to see the welcome message.

3. Use the `/predict` endpoint to upload a CSV file and get predictions.

## API Endpoints

- `GET /`: Returns a welcome message.
- `POST /predict`: Accepts a CSV file and returns predictions.
- [API Documentation](https://app-978501737888.us-central1.run.app/docs)

## CSV File Format

The CSV file should contain the following columns:

- Quick Ratio
- Fixed Assets to Assets
- Interest-bearing debt interest rate
- Total debt/Total net worth
- Borrowing dependency
- ROA(C) before interest and depreciation before interest
- Continuous Net Profit Growth Rate
- Research and development expense rate
- Allocation rate per person
- Revenue per person
- Cash/Current Liability
- Accounts Receivable Turnover
- Quick Assets/Total Assets
- Total income/Total expense
- Net Value Per Share (B)
- Cash Flow to Equity
- Non-industry income and expenditure/revenue
- After-tax Net Profit Growth Rate
- Inventory Turnover Rate (times)
- Total expense/Assets
- Net Value Growth Rate
- Operating Expense Rate
- Total Asset Growth Rate
- Cash Turnover Rate
- Current Liabilities/Liability
- Interest Expense Ratio
- Operating Profit Growth Rate
- Long-term fund suitability ratio (A)
- Cash Flow Per Share
- Average Collection Days

## API Users

- **Target users**: Banks giving out loans, investors.
- **Expected daily request volume**: 5 requests per day.
- **User requirements**: Batch processing.

## Diagram

Below is a text diagram explaining how someone interacts with the service:

1. **User** uploads a CSV file to the FastAPI server via the `/predict` endpoint.
2. **FastAPI Server** receives the CSV file and validates it for the required columns.
3. If the CSV file is valid, the **FastAPI Server** inputs the data to the **XGBoost Model** for prediction.
4. The **XGBoost Model** returns the predictions to the **FastAPI Server**.
5. The **FastAPI Server** sends the predictions back to the **User** in JSON format.

## Performance Metrics

- **Response time**:
  - ~10s between sending API request and receiving predictions (with cold start).
  - ~0.1s between sending API request and receiving predictions (no cold start).
- **Memory consumption**:
  - Minimum: 0
  - Maximum: 43% of 512 = 224MB
- **Cloud monitoring dashboards**:
  - Number of expected requests: ~5 daily