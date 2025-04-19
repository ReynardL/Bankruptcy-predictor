# Bankruptcy Predictor

This project is a web application that predicts financial returns based on input CSV files. It consists of a FastAPI-based backend for predictions and a Streamlit-based frontend for user interaction. The service uses a pre-trained XGBoost model to make predictions.

## Features

- User-friendly Streamlit frontend for uploading CSV files and viewing results.
- FastAPI backend for handling prediction requests.
- Upload a CSV file containing financial data.
- Validate the CSV file for required columns.
- Predict financial returns using the XGBoost model.
- Return predictions in JSON format.

## Requirements

- Python 3.10+
- FastAPI
- Streamlit
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

## Usage

### Run both frontend and backend with Docker Compose

```bash
docker-compose up
```

This will start both the FastAPI backend and the Streamlit frontend.

### Start the backend manually

```bash
uvicorn App.main:app --reload
```

### Start the frontend manually

If you want to start the frontend manually, run:

```bash
streamlit run app.py
```

After starting, open your browser and navigate to `http://localhost:8501` to use the web interface, or to `http://127.0.0.1:8000` to see the FastAPI welcome message.

You can use the `/predict` endpoint to upload a CSV file and get predictions via API, or use the Streamlit frontend for a graphical interface.

## Website

You can access the deployed website at:  
[https://app-978501737888.us-central1.run.app/](https://app-978501737888.us-central1.run.app/)

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

Below is a text diagram explaining how someone interacts with the API service:

1. **User** uploads a CSV file to the FastAPI server via the `/predict` endpoint.
2. **FastAPI Server** receives the CSV file and validates it for the required columns.
3. If the CSV file is valid, the **FastAPI Server** inputs the data to the **XGBoost Model** for prediction.
4. The **XGBoost Model** returns the predictions to the **FastAPI Server**.
5. The **FastAPI Server** sends the predictions back to the **User** in JSON format.

## Performance Metrics (API)

- **Response time**:
  - ~10s between sending API request and receiving predictions (with cold start).
  - ~0.1s between sending API request and receiving predictions (no cold start).
- **Memory consumption**:
  - Minimum: 0
  - Maximum: 43% of 512 = 224MB
- **Cloud monitoring dashboards**:
  - Number of expected requests: ~5 daily