import streamlit as st
import pandas as pd
import requests
import io
import os
import plotly.graph_objects as go
import plotly.express as px

api_url = os.environ.get("GCP_API_URL", "http://localhost:8080/predict")

# initialization
st.set_page_config(
    page_title="Bankruptcy Prediction System",
    page_icon="🏦",
    layout="wide"
)
st.title("Bankruptcy Prediction System")

required_columns = [
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

# helper functions

def categorize_confidence(probability):
    if probability < 0.2:
        return "Very Low Risk"
    elif probability < 0.4:
        return "Low Risk"
    elif probability < 0.6:
        return "Moderate Risk"
    elif probability < 0.8:
        return "High Risk"
    else:
        return "Very High Risk"

def display_sample_format():
    st.markdown("### Sample Format")
    sample_df = pd.DataFrame(columns=required_columns)
    sample_df.loc[0] = [0.1] * len(required_columns)
    st.dataframe(sample_df.head(1))

def predict_bankruptcy(file_content):
    try:
        files = {'file': ('data.csv', file_content, 'text/csv')}
        response = requests.post(api_url, files=files)
            
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: API returned status code {response.status_code}. Details: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def display_sample_explanation(idx):
    st.subheader(f"Row {idx}")
    
    pred = df['Bankruptcy Prediction'][idx]
    prob = df['Prediction Confidence'][idx]
    shap = df['Shap Value'][idx]
    risk = df['Risk Category'][idx]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", "Bankrupt" if pred == 1 else "Healthy")
        st.metric("Risk Category", risk)
    with col2:
        st.metric("Prediction Confidence", f"{prob:.2f}%")
    
    sorted_shap = sorted(shap.items(), key=lambda x: abs(x[1]))
    
    features = [item[0] for item in sorted_shap]
    values = [item[1] for item in sorted_shap]
    pos_features = [f for f, v in zip(features, values) if v > 0]
    pos_values = [v for v in values if v > 0]
    neg_features = [f for f, v in zip(features, values) if v <= 0]
    neg_values = [v for v in values if v <= 0]
    
    fig = go.Figure()
    if neg_features:
        fig.add_trace(go.Bar(
            y=neg_features,
            x=neg_values,
            orientation='h',
            marker_color='blue',
            name='Decreases Bankruptcy Risk',
            text=[f"{abs(v):.4f}" for v in neg_values],
            textposition='auto',
            showlegend=True
        ))
    if pos_features:
        fig.add_trace(go.Bar(
            y=pos_features,
            x=pos_values,
            orientation='h',
            marker_color='red',
            name='Increases Bankruptcy Risk',
            text=[f"{abs(v):.4f}" for v in pos_values],
            textposition='auto',
            showlegend=True
        ))
    
    fig.update_layout(
        title="Top Feature Contributions",
        xaxis_title='SHAP Value (Impact on Prediction)',
        yaxis=dict(
            title='Features',
            categoryorder='array',
            categoryarray=features
        ),
        height=750,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_confidence_distribution(df):
    df['Risk Category'] = df['Bankruptcy Confidence'].apply(lambda p: categorize_confidence(p))
    
    category_counts = df['Risk Category'].value_counts().reset_index()
    category_counts.columns = ['Risk Category', 'Count']
    
    all_categories = ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']
    for category in all_categories:
        if category not in category_counts['Risk Category'].values:
            category_counts = pd.concat([category_counts, pd.DataFrame({'Risk Category': [category], 'Count': [0]})])
    
    category_order = {cat: i for i, cat in enumerate(all_categories)}
    category_counts['Order'] = category_counts['Risk Category'].map(category_order)
    category_counts = category_counts.sort_values('Order')
    
    higher_risk_categories = ['Moderate Risk', 'High Risk', 'Very High Risk']
    higher_risk_df = category_counts[category_counts['Risk Category'].isin(higher_risk_categories)]

    color_map = {
        'Very Low Risk': 'green',
        'Low Risk': 'lightgreen',
        'Moderate Risk': 'yellow',
        'High Risk': 'orange',
        'Very High Risk': 'red'
    }
    
    # Create visualizations
    
    fig1 = px.bar(
        category_counts, 
        x='Risk Category', 
        y='Count',
        title='Distribution of Risk Categories',
        color='Risk Category',
        color_discrete_map=color_map
    )

    fig1.update_layout(
        xaxis=dict(
            categoryorder='array',
            categoryarray=all_categories
        ),
        height=400
    )
    
    fig2 = px.bar(
        category_counts, 
        x='Risk Category', 
        y='Count',
        title='Distribution of Risk Categories (Log Scale)',
        color='Risk Category',
        color_discrete_map=color_map,
        log_y=True
    )
    
    fig2.update_layout(
        xaxis=dict(
            categoryorder='array',
            categoryarray=all_categories
        ),
        height=400
    )
    
    fig3 = px.bar(
        higher_risk_df, 
        x='Risk Category', 
        y='Count',
        title='Distribution of Higher Risk Categories',
        color='Risk Category',
        color_discrete_map={cat: color_map[cat] for cat in higher_risk_categories}
    )
    
    fig3.update_layout(
        xaxis=dict(
            categoryorder='array',
            categoryarray=higher_risk_categories
        ),
        height=400
    )
    
    return {"full": fig1, "log_scale": fig2, "high_risk_focus": fig3}

def display_predictions(df, predictions):
    df['Bankruptcy Prediction'] = predictions['predictions']
    df['Bankruptcy Confidence'] = predictions['probabilities']
    df['Risk Category'] = [categorize_confidence(p) for p in df['Bankruptcy Confidence']]
    df['Prediction Confidence'] = df['Bankruptcy Confidence'].apply(lambda p: max(p, 1-p) * 100)
    df['Shap Value'] = predictions['shap_values']
    
    # Prediction Results
    st.subheader("Prediction Results")
    
    total_predictions = len(predictions['predictions'])
    bankruptcy_count = sum(predictions['predictions'])
    healthy_count = total_predictions - bankruptcy_count
    avg_confidence = df['Prediction Confidence'].mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Companies Analyzed", total_predictions)
        st.metric("Average Prediction Confidence", f"{avg_confidence:.2f}%")
        
    with col2:
        st.metric("Companies Predicted Healthy", healthy_count)
        st.metric("Companies Predicted Bankrupt", bankruptcy_count)
    
    risk_figs = display_confidence_distribution(df)
    
    # separate tabs
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Standard Distribution", "Log Scale View", "High Risk Focus"])
    
    with viz_tab1:
        st.plotly_chart(risk_figs["full"], use_container_width=True)
    
    with viz_tab2:
        st.plotly_chart(risk_figs["log_scale"], use_container_width=True)

    with viz_tab3:
        st.plotly_chart(risk_figs["high_risk_focus"], use_container_width=True)
    
    # Individual Company Explanations
    st.subheader("Individual Company Explanations")

    risk_filter = st.selectbox(
        "Filter by risk category:",
        options=['All'] + ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'],
    )
    
    if risk_filter == 'All':
        filtered_indices = list(range(len(df)))
    else:
        filtered_indices = [i for i, cat in enumerate(df['Risk Category']) if cat == risk_filter]
    
    if filtered_indices:
        format_func = lambda i: f"Row {i} - {'Bankrupt' if predictions['predictions'][i] == 1 else 'Healthy'} ({df['Prediction Confidence'][i]:.2f}%) - {df['Risk Category'][i]}"
        
        sorted_indices = sorted(filtered_indices)
        
        selected_row = st.selectbox(
            "Select a company to view detailed explanation:",
            options=sorted_indices,
            format_func=format_func
        )
        
        display_sample_explanation(selected_row)
    else:
        st.warning(f"No companies found in the '{risk_filter}' category.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This application uses an XGBoost machine learning model to predict the likelihood 
    of bankruptcy for companies based on their financial indicators. The model was 
    trained on historical bankruptcy data.
    """)
    
    st.header("Instructions")
    st.info("""
    1. Upload a CSV file containing the required financial indicators
    2. Examine the results
    
    **Understanding the Results:**
    - **Bankruptcy Prediction**: 0 = Healthy, 1 = Bankrupt
    - **Prediction Confidence**: Confidence score for the bankruptcy prediction (0-100%)
    - **Risk Category**: Risk level based on confidence:
        - Very Low Risk: 0-20%
        - Low Risk: 20-40%
        - Moderate Risk: 40-60%
        - High Risk: 60-80%
        - Very High Risk: 80-100%
    - **SHAP Values**: Explain how each feature impacts each prediction
    """)

# Main app layout

st.header("Upload Financial Data")
uploaded_file = st.file_uploader("Upload CSV File", type="csv")

with st.expander("View expected CSV format"):
    display_sample_format()

if uploaded_file is not None:
    try:
        file_content = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(file_content))
        
        with st.spinner("Processing data and making predictions..."):
            predictions = predict_bankruptcy(file_content.decode('utf-8'))
            
            if predictions:
                df = df[required_columns]

                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                display_predictions(df, predictions)
    except pd.errors.ParserError:
        st.error("Invalid CSV file. Please upload a properly formatted CSV file.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a CSV file to get bankruptcy predictions.")
