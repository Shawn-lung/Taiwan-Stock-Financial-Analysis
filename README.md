# Finance_stuff: DCF Valuation Tool

This project implements a Discounted Cash Flow (DCF) model for intrinsic stock valuation using Python. It retrieves financial data via yfinance, forecasts Free Cash Flow (FCF) using user-defined manual factors, and calculates an intrinsic stock price based on terminal value and discounting.

## Features

- **Automated Data Acquisition:** Fetches income statements, cash flows, and balance sheets using yfinance.
- **Forecasting Flexibility:** Accepts manual input for revenue growth, CAPEX, working capital, depreciation, and tax rate adjustments.
- **DCF Calculation:** Computes the Net Present Value (NPV) of forecasted FCF and terminal value to derive the intrinsic equity value.
- **Interactive UI:** A Streamlit app allows you to adjust parameters and visualize results interactively.
- **Unit Testing:** Includes unit tests using pytest to ensure model reliability.
- **Machine Learning-based growth prediction using Random Forest**
- **Sensitivity analysis with interactive visualization**
- **Historical financial data analysis**

## Prerequisites

- Python 3.8 or later
- Required libraries:
  - `yfinance`
  - `pandas`
  - `numpy`
  - `streamlit`
  - `pytest`

Install dependencies using pip:

```bash
pip install yfinance pandas numpy streamlit pytest
```

## Additional Requirements
```bash
pip install scikit-learn seaborn matplotlib
```

## API Key Setup

This tool uses multiple financial data sources for better accuracy. You'll need to set up API keys:

1. Alpha Vantage API (Free)
   - Visit https://www.alphavantage.co/support/#api-key
   - Sign up for a free API key
   - Set environment variable: `ALPHA_VANTAGE_KEY`

2. Financial Modeling Prep API (Optional)
   - Visit https://financialmodelingprep.com/developer
   - Sign up for an API key
   - Set environment variable: `FMP_KEY`

You can set the environment variables in your terminal:
```bash
# For Windows (PowerShell)
$env:ALPHA_VANTAGE_KEY="your_key_here"
$env:FMP_KEY="your_key_here"

# For Linux/Mac
export ALPHA_VANTAGE_KEY="your_key_here"
export FMP_KEY="your_key_here"
```

Or create a `.env` file in the project root:
```plaintext
ALPHA_VANTAGE_KEY=your_key_here
FMP_KEY=your_key_here
```
