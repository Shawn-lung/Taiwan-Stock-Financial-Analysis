# Taiwan Stock Financial Analysis System

A comprehensive system for financial data collection, analysis, and stock valuation with a focus on Taiwan stock market.

## Overview

This project provides a suite of tools for financial analysis and stock valuation with special enhancements for Taiwan stocks. It includes:

- Background data collection with rate limiting (respects API constraints)
- Industry-specific valuation models using machine learning
- DCF (Discounted Cash Flow) models with ML and deep learning enhancements
- Data visualization and analysis tools

## Installation

### Prerequisites
- Python 3.8 or higher
- A FinMind API key (for Taiwan stock data collection)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Finance_stuff.git
   cd Finance_stuff
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API keys:
   ```
   FINMIND_TOKEN=your_finmind_token_here
   ```

## System Components

### Data Collection

- `background_data_collector.py`: Collects financial data in the background with rate limiting (300 calls per hour)
- `industry_data_collector.py`: Collects industry-specific financial data for model training

### Valuation Models

- `dcf_model.py`: Discounted Cash Flow model for stock valuation
- `industry_valuation_model.py`: ML-based industry-specific valuation adjustments
- `ml_predictor.py`: Machine learning model for growth prediction
- `deep_learning.py`: Deep learning models for financial forecasting

### Integration

- `dcf_integrator.py`: Integrates all models (DCF, ML, DL, and industry-specific) for comprehensive valuation
- `train_models_from_db.py`: Trains models using data from the database

## Usage Examples

### Data Collection

Start the background data collector to build your database:

```python
from background_data_collector import BackgroundDataCollector

# Create collector and start it running
collector = BackgroundDataCollector(
    db_path="finance_data.db",
    collection_interval=8  # Check every 8 hours
)

# Start the background collection scheduler
collector.start_scheduler()

# Check collection progress
stats = collector.get_db_stats()
print(f"Stocks with complete data: {stats.get('stocks_with_complete_data', 0)}")
```

### Training Models

Train industry-specific valuation models:

```python
from train_models_from_db import train_industry_models_from_db

# Train models using database data
success = train_industry_models_from_db()
```

### Running Valuations

Run a comprehensive valuation with all model components:

```python
from dcf_integrator import IntegratedValuationModel

# Initialize the model with all components
model = IntegratedValuationModel(use_ml=True, use_dl=True, use_industry=True)

# Run valuation for a specific stock
result = model.run_valuation("2330.TW", "Semiconductors")

# Print results
print(f"\nValuation Results for {result['ticker']}:")
for model_name, price in result['models'].items():
    print(f"  {model_name.replace('_', ' ').title()}: {price:.2f}")
```

## Rate Limiting

Both collectors implement rate limiting to avoid hitting FinMind API limits:

- **Limit**: 300 API calls per hour
- **Behavior**: Automatically pauses when the limit is reached and resumes after the hour window

## Database Structure

The system uses SQLite to store financial data. Key tables include:

- `stock_info`: Basic stock information and industry classification
- `financial_statements`: Income statement data
- `balance_sheets`: Balance sheet data
- `cash_flows`: Cash flow statement data
- `stock_prices`: Historical price data
- `collection_log`: Log of data collection attempts

## ETF Filtering

The system automatically detects and filters out ETFs (Exchange Traded Funds) during data collection, because:

1. ETFs don't have traditional financial statements like companies do
2. DCF valuation models aren't applicable to ETFs
3. Including ETFs would waste API calls and storage space

ETF detection is based on Taiwan market stock code patterns, including:
- Leveraged ETFs (ending with "L" like 00650L)
- Inverse ETFs (ending with "R" like 00651R)
- Standard ETF numbering patterns (006xx, 00[6-9]xx series)
- Famous ETFs like 0050 (Taiwan 50 ETF)

## Notes

- The system is optimized for Taiwan stocks, particularly those listed on TWSE and TPEx
- Industry classification is standardized across different data sources
- Data collection is designed for long-term background operation to build up a comprehensive database

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
