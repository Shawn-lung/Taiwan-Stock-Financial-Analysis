import yfinance as yf
import pandas as pd
import requests
import json
from FinMind.data import DataLoader
from typing import Dict, Optional, List
import logging
from pathlib import Path
import os
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class FinancialDataFetcher:
    def __init__(self, cache_dir: str = "data_cache"):
        """Initialize FinMind data loader."""
        self.finmind = DataLoader()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def clear_cache(self, symbol: str = None):
        """Clear cache for specific symbol or all cached data."""
        try:
            if symbol:
                cache_file = self.cache_dir / f"{symbol}_financial_data.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                    logger.info(f"Cleared cache for {symbol}")
            else:
                for file in self.cache_dir.glob("*_financial_data.pkl"):
                    file.unlink()
                logger.info("Cleared all cached data")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def _convert_symbol(self, symbol: str, provider: str) -> str:
        """Convert stock symbol to provider-specific format."""
        # Remove any existing suffix
        base_symbol = symbol.split('.')[0]
        
        # Handle Taiwan stocks
        is_taiwan = '.TW' in symbol or '.TWO' in symbol
        
        if provider == 'finmind':
            if is_taiwan:
                return base_symbol  # FinMind uses just the number for TW stocks
            return symbol
        else:  # yfinance
            if is_taiwan:
                return f"{base_symbol}.TW"
            return symbol

    def inspect_data_source(self, data: Dict[str, pd.DataFrame]) -> str:
        """Determine the likely source of the financial data based on its structure."""
        try:
            if not data or 'income_statement' not in data:
                return "Unknown"
            
            df = data['income_statement']
            if df.empty:
                return "Unknown"
            
            # Update source detection logic
            required_metrics = ['Total Revenue', 'Operating Income', 'Net Income']
            if all(metric in df.index for metric in required_metrics):
                return "FinMind" if len(df.columns) >= 10 else "yfinance"
            
            return "Unknown"
        except Exception as e:
            logger.error(f"Error inspecting data source: {e}")
            return "Error"

    def print_data_summary(self, data: Dict[str, pd.DataFrame]):
        """Print a summary of the financial data."""
        try:
            source = self.inspect_data_source(data)
            logger.info(f"\n=== Financial Data Summary ===")
            logger.info(f"Data Source: {source}")
            
            for key, df in data.items():
                if df is not None and not df.empty:
                    logger.info(f"\n{key.upper()}:")
                    dates = sorted(df.columns)
                    logger.info(f"Years available: {len(dates)}")
                    logger.info(f"Date range: {dates[0]} to {dates[-1]}")
                    logger.info(f"Available metrics: {len(df.index)}")
                    if key == 'income_statement':
                        revenue_key = 'Total Revenue'
                        if revenue_key in df.index:
                            logger.info(f"Revenue data:")
                            revenue_series = df.loc[revenue_key]
                            revenue_series.name = revenue_key
                            logger.info(revenue_series)
        except Exception as e:
            logger.error(f"Error printing data summary: {e}")

    def get_financial_data(self, symbol: str, years: int = 10, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Get financial data from available sources."""
        if force_refresh:
            logger.info(f"Force refreshing data for {symbol}")
            self.clear_cache(symbol)

        # Check cache first
        cache_file = self.cache_dir / f"{symbol}_financial_data.pkl"
        if not force_refresh and cache_file.exists():
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_time < timedelta(days=1):
                logger.info(f"Using cached data for {symbol}")
                return pd.read_pickle(cache_file)

        # Handle Taiwan stocks with FinMind
        if '.TW' in symbol or '.TWO' in symbol:
            base_symbol = symbol.split('.')[0]
            logger.info(f"Trying FinMind with symbol: {base_symbol}")
            data = self._get_finmind_data(base_symbol)
            if self._validate_data(data):
                logger.info("Successfully retrieved data from FinMind")
                self.print_data_summary(data)
                self._cache_data(symbol, data)
                return data

        # Use yfinance as fallback
        logger.info(f"Using yfinance as fallback with symbol: {symbol}")
        data = self._get_yfinance_data(symbol)
        if self._validate_data(data):
            logger.info("Successfully retrieved data from yfinance")
            self.print_data_summary(data)
            self._cache_data(symbol, data)
            return data

        logger.error("Could not retrieve valid data from any source")
        return {}

    def _validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate the data quality."""
        if not data:
            return False

        try:
            for key, df in data.items():
                if df.empty:
                    logger.warning(f"Empty dataframe for {key}")
                    return False

                # Log structure
                logger.info(f"\nValidating {key}:")
                if isinstance(df.columns[0], (datetime, pd.Timestamp)):
                    date_str = [d.strftime('%Y-%m-%d') for d in df.columns]
                else:
                    date_str = [str(d) for d in df.columns]
                logger.info(f"Columns: {date_str}")
                logger.info(f"Metrics: {df.index.tolist()}")

                # Check required metrics
                if key == 'income_statement':
                    required = ['Total Revenue', 'Operating Income', 'Net Income']
                    found = [m for m in required if m in df.index]
                    if not found:
                        logger.warning(f"Missing required metrics: {required}")
                        return False
                    logger.info(f"Found required metrics: {found}")

            return True
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False

    def _cache_data(self, symbol: str, data: Dict[str, pd.DataFrame]):
        """Cache the data if valid."""
        try:
            cache_file = self.cache_dir / f"{symbol}_financial_data.pkl"
            pd.to_pickle(data, cache_file)
            logger.info(f"Cached data for {symbol}")
        except Exception as e:
            logger.error(f"Error caching data: {e}")

    def _get_yfinance_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get financial data from yfinance as fallback."""
        try:
            stock = yf.Ticker(symbol)
            
            # Get the financial statements
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            
            # Ensure proper orientation (dates in columns)
            if not financials.empty and isinstance(financials.index[0], str):
                financials = financials.T
            if not balance_sheet.empty and isinstance(balance_sheet.index[0], str):
                balance_sheet = balance_sheet.T
            if not cashflow.empty and isinstance(cashflow.index[0], str):
                cashflow = cashflow.T
            
            logger.info(f"Fetched yfinance data: {len(financials)} years of financial statements")
            
            return {
                'income_statement': financials,
                'balance_sheet': balance_sheet,
                'cash_flow': cashflow
            }
        except Exception as e:
            logger.error(f"Error fetching yfinance data: {e}")
            return {}

    def _get_finmind_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get financial data from FinMind."""
        try:
            # Get all available data since 2010
            income_stmt = self.finmind.taiwan_stock_financial_statement(
                stock_id=symbol,
                start_date='2010-01-01',
                end_date=pd.Timestamp.now().strftime('%Y-%m-%d')
            )

            balance_stmt = self.finmind.taiwan_stock_balance_sheet(
                stock_id=symbol,
                start_date='2010-01-01',
                end_date=pd.Timestamp.now().strftime('%Y-%m-%d')
            )

            cash_stmt = self.finmind.taiwan_stock_cash_flows_statement(
                stock_id=symbol,
                start_date='2010-01-01',
                end_date=pd.Timestamp.now().strftime('%Y-%m-%d')
            )

            logger.info(f"Raw FinMind data shapes - Income: {income_stmt.shape}, Balance: {balance_stmt.shape}, Cash Flow: {cash_stmt.shape}")

            def process_statement(df: pd.DataFrame, available_cols: List[str] = None) -> pd.DataFrame:
                """Process financial statement with dynamic column selection."""
                if df.empty:
                    return pd.DataFrame()

                # Convert date and pivot
                df['date'] = pd.to_datetime(df['date'])
                df_pivot = df.pivot(index='date', columns='type', values='value')
                
                # Get available columns that exist in the data
                if available_cols:
                    existing_cols = [col for col in available_cols if col in df_pivot.columns]
                    if existing_cols:
                        df_pivot = df_pivot[existing_cols]
                
                # Group by year end and get last value
                yearly_data = (df_pivot
                    .groupby(pd.Grouper(freq='YE'))  # Use YE instead of A
                    .agg('last')
                    .ffill())  # Forward fill missing values
                
                return yearly_data.T

            # Define mappings with fallback columns
            income_cols = {
                'primary': ['Revenue', 'OperatingIncome', 'NetIncome', 'TAX'],
                'fallback': ['OperatingRevenue', 'OperatingProfit', 'ProfitAfterTax', 'IncomeTaxExpense']
            }

            balance_cols = {
                'primary': ['TotalAssets', 'CurrentAssets', 'CurrentLiabilities', 'LongtermBorrowings'],
                'fallback': ['Assets', 'CurrentAsset', 'CurrentLiability', 'LongTermBorrowing']
            }

            cash_cols = {
                'primary': ['CashFlowsFromOperatingActivities', 'PropertyAndPlantAndEquipment', 'Depreciation'],
                'fallback': ['NetCashProvidedByOperatingActivities', 'AcquisitionOfPropertyPlantAndEquipment', 'DepreciationExpense']
            }

            # Process statements with available columns
            income_data = process_statement(income_stmt, income_cols['primary'] + income_cols['fallback'])
            balance_data = process_statement(balance_stmt, balance_cols['primary'] + balance_cols['fallback'])
            cash_data = process_statement(cash_stmt, cash_cols['primary'] + cash_cols['fallback'])

            # Map column names to standard format
            column_maps = {
                'Revenue': 'Total Revenue',
                'OperatingRevenue': 'Total Revenue',
                'OperatingIncome': 'Operating Income',
                'OperatingProfit': 'Operating Income',
                'NetIncome': 'Net Income',
                'ProfitAfterTax': 'Net Income',
                'TAX': 'Tax Provision',
                'IncomeTaxExpense': 'Tax Provision'
                # Add more mappings as needed
            }

            # Rename columns if they exist
            for df in [income_data, balance_data, cash_data]:
                if not df.empty:
                    rename_cols = {old: new for old, new in column_maps.items() if old in df.index}
                    df.rename(index=rename_cols, inplace=True)

            if not income_data.empty:
                logger.info(f"Successfully processed FinMind data with {len(income_data.columns)} years")
                logger.info(f"Income statement metrics: {income_data.index.tolist()}")
                
                return {
                    'income_statement': income_data,
                    'balance_sheet': balance_data,
                    'cash_flow': cash_data
                }

        except Exception as e:
            logger.error(f"Error fetching FinMind data: {e}")
            import traceback
            logger.error(traceback.format_exc())

        return {}

    def _merge_financial_data(self, data1: Dict[str, pd.DataFrame], 
                            data2: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Merge financial data from different sources."""
        merged = {}
        for key in ['income_statement', 'balance_sheet', 'cash_flow']:
            df1 = data1.get(key, pd.DataFrame())
            df2 = data2.get(key, pd.DataFrame())
            
            # Ensure dates are in columns for both dataframes
            if not df1.empty and isinstance(df1.index[0], (datetime, str)):
                df1 = df1.T
            if not df2.empty and isinstance(df2.index[0], (datetime, str)):
                df2 = df2.T
            
            if df1.empty and not df2.empty:
                merged[key] = df2
            elif not df1.empty and df2.empty:
                merged[key] = df1
            elif not df1.empty and not df2.empty:
                # Merge and remove duplicates, keeping the first source's data
                merged[key] = pd.concat([df1, df2]).loc[~pd.concat([df1, df2]).index.duplicated(keep='first')]
            
            # Ensure the final dataframe has metrics in index and dates in columns
            if not merged.get(key, pd.DataFrame()).empty:
                if not isinstance(merged[key].columns[0], (datetime, pd.Timestamp)):
                    merged[key] = merged[key].T
                
        return merged

    def _is_data_sufficient(self, data: Dict[str, pd.DataFrame], min_years: int = 5) -> bool:
        """Check if we have enough years of data."""
        if not data:
            return False
        
        return all(
            not df.empty and len(df.index) >= min_years 
            for df in data.values()
        )
