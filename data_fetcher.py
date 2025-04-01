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

    def get_market_data(self, symbol: str, force_refresh: bool = False) -> Dict:
        """Get market data including stock price, market cap, and trading info."""
        try:
            cache_file = self.cache_dir / f"{symbol}_market_data.pkl"
            
            # Use cached data if available and not forcing refresh
            if not force_refresh and cache_file.exists():
                cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - cache_time < timedelta(days=1):
                    logger.info(f"Using cached market data for {symbol}")
                    return pd.read_pickle(cache_file)
            
            # Handle Taiwan stocks
            is_taiwan = '.TW' in symbol or '.TWO' in symbol
            
            market_data = {}
            
            # Try yfinance first
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                
                # Add diagnostic logging to see what we're getting
                logger.info(f"YFinance data for {symbol}: Price: {info.get('regularMarketPrice')}, Shares: {info.get('sharesOutstanding')}")
                
                # Get price and shares
                price = info.get('regularMarketPrice', None)
                shares = info.get('sharesOutstanding', None)
                
                # For Taiwan stocks, sometimes 'sharesOutstanding' is missing but 'floatShares' is available
                if shares is None and 'floatShares' in info and info['floatShares'] is not None:
                    shares = info['floatShares']
                    logger.info(f"Using floatShares as substitute for sharesOutstanding: {shares:,.0f}")
                
                # For Taiwan stocks, also check 'currentPrice' if 'regularMarketPrice' is None
                if price is None and 'currentPrice' in info and info['currentPrice'] is not None:
                    price = info['currentPrice']
                    logger.info(f"Using currentPrice as substitute for regularMarketPrice: {price}")
                
                if price is not None and shares is not None and shares > 0:
                    market_cap = price * shares
                    
                    # Log the calculation explicitly
                    logger.info(f"Successfully calculated market cap: {price} × {shares:,.0f} = {market_cap:,.0f}")
                    
                    market_data = {
                        'price': price,
                        'shares_outstanding': shares,
                        'market_cap': market_cap,
                        'beta': info.get('beta', None),
                        'source': 'yfinance'
                    }
                    
                    pd.to_pickle(market_data, cache_file)
                    return market_data
                else:
                    # Be more specific about what's missing
                    if price is None:
                        logger.warning(f"Price data missing for {symbol}")
                    if shares is None or shares <= 0:
                        logger.warning(f"Shares outstanding data missing or invalid for {symbol}")
                    
                    # Store partial data and continue to try other sources
                    if price is not None:
                        market_data['price'] = price
                    if shares is not None:
                        market_data['shares_outstanding'] = shares
                    if price is not None and shares is not None:
                        market_data['market_cap'] = price * shares
                    market_data['beta'] = info.get('beta', None)
                    market_data['source'] = 'yfinance'
            except Exception as e:
                logger.warning(f"Error getting yfinance market data for {symbol}: {e}")
            
            # For Taiwan stocks, try FinMind if yfinance didn't provide complete data
            if is_taiwan and ('market_cap' not in market_data or market_data.get('market_cap') is None):
                try:
                    base_symbol = symbol.split('.')[0]
                    logger.info(f"Trying FinMind for market data on {base_symbol}")
                    
                    # Get stock info from FinMind
                    stock_info = self.finmind.taiwan_stock_info()
                    stock_row = stock_info[stock_info['stock_id'] == base_symbol]
                    
                    if not stock_row.empty:
                        # Get latest price data
                        price_data = self.finmind.taiwan_stock_daily(
                            stock_id=base_symbol,
                            start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                            end_date=datetime.now().strftime('%Y-%m-%d')
                        )
                        
                        if not price_data.empty:
                            latest_price = float(price_data.iloc[-1]['close'])
                            logger.info(f"Latest price from FinMind: {latest_price}")
                            market_data['price'] = latest_price
                            
                            # Try to get shares outstanding data
                            # First look for market cap in stock_info
                            if 'market_value' in stock_row.columns and not pd.isna(stock_row['market_value'].values[0]):
                                market_cap = float(stock_row['market_value'].values[0]) * 1000  # Convert thousands to full value
                                shares_outstanding = market_cap / latest_price
                                logger.info(f"Estimated shares from market_value: {shares_outstanding:,.0f}")
                                market_data['shares_outstanding'] = shares_outstanding
                                market_data['market_cap'] = market_cap
                            else:
                                # Try to get shares from capital reduction data or monthly revenue
                                try:
                                    shares_data = self.finmind.taiwan_stock_capital_reduction(
                                        stock_id=base_symbol
                                    )
                                    
                                    if not shares_data.empty:
                                        # Get latest shares data
                                        latest_shares = float(shares_data.iloc[-1]['outstanding_share'])
                                        logger.info(f"Shares from capital reduction: {latest_shares:,.0f}")
                                        market_data['shares_outstanding'] = latest_shares
                                        market_data['market_cap'] = latest_price * latest_shares
                                    else:
                                        # Fallback: Try to estimate from financial statements 
                                        financial_data = self.get_financial_data(symbol)
                                        if financial_data and 'balance_sheet' in financial_data:
                                            balance = financial_data['balance_sheet']
                                            
                                            # Look for Common Stock or similar entries
                                            for key in ['Common Stock', 'CommonStocks', 'Capital', 'IssuedCapital']:
                                                if key in balance.index:
                                                    common_stock = float(balance.loc[key, balance.columns[-1]])
                                                    # Typical par value for Taiwan stocks is NT$10
                                                    shares_est = common_stock / 10
                                                    logger.info(f"Estimated shares from {key}: {shares_est:,.0f}")
                                                    market_data['shares_outstanding'] = shares_est
                                                    market_data['market_cap'] = latest_price * shares_est
                                                    break
                                except Exception as e:
                                    logger.warning(f"Error getting shares data: {e}")

                            # Explicitly recalculate market cap to ensure consistency
                            if 'shares_outstanding' in market_data and market_data['shares_outstanding'] is not None:
                                shares = market_data['shares_outstanding']
                                price = market_data['price']
                                market_data['market_cap'] = price * shares
                                logger.info(f"Calculated market cap: {market_data['market_cap']:,.0f}")
                            
                            # Get beta or use industry average
                            if 'beta' not in market_data or market_data['beta'] is None:
                                # Estimate beta for Taiwan stocks based on stock number
                                if base_symbol in ['2330', '2454', '2379', '2337', '2408']:
                                    beta = 1.3  # Semiconductors
                                elif base_symbol in ['2317', '2382', '2354', '2353', '2474']:
                                    beta = 1.15  # Hardware
                                else:
                                    beta = 1.2  # Default for tech
                                
                                market_data['beta'] = beta
                                logger.info(f"Using estimated beta: {beta}")
                            
                            market_data['source'] = 'finmind'
                            pd.to_pickle(market_data, cache_file)
                            logger.info(f"Retrieved market data for {symbol} from FinMind")
                except Exception as e:
                    logger.warning(f"Error getting FinMind market data for {symbol}: {e}")
            
            # Final fallback: For Taiwan stocks, try to estimate market cap from financial data if still missing
            if is_taiwan and ('market_cap' not in market_data or market_data.get('market_cap') is None):
                try:
                    # Get financial data if not already fetched
                    financial_data = self.get_financial_data(symbol, force_refresh=False)
                    
                    if financial_data and 'balance_sheet' in financial_data:
                        balance = financial_data['balance_sheet']
                        
                        # Look for market value indicators in the balance sheet
                        market_value_keys = ['TotalEquity', 'Total Equity', 'StockholdersEquity']
                        for key in market_value_keys:
                            if key in balance.index:
                                # Use book equity as a rough approximation, typically with a multiplier
                                book_equity = float(balance.loc[key, balance.columns[-1]])
                                
                                # Typical price-to-book ratios by country
                                pb_ratio = 2.0  # Standard value
                                country_code = symbol.split('.')[-1] if '.' in symbol else 'US'  # Extract from symbol instead of undefined attribute
                                if country_code == 'TW':
                                    # Taiwan tech companies typically trade at higher P/B ratios
                                    if symbol.startswith('2330') or symbol.startswith('2454'):
                                        pb_ratio = 4.0  # Higher P/B for leading tech
                                    else:
                                        pb_ratio = 2.5  # Standard Taiwan premium
                                
                                # Estimate market cap from book value
                                estimated_market_cap = book_equity * pb_ratio
                                logger.info(f"Estimated market cap from book equity: {book_equity:,.0f} × {pb_ratio} = {estimated_market_cap:,.0f}")
                                
                                # Only use this if we don't have any market cap yet
                                if 'market_cap' not in market_data or market_data.get('market_cap') is None:
                                    market_data['market_cap'] = estimated_market_cap
                                    market_data['source'] = 'estimated'
                                    
                                    # If we have price but not shares, estimate shares
                                    if 'price' in market_data and market_data['price'] is not None and market_data['price'] > 0:
                                        estimated_shares = estimated_market_cap / market_data['price']
                                        market_data['shares_outstanding'] = estimated_shares
                                        logger.info(f"Estimated shares outstanding: {estimated_shares:,.0f}")
                                
                                break
                except Exception as e:
                    logger.warning(f"Error estimating market cap from financial data: {e}")
            
            # If we get here, we have partial data or no data
            if market_data:
                # Double-check that market cap is consistent with price and shares
                if 'price' in market_data and 'shares_outstanding' in market_data:
                    price = market_data['price']
                    shares = market_data['shares_outstanding']
                    if price is not None and shares is not None:
                        market_data['market_cap'] = price * shares
                        logger.info(f"Verified market cap calculation: {market_data['market_cap']:,.0f}")
                        
                pd.to_pickle(market_data, cache_file)
                logger.info(f"Saved market data for {symbol}")
                return market_data
            
            logger.error(f"Failed to get market data for {symbol}")
            return {}
        
        except Exception as e:
            logger.error(f"Error in get_market_data: {e}")
            return {}

    def get_risk_free_rate(self, country='US'):
        """Get current risk-free rate for the specified country."""
        try:
            cache_file = self.cache_dir / f"risk_free_rate_{country}.pkl"
            
            # Use cached data if available and fresh (updated daily)
            if cache_file.exists():
                cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - cache_time < timedelta(days=1):
                    data = pd.read_pickle(cache_file)
                    logger.info(f"Using cached risk-free rate for {country}: {data['rate']:.2%}")
                    return data['rate']
            
            # Default rates by country (fallback)
            default_rates = {
                'US': 0.035,  # US 10-year Treasury
                'TW': 0.015,  # Taiwan government bond
                'HK': 0.02,   # Hong Kong
                'JP': 0.005,  # Japan
                'UK': 0.025,  # UK
                'EU': 0.025   # Europe
            }
            
            rate = None
            
            # Try to fetch current rates
            if country == 'US':
                try:
                    # US 10-year Treasury from yfinance
                    treasury = yf.Ticker('^TNX')
                    history = treasury.history(period="1d")
                    if not history.empty:
                        # Convert from percentage to decimal
                        rate = float(history['Close'].iloc[-1]) / 100
                except Exception as e:
                    logger.warning(f"Error fetching US Treasury rate: {e}")
                    
            elif country == 'TW':
                try:
                    # Try to get Taiwan 10-year government bond rate
                    # First attempt using FinMind
                    taiwan_rates = self.finmind.taiwan_government_bond()
                    if not taiwan_rates.empty:
                        latest = taiwan_rates[taiwan_rates['date'] == taiwan_rates['date'].max()]
                        if not latest.empty and '10Y' in latest['title'].values:
                            rate_row = latest[latest['title'] == '10Y']
                            rate = float(rate_row['yield'].iloc[0]) / 100
                except Exception as e:
                    logger.warning(f"Error fetching Taiwan bond rate: {e}")
            
            # If we couldn't get a current rate, use the default
            if rate is None or rate <= 0:
                rate = default_rates.get(country, 0.035)  # Default to US rate
                logger.info(f"Using default risk-free rate for {country}: {rate:.2%}")
            else:
                logger.info(f"Retrieved current risk-free rate for {country}: {rate:.2%}")
            
            # Cache the result
            pd.to_pickle({'rate': rate, 'date': datetime.now()}, cache_file)
            
            return rate
            
        except Exception as e:
            logger.error(f"Error getting risk-free rate: {e}")
            # Fallback to a reasonable default
            return 0.035  # 3.5% as fallback
