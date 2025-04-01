import pandas as pd
import numpy as np
import os
import pickle
import logging
import time
import random
from FinMind.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import traceback
import threading
import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime%s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaiwanIndustryDataCollector:
    """Collect and process financial data for Taiwan stocks by industry."""
    
    def __init__(self, data_dir: str = "industry_data", lookback_years: int = 5, 
                rate_limit_delay: float = 2.0, max_retries: int = 3):
        """Initialize the industry data collector.
        
        Args:
            data_dir: Directory to store collected data
            lookback_years: Number of years of historical data to collect
            rate_limit_delay: Delay between API calls to avoid rate limits
            max_retries: Maximum number of retries for failed API calls
        """
        self.data_dir = data_dir
        self.lookback_years = lookback_years
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        
        # Rate limiting variables
        self.api_call_count = 0
        self.api_call_start_time = datetime.datetime.now()
        self.api_call_lock = threading.Lock()  # For thread safety
        self.API_CALL_LIMIT = 300  # Maximum calls before pausing
        self.API_RATE_INTERVAL = 3600  # Pause for this many seconds (1 hour)
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize FinMind data loader
        self.finmind = DataLoader()
        
        # Try to authenticate with FinMind if credentials exist
        try:
            from dotenv import load_dotenv
            load_dotenv()
            import os
            token = os.getenv("FINMIND_TOKEN")
            if token:
                self.finmind.login(token=token)
                logger.info("Authenticated with FinMind API")
        except Exception as e:
            logger.warning(f"Could not authenticate with FinMind: {e}")
        
        # Enhanced industry mapping for Taiwan stocks
        self.industry_map = {
            # Original mappings
            "光電業": "Electronics",
            "半導體業": "Semiconductors",
            "電子零組件業": "Electronics Manufacturing",
            "電腦及週邊設備業": "Computer Hardware",
            "通信網路業": "Telecommunications",
            "其他電子業": "Electronics",
            "電子通路業": "Electronics Distribution",
            "資訊服務業": "Technology Services",
            "金融業": "Financial Services",
            "銀行業": "Banking",
            "證券業": "Securities",
            "保險業": "Insurance",
            "食品工業": "Food & Beverage",
            "塑膠工業": "Materials",
            "紡織纖維": "Textiles",
            "建材營造": "Construction",
            "化學工業": "Chemicals",
            "生技醫療": "Healthcare",
            "油電燃氣": "Utilities",
            "文化創意": "Media & Entertainment",
            "電機機械": "Industrial Equipment",
            "貿易百貨": "Retail",
            "其他": "Other",
            
            # Additional mappings for common Taiwan industries
            "汽車工業": "Automotive",
            "觀光事業": "Tourism & Hospitality",
            "航運業": "Shipping & Transportation",
            "其他金融業": "Financial Services",
            "鋼鐵工業": "Steel & Metals",
            "橡膠工業": "Rubber & Plastics",
            "造紙工業": "Paper & Packaging",
            "水泥工業": "Cement & Construction Materials",
            "農業科技": "Agricultural Technology",
            "電子商務": "E-Commerce",
            "電信服務": "Telecommunications Services",
            "交通運輸": "Transportation",
            "環保工程": "Environmental Services",
            "公用事業": "Utilities",
            "薄膜電晶體液晶顯示器": "LCD Manufacturing",
            "不動產投資信託": "REITs",
            "投資控股": "Investment Holdings",
            
            # New mappings for unmapped categories
            "電器電纜": "Electrical Equipment",
            "農業科技業": "Agricultural Technology",
            "觀光餐旅": "Tourism & Hospitality",
            "生技醫療業": "Healthcare",
            "綠能環保": "Green Energy",
            "運動休閒": "Sports & Leisure",
            "電子工業": "Electronics",
            "運動休閒類": "Sports & Leisure",
            "化學生技醫療": "Healthcare",
            "其他電子類": "Electronics",
            "玻璃陶瓷": "Glass & Ceramics",
            "居家生活": "Home & Living",
            "創新板股票": "Innovation Board",
            "創新版股票": "Innovation Board",
            "油電燃氣業": "Utilities",
            "數位雲端類": "Cloud Computing",
            "金融保險": "Financial Services",
            "居家生活類": "Home & Living",
            "文化創意業": "Media & Entertainment",
            "綠能環保類": "Green Energy",
            "電子商務業": "E-Commerce",
            "數位雲端": "Cloud Computing",
            "建材 營造": "Construction", # Space in original
            "化 學生技醫療": "Healthcare", # Space in original
        }
    
    def _check_rate_limit(self):
        """Check API rate limits and pause if necessary."""
        with self.api_call_lock:
            self.api_call_count += 1
            current_time = datetime.datetime.now()
            elapsed = (current_time - self.api_call_start_time).total_seconds()
            
            # Log every 50 calls
            if self.api_call_count % 50 == 0:
                logger.info(f"API call count: {self.api_call_count} in {elapsed:.1f} seconds")
            
            # If we've reached the limit, pause for the specified interval
            if self.api_call_count >= self.API_CALL_LIMIT:
                pause_time = max(1, self.API_RATE_INTERVAL - elapsed)
                if pause_time > 0:
                    logger.warning(f"Reached API call limit ({self.API_CALL_LIMIT}). Pausing for {pause_time/60:.1f} minutes.")
                    time.sleep(pause_time)
                
                # Reset the counter and timer
                self.api_call_count = 0
                self.api_call_start_time = datetime.datetime.now()
    
    def _is_etf(self, stock_id: str) -> bool:
        """Determine if a stock ID is likely an ETF or other non-company security."""
        # Pattern detection for ETFs and special financial instruments
        filter_patterns = [
            # All stocks starting with "00" are ETFs (like 0050, 0052, 0056, etc.)
            r'^00\d+',
            # All stocks starting with single "0" (like 020000)
            r'^0\d+',
            # Leveraged and inverse ETFs (e.g., 00657L, 00632R)
            r'^\d{5}[LR]$',
            # Trust certificates and REITs (ending with T)
            r'\d+T$',
            # Other ETF numbering patterns
            r'^T[0-9]{2}',  # Known ETF issuers
            # Other types of ETFs
            r'^00[0-9]{2}B',
            r'^00[0-9]{2}U'
        ]
        
        # Check if stock_id matches any filter pattern
        for pattern in filter_patterns:
            if re.match(pattern, stock_id):
                return True
        
        return False
    
    def get_taiwan_stock_list(self) -> pd.DataFrame:
        """Get list of Taiwan stocks with industry classifications."""
        try:
            # Get Taiwan stock info from FinMind
            self._check_rate_limit()  # Check rate limit before API call
            stock_info = self.finmind.taiwan_stock_info()
            
            if stock_info.empty:
                logger.error("Failed to get stock info from FinMind")
                return pd.DataFrame()
            
            # Filter out ETFs before mapping industry codes
            original_count = len(stock_info)
            stock_info['is_etf'] = stock_info['stock_id'].apply(self._is_etf)
            non_etf_stocks = stock_info[~stock_info['is_etf']].copy()
            etf_count = original_count - len(non_etf_stocks)
            logger.info(f"Filtered out {etf_count} ETFs from stock list, keeping {len(non_etf_stocks)} regular stocks")
            
            # Map industry codes to standardized names
            non_etf_stocks['industry'] = non_etf_stocks['industry_category'].map(self.industry_map).fillna("Other")
            
            logger.info(f"Retrieved {len(non_etf_stocks)} Taiwan stocks (excluding ETFs)")
            return non_etf_stocks
            
        except Exception as e:
            logger.error(f"Error getting Taiwan stock list: {e}")
            return pd.DataFrame()
    
    def collect_industry_financial_data(self, max_stocks_per_industry: int = 25, 
                                       parallel: bool = True, max_workers: int = 8,
                                       prioritize_industries: List[str] = None) -> Dict:
        """Collect financial data for stocks grouped by industry."""
        try:
            # Reset rate limit counter at the beginning
            with self.api_call_lock:
                self.api_call_count = 0
                self.api_call_start_time = datetime.datetime.now()
                
            # Get stock list
            stock_info = self.get_taiwan_stock_list()
            
            if stock_info.empty:
                logger.error("Failed to get stock information")
                return {}
            
            # Group by industry
            industry_groups = stock_info.groupby('industry')
            
            # Determine which industries to process
            if prioritize_industries:
                industries = [ind for ind in prioritize_industries if ind in industry_groups.groups]
                # Add remaining industries
                industries.extend([ind for ind in industry_groups.groups if ind not in prioritize_industries])
            else:
                industries = list(industry_groups.groups.keys())
            
            # Process each industry
            industry_data = {}
            for industry in industries:
                logger.info(f"Processing {industry} industry")
                
                # Get stocks in this industry, limit to max_stocks_per_industry
                stocks_in_industry = industry_groups.get_group(industry)
                selected_stocks = stocks_in_industry.sample(min(max_stocks_per_industry, len(stocks_in_industry)))
                
                # Collect data for selected stocks
                industry_stocks = {}
                
                if parallel:
                    # Parallel processing
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {}
                        for _, row in selected_stocks.iterrows():
                            stock_id = row['stock_id']
                            futures[executor.submit(self._collect_stock_data, stock_id)] = stock_id
                        
                        for future in futures:
                            stock_id = futures[future]
                            try:
                                data = future.result()
                                if data and not all(df.empty for df in data.values()):
                                    industry_stocks[stock_id] = data
                                    logger.info(f"Added data for {stock_id}")
                            except Exception as e:
                                logger.error(f"Error collecting data for {stock_id}: {e}")
                else:
                    # Sequential processing
                    for _, row in selected_stocks.iterrows():
                        stock_id = row['stock_id']
                        try:
                            data = self._collect_stock_data(stock_id)
                            if data and not all(df.empty for df in data.values()):
                                industry_stocks[stock_id] = data
                                logger.info(f"Added data for {stock_id}")
                        except Exception as e:
                            logger.error(f"Error collecting data for {stock_id}: {e}")
                
                # Only add industry if we have data for at least one stock
                if industry_stocks:
                    industry_data[industry] = industry_stocks
                    logger.info(f"Collected data for {len(industry_stocks)} stocks in {industry}")
                
                # Save data after each industry
                self._save_industry_data(industry_data)
            
            return industry_data
            
        except Exception as e:
            logger.error(f"Error collecting industry data: {e}")
            return {}
    
    def _collect_stock_data(self, stock_id: str) -> Dict[str, pd.DataFrame]:
        """Collect financial data for a single stock."""
        try:
            # Calculate date range
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            start_date = (pd.Timestamp.now() - pd.DateOffset(years=self.lookback_years)).strftime('%Y-%m-%d')
            
            # Initialize retries
            retries = 0
            data = {
                'financial_statement': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame(),
                'price_data': pd.DataFrame()
            }
            
            while retries < self.max_retries:
                try:
                    # Get financial statement data
                    self._check_rate_limit()  # Check rate limit before API call
                    financial_statement = self.finmind.taiwan_stock_financial_statement(
                        stock_id=stock_id,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if not financial_statement.empty:
                        data['financial_statement'] = financial_statement
                    
                    # Add delay to avoid rate limits
                    time.sleep(self.rate_limit_delay)
                    
                    # Get balance sheet data
                    self._check_rate_limit()  # Check rate limit before API call
                    balance_sheet = self.finmind.taiwan_stock_balance_sheet(
                        stock_id=stock_id,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if not balance_sheet.empty:
                        data['balance_sheet'] = balance_sheet
                    
                    # Add delay to avoid rate limits
                    time.sleep(self.rate_limit_delay)
                    
                    # Get cash flow data
                    self._check_rate_limit()  # Check rate limit before API call
                    cash_flow = self.finmind.taiwan_stock_cash_flows_statement(
                        stock_id=stock_id,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if not cash_flow.empty:
                        data['cash_flow'] = cash_flow
                    
                    # Add delay to avoid rate limits
                    time.sleep(self.rate_limit_delay)
                    
                    # Get price data
                    self._check_rate_limit()  # Check rate limit before API call
                    price_data = self.finmind.taiwan_stock_daily(
                        stock_id=stock_id,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if not price_data.empty:
                        data['price_data'] = price_data
                    
                    # If we got here, we successfully collected data
                    break
                    
                except Exception as e:
                    retries += 1
                    logger.warning(f"Error on attempt {retries} for {stock_id}: {e}")
                    
                    # Add longer delay after failure
                    if "rate limit" in str(e).lower() or "upper limit" in str(e).lower():
                        sleep_time = 60 + random.uniform(30, 60)  # 1-2 minutes
                        logger.warning(f"Hit rate limit, waiting {sleep_time:.1f} seconds")
                        time.sleep(sleep_time)
                    else:
                        time.sleep(self.rate_limit_delay * 2)
            
            # Calculate collected data percentage
            collected_count = sum(1 for df in data.values() if not df.empty)
            percentage = (collected_count / len(data)) * 100
            
            logger.info(f"Collected {percentage:.1f}% of data types for {stock_id}")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting data for {stock_id}: {e}")
            return {
                'financial_statement': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame(), 
                'price_data': pd.DataFrame()
            }
    
    def _save_industry_data(self, industry_data: Dict):
        """Save industry financial data to pickle file."""
        try:
            file_path = os.path.join(self.data_dir, "taiwan_industry_financial_data.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(industry_data, f)
            logger.info(f"Saved industry data to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving industry data: {e}")
            return False
    
    def _load_industry_data(self) -> Dict:
        """Load industry financial data from pickle file."""
        try:
            file_path = os.path.join(self.data_dir, "taiwan_industry_financial_data.pkl")
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    industry_data = pickle.load(f)
                logger.info(f"Loaded industry data from {file_path}")
                return industry_data
            else:
                logger.warning(f"Industry data file not found at {file_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading industry data: {e}")
            return {}
    
    def _extract_financial_metrics(self, stock_id: str, data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Extract financial metrics from raw data for a stock."""
        try:
            # Extract metrics from financial statements, balance sheets, and cash flows
            financial_records = []
            
            # Process financial statements
            financial_stmt = data.get('financial_statement', pd.DataFrame())
            balance_sheet = data.get('balance_sheet', pd.DataFrame())
            cash_flow = data.get('cash_flow', pd.DataFrame())
            price_data = data.get('price_data', pd.DataFrame())
            
            if financial_stmt.empty or balance_sheet.empty:
                return []
            
            # Group by report date (use date column from financial_stmt)
            dates = sorted(financial_stmt['date'].unique())
            
            for i, report_date in enumerate(dates):
                try:
                    # Filter data for this date
                    period_fs = financial_stmt[financial_stmt['date'] == report_date]
                    period_bs = balance_sheet[balance_sheet['date'] == report_date]
                    
                    # Skip if no financial statement or balance sheet data for this period
                    if period_fs.empty or period_bs.empty:
                        continue
                    
                    # Create a record for this period
                    record = {
                        'stock_id': stock_id,
                        'timestamp': pd.to_datetime(report_date),
                    }
                    
                    # Extract revenue
                    revenue_fields = ['Revenue', 'OperatingRevenue']
                    revenue = None
                    for field in revenue_fields:
                        rev_rows = period_fs[period_fs['type'] == field]
                        if not rev_rows.empty:
                            revenue = float(rev_rows['value'].iloc[0])
                            break
                    
                    if revenue is None or revenue <= 0:
                        continue  # Skip if no valid revenue
                    
                    record['revenue'] = revenue
                    
                    # Calculate prior period growth if available
                    if i > 0:
                        prior_date = dates[i-1]
                        prior_fs = financial_stmt[financial_stmt['date'] == prior_date]
                        prior_revenue = None
                        for field in revenue_fields:
                            prior_rev_rows = prior_fs[prior_fs['type'] == field]
                            if not prior_rev_rows.empty:
                                prior_revenue = float(prior_rev_rows['value'].iloc[0])
                                break
                        
                        if prior_revenue is not None and prior_revenue > 0:
                            growth_rate = (revenue - prior_revenue) / prior_revenue
                            record['historical_growth'] = growth_rate
                    
                    # Extract operating and net income
                    operating_fields = ['OperatingIncome', 'OperatingProfit']
                    operating_income = None
                    for field in operating_fields:
                        op_rows = period_fs[period_fs['type'] == field]
                        if not op_rows.empty:
                            operating_income = float(op_rows['value'].iloc[0])
                            break
                    
                    if operating_income is not None:
                        record['operating_income'] = operating_income
                        record['operating_margin'] = operating_income / revenue
                    
                    net_income_fields = ['NetIncome', 'ProfitAfterTax']
                    net_income = None
                    for field in net_income_fields:
                        net_rows = period_fs[period_fs['type'] == field]
                        if not net_rows.empty:
                            net_income = float(net_rows['value'].iloc[0])
                            break
                    
                    if net_income is not None:
                        record['net_income'] = net_income
                        record['net_margin'] = net_income / revenue
                    
                    # Extract balance sheet metrics
                    total_assets_fields = ['TotalAssets', 'Assets']
                    total_assets = None
                    for field in total_assets_fields:
                        asset_rows = period_bs[period_bs['type'] == field]
                        if not asset_rows.empty:
                            total_assets = float(asset_rows['value'].iloc[0])
                            break
                    
                    if total_assets is not None and total_assets > 0:
                        record['total_assets'] = total_assets
                        
                        # Calculate ROA if we have net income
                        if net_income is not None:
                            record['roa'] = net_income / total_assets
                    
                    # Get shareholders' equity
                    equity_fields = ['TotalEquity', 'StockholdersEquity']
                    total_equity = None
                    for field in equity_fields:
                        equity_rows = period_bs[period_bs['type'] == field]
                        if not equity_rows.empty:
                            total_equity = float(equity_rows['value'].iloc[0])
                            break
                    
                    if total_equity is not None and total_equity > 0:
                        record['total_equity'] = total_equity
                        
                        # Calculate ROE if we have net income
                        if net_income is not None:
                            record['roe'] = net_income / total_equity
                        
                        # Calculate debt-to-equity
                        if total_assets is not None:
                            total_liabilities = total_assets - total_equity
                            record['debt_to_equity'] = total_liabilities / total_equity
                            record['equity_to_assets'] = total_equity / total_assets
                    
                    # Get cash flow data if available
                    if not cash_flow.empty:
                        period_cf = cash_flow[cash_flow['date'] == report_date]
                        
                        if not period_cf.empty:
                            # Operating cash flow
                            ocf_fields = ['CashFlowsFromOperatingActivities', 'NetCashProvidedByOperatingActivities']
                            ocf = None
                            for field in ocf_fields:
                                ocf_rows = period_cf[period_cf['type'] == field]
                                if not ocf_rows.empty:
                                    ocf = float(ocf_rows['value'].iloc[0])
                                    break
                            
                            if ocf is not None:
                                record['operating_cash_flow'] = ocf
                                record['ocf_to_revenue'] = ocf / revenue
                            
                            # Capital expenditure
                            capex_fields = ['PropertyAndPlantAndEquipment', 'AcquisitionOfPropertyPlantAndEquipment']
                            capex = None
                            for field in capex_fields:
                                capex_rows = period_cf[period_cf['type'] == field]
                                if not capex_rows.empty:
                                    capex = float(capex_rows['value'].iloc[0])
                                    break
                            
                            if capex is not None:
                                record['capex'] = capex
                                record['capex_to_revenue'] = capex / revenue
                                
                                # Calculate free cash flow
                                if ocf is not None:
                                    fcf = ocf - capex
                                    record['free_cash_flow'] = fcf
                                    record['fcf_to_revenue'] = fcf / revenue
                    
                    # Get future performance from price data (if available)
                    if not price_data.empty:
                        # Convert report date to timestamp
                        report_ts = pd.to_datetime(report_date)
                        
                        # Get price at report date (or nearest after)
                        price_data['date'] = pd.to_datetime(price_data['date'])
                        
                        # Find closest date after report date
                        future_prices = price_data[price_data['date'] >= report_ts]
                        if not future_prices.empty:
                            # Starting price (closest date after report)
                            start_price = future_prices.iloc[0]['close']
                            
                            # Calculate 6-month future price
                            future_date_6m = report_ts + pd.DateOffset(months=6)
                            future_prices_6m = price_data[(price_data['date'] >= future_date_6m)]
                            
                            if not future_prices_6m.empty:
                                future_price_6m = future_prices_6m.iloc[0]['close']
                                future_return_6m = (future_price_6m - start_price) / start_price
                                record['future_6m_return'] = future_return_6m
                            
                            # Calculate 12-month future price
                            future_date_12m = report_ts + pd.DateOffset(months=12)
                            future_prices_12m = price_data[(price_data['date'] >= future_date_12m)]
                            
                            if not future_prices_12m.empty:
                                future_price_12m = future_prices_12m.iloc[0]['close']
                                future_return_12m = (future_price_12m - start_price) / start_price
                                record['future_12m_return'] = future_return_12m
                    
                    # Add to records only if we have enough data
                    required_fields = ['revenue', 'operating_margin', 'net_margin']
                    if all(field in record for field in required_fields):
                        financial_records.append(record)
                    
                except Exception as e:
                    logger.debug(f"Error processing period {report_date} for {stock_id}: {e}")
            
            # Calculate average historical growth and add to each record
            if len(financial_records) > 1:
                growth_rates = [r.get('historical_growth') for r in financial_records if 'historical_growth' in r]
                if growth_rates:
                    historical_growth_mean = np.mean(growth_rates)
                    historical_growth_std = np.std(growth_rates)
                    
                    for r in financial_records:
                        r['historical_growth_mean'] = historical_growth_mean
                        r['historical_growth_std'] = historical_growth_std
            
            return financial_records
            
        except Exception as e:
            logger.error(f"Error extracting metrics for {stock_id}: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def prepare_training_data(self):
        """Prepare training data for industry valuation models."""
        try:
            # Load industry data
            industry_data = self._load_industry_data()
            
            if not industry_data:
                logger.error("No industry data found - run collect_industry_financial_data first")
                return {}
            
            training_datasets = {}
            
            # Process each industry
            for industry, stocks in industry_data.items():
                logger.info(f"Preparing training data for {industry} industry with {len(stocks)} stocks")
                
                # Extract metrics from all stocks in this industry
                industry_records = []
                
                for stock_id, data in stocks.items():
                    try:
                        metrics = self._extract_financial_metrics(stock_id, data)
                        if metrics:
                            industry_records.extend(metrics)
                    except Exception as e:
                        logger.error(f"Error extracting metrics for {stock_id}: {e}")
                
                # Create industry dataset if we have enough records
                if industry_records:
                    industry_df = pd.DataFrame(industry_records)
                    
                    # Save industry training data
                    training_file = os.path.join(self.data_dir, f"{industry.replace(' ', '_').lower()}_training.csv")
                    industry_df.to_csv(training_file, index=False)
                    
                    training_datasets[industry] = industry_df
                    logger.info(f"Created training dataset for {industry} with {len(industry_df)} records")
            
            return training_datasets
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return {}
    
    def get_industry_benchmark_metrics(self) -> pd.DataFrame:
        """Calculate benchmark financial metrics by industry."""
        try:
            industry_metrics = []
            
            # Load the training datasets
            training_datasets = {}
            for industry_file in os.listdir(self.data_dir):
                if industry_file.endswith('_training.csv'):
                    industry_name = industry_file.replace('_training.csv', '').replace('_', ' ')
                    df = pd.read_csv(os.path.join(self.data_dir, industry_file))
                    
                    if df.empty:
                        continue
                    
                    training_datasets[industry_name] = df
            
            if not training_datasets:
                logger.error("No training datasets found")
                return pd.DataFrame()
            
            # Calculate benchmarks for each industry
            for industry_name, df in training_datasets.items():
                # Calculate key metrics
                metrics = {
                    'industry': industry_name,
                    'stock_count': df['stock_id'].nunique(),
                    'record_count': len(df)
                }
                
                # Calculate median values for key metrics
                for metric in ['historical_growth_mean', 'operating_margin', 'net_margin', 
                              'roa', 'roe', 'debt_to_equity', 'ocf_to_revenue',
                              'capex_to_revenue', 'fcf_to_revenue']:
                    if metric in df.columns:
                        metrics[f'{metric}_median'] = df[metric].median()
                        metrics[f'{metric}_mean'] = df[metric].mean()
                        metrics[f'{metric}_std'] = df[metric].std()
                
                industry_metrics.append(metrics)
            
            # Create benchmark DataFrame
            benchmarks = pd.DataFrame(industry_metrics)
            
            # Save to CSV
            benchmark_file = os.path.join(self.data_dir, 'industry_benchmarks.csv')
            benchmarks.to_csv(benchmark_file, index=False)
            
            logger.info(f"Generated benchmarks for {len(benchmarks)} industries")
            return benchmarks
            
        except Exception as e:
            logger.error(f"Error calculating industry benchmarks: {e}")
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    collector = TaiwanIndustryDataCollector(
        lookback_years=5,
        rate_limit_delay=2.0,
        max_retries=3
    )
    
    # Collect financial data for Taiwan industries
    collector.collect_industry_financial_data(
        max_stocks_per_industry=5,
        parallel=True,
        max_workers=2
    )
    
    # Prepare training data
    collector.prepare_training_data()
    
    # Generate industry benchmarks
    collector.get_industry_benchmark_metrics()
