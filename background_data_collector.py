import sqlite3
import pandas as pd
import numpy as np
import logging
import time
import os
import schedule
import threading
import datetime
from FinMind.data import DataLoader
import random
from typing import List, Dict, Optional, Tuple
import json
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("background_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BackgroundDataCollector:
    """
    A utility that collects financial data in the background and stores it in a SQLite database
    to minimize API calls and avoid rate limits.
    """
    
    def __init__(self, db_path: str = "finance_data.db", collection_interval: int = 12):
        """Initialize the background data collector.
        
        Args:
            db_path: Path to the SQLite database file
            collection_interval: Hours between collection runs (default: 12 hours)
        """
        self.db_path = db_path
        self.collection_interval = collection_interval
        self.finmind = DataLoader()
        self.is_collecting = False
        self.scheduler_thread = None
        
        # Rate limiting variables
        self.api_call_count = 0
        self.api_call_start_time = datetime.datetime.now()
        self.api_call_lock = threading.Lock()  # For thread safety
        self.API_CALL_LIMIT = 300  # Maximum calls before pausing
        self.API_RATE_INTERVAL = 3600  # Pause for this many seconds (1 hour)
        
        # Create database and tables if they don't exist
        self._initialize_database()
        
        # Try to authenticate with FinMind if credentials exist
        try:
            from dotenv import load_dotenv
            load_dotenv()
            token = os.getenv("FINMIND_TOKEN")
            if token:
                self.finmind.login(token=token)
                logger.info("Authenticated with FinMind API")
        except Exception as e:
            logger.warning(f"Could not authenticate with FinMind: {e}")
        
        # Industry mapping for Taiwan stocks
        self.industry_map = {
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
            "其他": "Other"
        }
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create stock_info table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_info (
                stock_id TEXT PRIMARY KEY,
                stock_name TEXT,
                industry TEXT,
                last_updated TIMESTAMP
            )
            ''')
            
            # Create financial_statements table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_statements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_id TEXT,
                date TEXT,
                metric_type TEXT,
                value REAL,
                UNIQUE(stock_id, date, metric_type)
            )
            ''')
            
            # Create balance_sheets table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS balance_sheets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_id TEXT,
                date TEXT,
                metric_type TEXT,
                value REAL,
                UNIQUE(stock_id, date, metric_type)
            )
            ''')
            
            # Create cash_flows table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS cash_flows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_id TEXT,
                date TEXT,
                metric_type TEXT,
                value REAL,
                UNIQUE(stock_id, date, metric_type)
            )
            ''')
            
            # Create stock_prices table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_id TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                UNIQUE(stock_id, date)
            )
            ''')
            
            # Create collection_log table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                stock_id TEXT,
                data_type TEXT,
                status TEXT,
                message TEXT
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def start_scheduler(self):
        """Start the background data collection scheduler."""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("Scheduler already running")
            return
        
        # Set up the schedule
        schedule.every(self.collection_interval).hours.do(self.collect_batch)
        
        # Also run immediately when starting
        schedule.run_all()
        
        # Start the scheduler in a separate thread
        def run_scheduler():
            while self.is_collecting:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.is_collecting = True
        self.scheduler_thread = threading.Thread(target=run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info(f"Scheduler started, collecting data every {self.collection_interval} hours")
    
    def stop_scheduler(self):
        """Stop the background data collection scheduler."""
        self.is_collecting = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
            logger.info("Scheduler stopped")
    
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

    def collect_batch(self):
        """Collect a batch of data from FinMind API."""
        try:
            # Get the list of Taiwan stocks
            stock_info = self._get_and_save_stock_list()
            
            if stock_info.empty:
                logger.error("Failed to get stock information")
                return
            
            # Determine which stocks to collect in this batch
            stocks_to_collect = self._select_stocks_for_batch(stock_info)
            
            logger.info(f"Collecting data for {len(stocks_to_collect)} stocks in this batch")
            
            # Reset rate limit counter at the start of each batch
            with self.api_call_lock:
                self.api_call_count = 0
                self.api_call_start_time = datetime.datetime.now()
            
            # Collect data for each stock
            for i, (stock_id, industry) in enumerate(stocks_to_collect):
                try:
                    # Add jitter to avoid predictable request patterns
                    jitter = random.uniform(1.0, 3.0)
                    time.sleep(jitter)
                    
                    # Log progress
                    logger.info(f"Collecting data for {stock_id} ({industry}) - {i+1}/{len(stocks_to_collect)}")
                    
                    # Get financial data
                    self._collect_and_save_financial_data(stock_id, industry)
                    
                    # Get price data
                    self._collect_and_save_price_data(stock_id)
                    
                    # Add a longer delay after each stock to avoid rate limits
                    time.sleep(random.uniform(5.0, 10.0))
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error collecting data for {stock_id}: {error_msg}")
                    
                    # If we hit a rate limit, wait much longer and then continue
                    if "402" in error_msg or "rate limit" in error_msg.lower() or "upper limit" in error_msg.lower():
                        logger.warning(f"Rate limit hit. Pausing for 15 minutes before continuing.")
                        self._log_collection_attempt(stock_id, "all", "rate_limited", error_msg)
                        time.sleep(900)  # Wait 15 minutes
                    else:
                        self._log_collection_attempt(stock_id, "all", "error", error_msg)
            
            logger.info(f"Batch collection completed at {datetime.datetime.now()}")
            
        except Exception as e:
            logger.error(f"Error in batch collection: {e}")
            logger.error(traceback.format_exc())
    
    def _get_and_save_stock_list(self) -> pd.DataFrame:
        """Get a list of all Taiwan stocks and save to the database."""
        try:
            # Check when we last updated the stock list
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(last_updated) FROM stock_info")
            last_updated = cursor.fetchone()[0]
            
            # If we updated the stock list recently, reuse it
            if (last_updated):
                last_updated_date = datetime.datetime.fromisoformat(last_updated)
                if (datetime.datetime.now() - last_updated_date).days < 7:
                    # List is less than 7 days old, retrieve from DB
                    cursor.execute("SELECT stock_id, stock_name, industry FROM stock_info")
                    stocks = cursor.fetchall()
                    conn.close()
                    
                    if stocks:
                        logger.info(f"Using cached stock list with {len(stocks)} stocks")
                        # Convert to DataFrame
                        stock_info = pd.DataFrame(stocks, columns=['stock_id', 'stock_name', 'industry'])
                        return stock_info
            
            # Get Taiwan stock info from FinMind
            self._check_rate_limit()  # Check rate limit before API call
            stock_info = self.finmind.taiwan_stock_info()
            
            if stock_info.empty:
                logger.error("Failed to get stock info from FinMind")
                return pd.DataFrame()
            
            # Map industry codes to standardized names
            stock_info['industry'] = stock_info['industry_category'].map(self.industry_map).fillna("Other")
            
            # Save to database
            now = datetime.datetime.now().isoformat()
            for _, row in stock_info.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO stock_info (stock_id, stock_name, industry, last_updated)
                    VALUES (?, ?, ?, ?)
                ''', (row['stock_id'], row['stock_name'], row['industry'], now))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Retrieved and saved {len(stock_info)} Taiwan stocks")
            return stock_info[['stock_id', 'stock_name', 'industry']]
            
        except Exception as e:
            logger.error(f"Error getting Taiwan stock list: {e}")
            return pd.DataFrame()
    
    def _select_stocks_for_batch(self, stock_info: pd.DataFrame) -> List[Tuple[str, str]]:
        """Select a batch of stocks to collect data for, prioritizing least recently updated."""
        try:
            # Get the last collection time for each stock
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.stock_id, s.industry, MAX(l.timestamp) as last_collected
                FROM stock_info s
                LEFT JOIN collection_log l ON s.stock_id = l.stock_id AND l.status = 'success'
                GROUP BY s.stock_id
                ORDER BY last_collected ASC NULLS FIRST
            ''')
            
            collection_times = cursor.fetchall()
            conn.close()
            
            # Create a dictionary of stock_id -> (industry, last_collected)
            stock_collection_times = {row[0]: (row[1], row[2]) for row in collection_times}
            
            # Prioritize stocks that have never been collected
            never_collected = []
            last_collected_stocks = []
            
            for _, row in stock_info.iterrows():
                stock_id = row['stock_id']
                industry = row['industry']
                
                if stock_id not in stock_collection_times:
                    never_collected.append((stock_id, industry))
                else:
                    last_collected_stocks.append((stock_id, stock_collection_times[stock_id][0], stock_collection_times[stock_id][1]))
            
            # Sort by last collection time (oldest first)
            last_collected_stocks.sort(key=lambda x: x[2] if x[2] else "")
            
            # Combine never collected and last collected stocks
            all_ordered_stocks = [(s[0], s[1]) for s in never_collected] + [(s[0], s[1]) for s in last_collected_stocks]
            
            # Limit to a small batch size to avoid rate limits
            batch_size = min(10, len(all_ordered_stocks))
            
            # Return the batch to collect
            return all_ordered_stocks[:batch_size]
            
        except Exception as e:
            logger.error(f"Error selecting stocks for batch: {e}")
            
            # Fallback to a random selection from stock_info
            if not stock_info.empty:
                sample_size = min(5, len(stock_info))
                sample = stock_info.sample(sample_size)
                return [(row['stock_id'], row['industry']) for _, row in sample.iterrows()]
            
            return []
    
    def _collect_and_save_financial_data(self, stock_id: str, industry: str):
        """Collect and save financial data for a single stock."""
        try:
            # Calculate date range - go back 5 years
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=5*365)).strftime('%Y-%m-%d')
            
            # Get financial statement data
            try:
                self._check_rate_limit()  # Check rate limit before API call
                financial_statement = self.finmind.taiwan_stock_financial_statement(
                    stock_id=stock_id,
                    start_date=start_date,
                    end_date=end_date
                )
                
                self._save_financial_statement(stock_id, financial_statement)
                self._log_collection_attempt(stock_id, "financial_statement", "success", f"Collected {len(financial_statement)} rows")
                time.sleep(random.uniform(1.0, 2.0))  # Small delay between requests
                
            except Exception as e:
                logger.error(f"Error collecting financial statement for {stock_id}: {e}")
                self._log_collection_attempt(stock_id, "financial_statement", "error", str(e))
            
            # Get balance sheet data
            try:
                self._check_rate_limit()  # Check rate limit before API call
                balance_sheet = self.finmind.taiwan_stock_balance_sheet(
                    stock_id=stock_id,
                    start_date=start_date,
                    end_date=end_date
                )
                
                self._save_balance_sheet(stock_id, balance_sheet)
                self._log_collection_attempt(stock_id, "balance_sheet", "success", f"Collected {len(balance_sheet)} rows")
                time.sleep(random.uniform(1.0, 2.0))  # Small delay between requests
                
            except Exception as e:
                logger.error(f"Error collecting balance sheet for {stock_id}: {e}")
                self._log_collection_attempt(stock_id, "balance_sheet", "error", str(e))
            
            # Get cash flow data
            try:
                self._check_rate_limit()  # Check rate limit before API call
                cash_flow = self.finmind.taiwan_stock_cash_flows_statement(
                    stock_id=stock_id,
                    start_date=start_date,
                    end_date=end_date
                )
                
                self._save_cash_flow(stock_id, cash_flow)
                self._log_collection_attempt(stock_id, "cash_flow", "success", f"Collected {len(cash_flow)} rows")
                time.sleep(random.uniform(1.0, 2.0))  # Small delay between requests
                
            except Exception as e:
                logger.error(f"Error collecting cash flow for {stock_id}: {e}")
                self._log_collection_attempt(stock_id, "cash_flow", "error", str(e))
            
        except Exception as e:
            logger.error(f"Error in financial data collection for {stock_id}: {e}")
    
    def _collect_and_save_price_data(self, stock_id: str):
        """Collect and save price data for a single stock."""
        try:
            # Calculate date range - go back 5 years
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=5*365)).strftime('%Y-%m-%d')
            
            # Get price data
            try:
                self._check_rate_limit()  # Check rate limit before API call
                price_data = self.finmind.taiwan_stock_daily(
                    stock_id=stock_id,
                    start_date=start_date,
                    end_date=end_date
                )
                
                self._save_price_data(stock_id, price_data)
                self._log_collection_attempt(stock_id, "price_data", "success", f"Collected {len(price_data)} rows")
                
            except Exception as e:
                logger.error(f"Error collecting price data for {stock_id}: {e}")
                self._log_collection_attempt(stock_id, "price_data", "error", str(e))
            
        except Exception as e:
            logger.error(f"Error in price data collection for {stock_id}: {e}")
    
    def _save_financial_statement(self, stock_id: str, data: pd.DataFrame):
        """Save financial statement data to the database."""
        if data.empty:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for _, row in data.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO financial_statements (stock_id, date, metric_type, value)
                    VALUES (?, ?, ?, ?)
                ''', (stock_id, row['date'], row['type'], float(row['value'])))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving financial statement for {stock_id}: {e}")
    
    def _save_balance_sheet(self, stock_id: str, data: pd.DataFrame):
        """Save balance sheet data to the database."""
        if data.empty:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for _, row in data.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO balance_sheets (stock_id, date, metric_type, value)
                    VALUES (?, ?, ?, ?)
                ''', (stock_id, row['date'], row['type'], float(row['value'])))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving balance sheet for {stock_id}: {e}")
    
    def _save_cash_flow(self, stock_id: str, data: pd.DataFrame):
        """Save cash flow data to the database."""
        if data.empty:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for _, row in data.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO cash_flows (stock_id, date, metric_type, value)
                    VALUES (?, ?, ?, ?)
                ''', (stock_id, row['date'], row['type'], float(row['value'])))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving cash flow for {stock_id}: {e}")
    
    def _save_price_data(self, stock_id: str, data: pd.DataFrame):
        """Save price data to the database."""
        if data.empty:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for _, row in data.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO stock_prices (stock_id, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stock_id, 
                    row['date'], 
                    float(row['open']), 
                    float(row['max']), 
                    float(row['min']), 
                    float(row['close']), 
                    int(row['Trading_Volume'])
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving price data for {stock_id}: {e}")
    
    def _log_collection_attempt(self, stock_id: str, data_type: str, status: str, message: str):
        """Log a collection attempt to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO collection_log (timestamp, stock_id, data_type, status, message)
                VALUES (?, ?, ?, ?, ?)
            ''', (now, stock_id, data_type, status, message))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging collection attempt: {e}")
    
    # ---- Data retrieval methods ----
    
    def get_financial_data(self, stock_id: str) -> Dict[str, pd.DataFrame]:
        """Get all financial data for a stock from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get financial statement data
            financial_stmt = pd.read_sql_query(
                "SELECT date, metric_type as type, value FROM financial_statements WHERE stock_id = ? ORDER BY date",
                conn,
                params=(stock_id,)
            )
            
            # Get balance sheet data
            balance_sheet = pd.read_sql_query(
                "SELECT date, metric_type as type, value FROM balance_sheets WHERE stock_id = ? ORDER BY date",
                conn,
                params=(stock_id,)
            )
            
            # Get cash flow data
            cash_flow = pd.read_sql_query(
                "SELECT date, metric_type as type, value FROM cash_flows WHERE stock_id = ? ORDER BY date",
                conn,
                params=(stock_id,)
            )
            
            # Get price data
            price_data = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM stock_prices WHERE stock_id = ? ORDER BY date",
                conn,
                params=(stock_id,)
            )
            
            conn.close()
            
            # Create the return dict in the same format as the original collector
            return {
                'financial_statement': financial_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'price_data': price_data
            }
            
        except Exception as e:
            logger.error(f"Error getting financial data for {stock_id}: {e}")
            return {
                'financial_statement': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame(),
                'price_data': pd.DataFrame()
            }
    
    def get_industry_data(self, industry: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Get all financial data for stocks in a specific industry."""
        try:
            # Get all stocks in this industry
            conn = sqlite3.connect(self.db_path)
            stocks = pd.read_sql_query(
                "SELECT stock_id FROM stock_info WHERE industry = ?",
                conn,
                params=(industry,)
            )
            conn.close()
            
            if stocks.empty:
                logger.warning(f"No stocks found for industry: {industry}")
                return {}
            
            # Get data for each stock
            industry_data = {}
            for _, row in stocks.iterrows():
                stock_id = row['stock_id']
                data = self.get_financial_data(stock_id)
                
                # Only add if we have meaningful data
                if not data['financial_statement'].empty and not data['balance_sheet'].empty:
                    industry_data[stock_id] = data
            
            logger.info(f"Retrieved data for {len(industry_data)} stocks in {industry} industry")
            return industry_data
            
        except Exception as e:
            logger.error(f"Error getting industry data for {industry}: {e}")
            return {}
    
    def get_collection_status(self) -> pd.DataFrame:
        """Get collection status information from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get latest successful collection for each stock and data type
            status_df = pd.read_sql_query('''
                SELECT s.stock_id, s.stock_name, s.industry, 
                       MAX(CASE WHEN l.data_type = 'financial_statement' AND l.status = 'success' THEN l.timestamp ELSE NULL END) as fs_last_update,
                       MAX(CASE WHEN l.data_type = 'balance_sheet' AND l.status = 'success' THEN l.timestamp ELSE NULL END) as bs_last_update,
                       MAX(CASE WHEN l.data_type = 'cash_flow' AND l.status = 'success' THEN l.timestamp ELSE NULL END) as cf_last_update,
                       MAX(CASE WHEN l.data_type = 'price_data' AND l.status = 'success' THEN l.timestamp ELSE NULL END) as price_last_update
                FROM stock_info s
                LEFT JOIN collection_log l ON s.stock_id = l.stock_id
                GROUP BY s.stock_id
                ORDER BY s.industry, s.stock_id
            ''', conn)
            
            conn.close()
            
            return status_df
            
        except Exception as e:
            logger.error(f"Error getting collection status: {e}")
            return pd.DataFrame()
    
    def export_to_csv(self, output_dir: str = "exported_data"):
        """Export all data from the database to CSV files."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            
            # Export stock info
            stock_info = pd.read_sql_query("SELECT * FROM stock_info", conn)
            stock_info.to_csv(os.path.join(output_dir, "stock_info.csv"), index=False)
            
            # Export collection log
            collection_log = pd.read_sql_query("SELECT * FROM collection_log", conn)
            collection_log.to_csv(os.path.join(output_dir, "collection_log.csv"), index=False)
            
            # Get all unique stock IDs
            stocks = stock_info['stock_id'].unique()
            
            # Create industry directory for organized export
            industry_dir = os.path.join(output_dir, "by_industry")
            os.makedirs(industry_dir, exist_ok=True)
            
            # Group by industry for easier data organization
            for industry in stock_info['industry'].unique():
                industry_stocks = stock_info[stock_info['industry'] == industry]['stock_id'].tolist()
                if not industry_stocks:
                    continue
                
                ind_dir = os.path.join(industry_dir, industry.replace(' ', '_').lower())
                os.makedirs(ind_dir, exist_ok=True)
                
                # Export industry data to JSON for easier consumption
                industry_data = {}
                for stock_id in industry_stocks:
                    data = self.get_financial_data(stock_id)
                    
                    # Convert DataFrames to dictionaries for JSON serialization
                    stock_data = {}
                    for key, df in data.items():
                        if not df.empty:
                            stock_data[key] = df.to_dict(orient='records')
                    
                    if stock_data:
                        industry_data[stock_id] = stock_data
                
                # Save industry data to JSON file
                with open(os.path.join(ind_dir, f"{industry.replace(' ', '_').lower()}_data.json"), 'w') as f:
                    json.dump(industry_data, f)
            
            conn.close()
            
            logger.info(f"Exported all data to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data to CSV: {e}")
            return False
    
    def get_db_stats(self) -> Dict:
        """Get statistics about the database contents."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Get table counts
            for table in ['stock_info', 'financial_statements', 'balance_sheets', 'cash_flows', 'stock_prices', 'collection_log']:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Get industry counts
            cursor.execute("SELECT industry, COUNT(*) FROM stock_info GROUP BY industry")
            stats['industries'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get collection success rate
            cursor.execute("SELECT status, COUNT(*) FROM collection_log GROUP BY status")
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            total_attempts = sum(status_counts.values())
            if total_attempts > 0:
                stats['success_rate'] = (status_counts.get('success', 0) / total_attempts) * 100
            else:
                stats['success_rate'] = 0
            
            # Get stocks with complete data
            cursor.execute('''
                SELECT COUNT(DISTINCT s.stock_id)
                FROM stock_info s
                WHERE EXISTS (SELECT 1 FROM financial_statements fs WHERE fs.stock_id = s.stock_id)
                AND EXISTS (SELECT 1 FROM balance_sheets bs WHERE bs.stock_id = s.stock_id)
                AND EXISTS (SELECT 1 FROM cash_flows cf WHERE cf.stock_id = s.stock_id)
                AND EXISTS (SELECT 1 FROM stock_prices sp WHERE sp.stock_id = s.stock_id)
            ''')
            stats['stocks_with_complete_data'] = cursor.fetchone()[0]
            
            conn.close()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Create a background collector instance
    collector = BackgroundDataCollector(
        db_path="finance_data.db",
        collection_interval=1  # hours
    )
    
    # Start the collection scheduler (runs immediately and then every 12 hours)
    collector.start_scheduler()
    
    try:
        # Run for a period of time (e.g., 24 hours)
        print("Background collector started. Press Ctrl+C to stop...")
        time.sleep(24 * 60 * 60)  # Run for 24 hours 
    except KeyboardInterrupt:
        print("Stopping background collector...")
    finally:
        # Stop the scheduler
        collector.stop_scheduler()
        
        # Export the collected data to CSV
        collector.export_to_csv()
        
        # Show statistics
        stats = collector.get_db_stats()
        print("\nDatabase Statistics:")
        print(f"Total stocks: {stats.get('stock_info_count', 0)}")
        print(f"Financial statements: {stats.get('financial_statements_count', 0)}")
        print(f"Balance sheets: {stats.get('balance_sheets_count', 0)}")
        print(f"Cash flows: {stats.get('cash_flows_count', 0)}")
        print(f"Stock prices: {stats.get('stock_prices_count', 0)}")
        print(f"Collection success rate: {stats.get('success_rate', 0):.1f}%")
        print(f"Stocks with complete data: {stats.get('stocks_with_complete_data', 0)}")
