import sqlite3
import pandas as pd
import logging
import os
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DBFinancialDataProvider:
    """Provider for financial data from SQLite database."""
    
    def __init__(self, db_path: str = "finance_data.db"):
        """Initialize the provider."""
        self.db_path = db_path
    
    def get_stock_data(self, stock_id: str) -> Dict:
        """Get financial data for a stock from the database."""
        try:
            # Create connection
            conn = sqlite3.connect(self.db_path)
            
            # Get financial statements
            financial_stmt = pd.read_sql_query(
                "SELECT * FROM financial_statements WHERE stock_id = ?",
                conn,
                params=(stock_id,)
            )
            
            # Get balance sheets
            balance_sheet = pd.read_sql_query(
                "SELECT * FROM balance_sheets WHERE stock_id = ?",
                conn,
                params=(stock_id,)
            )
            
            # Get cash flows
            cash_flow = pd.read_sql_query(
                "SELECT * FROM cash_flows WHERE stock_id = ?",
                conn,
                params=(stock_id,)
            )
            
            # Get stock prices
            price_data = pd.read_sql_query(
                "SELECT * FROM stock_prices WHERE stock_id = ?",
                conn,
                params=(stock_id,)
            )
            
            conn.close()
            
            # Return data dict (will never have empty attribute)
            return {
                'financial_statement': financial_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'price_data': price_data
            }
            
        except Exception as e:
            logger.error(f"Error getting data for {stock_id}: {e}")
            # Return empty DataFrames, not empty dict
            return {
                'financial_statement': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame(),
                'price_data': pd.DataFrame()
            }
    
    def get_industry(self, stock_id: str) -> Optional[str]:
        """Get industry classification for a stock."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.cursor()
            cursor.execute("SELECT industry FROM stock_info WHERE stock_id = ?", (stock_id,))
            result = cursor.fetchone()
            
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error getting industry for {stock_id}: {e}")
            return None
