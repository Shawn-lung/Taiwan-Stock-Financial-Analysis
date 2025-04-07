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
    
    def get_financial_data(self, stock_id: str) -> list:
        """Get consolidated financial data for deep learning models.
        
        Args:
            stock_id: The stock ID to retrieve data for
            
        Returns:
            List of financial data points with consistent format for deep learning
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # First, check the actual column names in the financial_statements table
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(financial_statements)")
            fs_columns = [col[1] for col in cursor.fetchall()]
            
            cursor.execute("PRAGMA table_info(balance_sheets)")
            bs_columns = [col[1] for col in cursor.fetchall()]
            
            logger.debug(f"Financial statements columns: {fs_columns}")
            logger.debug(f"Balance sheets columns: {bs_columns}")
            
            # Find appropriate column names for revenue, operating_income, net_income
            revenue_col = next((col for col in fs_columns if 'revenue' in col.lower()), None)
            op_income_col = next((col for col in fs_columns if 'operating_income' in col.lower() or 'op_income' in col.lower()), None)
            net_income_col = next((col for col in fs_columns if 'net_income' in col.lower() or 'profit' in col.lower()), None)
            
            # Find appropriate column names for balance sheet data
            assets_col = next((col for col in bs_columns if 'total_assets' in col.lower() or 'assets' in col.lower()), None)
            equity_col = next((col for col in bs_columns if 'total_equity' in col.lower() or 'equity' in col.lower()), None)
            
            # Build a dynamic query based on available columns
            select_parts = [f"fs.stock_id, fs.date"]
            
            if revenue_col:
                select_parts.append(f"fs.{revenue_col} as revenue")
            else:
                select_parts.append("NULL as revenue")
                
            if op_income_col:
                select_parts.append(f"fs.{op_income_col} as operating_income")
            else:
                select_parts.append("NULL as operating_income")
                
            if net_income_col:
                select_parts.append(f"fs.{net_income_col} as net_income")
            else:
                select_parts.append("NULL as net_income")
                
            if assets_col:
                select_parts.append(f"bs.{assets_col} as total_assets")
            else:
                select_parts.append("NULL as total_assets")
                
            if equity_col:
                select_parts.append(f"bs.{equity_col} as total_equity")
            else:
                select_parts.append("NULL as total_equity")
                
            # Build and execute the query
            query = f"""
            SELECT {', '.join(select_parts)}
            FROM financial_statements fs
            LEFT JOIN balance_sheets bs ON fs.stock_id = bs.stock_id AND fs.date = bs.date
            WHERE fs.stock_id = ?
            ORDER BY fs.date DESC
            """
            
            logger.debug(f"Executing query: {query}")
            cursor.execute(query, (stock_id,))
            results = cursor.fetchall()
            
            conn.close()
            
            # Convert to list of dictionaries for easier processing
            financial_data = []
            for row in results:
                financial_data.append({
                    'stock_id': row[0],
                    'date': row[1],
                    'revenue': row[2],
                    'operating_income': row[3], 
                    'net_income': row[4],
                    'total_assets': row[5],
                    'total_equity': row[6]
                })
            
            logger.info(f"Retrieved {len(financial_data)} financial data points for {stock_id} from database")
            return financial_data
            
        except Exception as e:
            logger.error(f"Error retrieving financial data for {stock_id}: {e}")
            return []
    
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
