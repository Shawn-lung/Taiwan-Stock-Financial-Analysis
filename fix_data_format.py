#!/usr/bin/env python3
"""
Fix data format issues and diagnose data structure problems in the database.
"""

import os
import sqlite3
import pandas as pd
import logging
import pickle
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_fix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def backup_database(db_path="finance_data.db"):
    """Create a backup of the database before modifications."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{db_path}.backup_{timestamp}"
        shutil.copy2(db_path, backup_path)
        logger.info(f"Created database backup at {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create database backup: {e}")
        return False

def cleanup_industry_dir(data_dir="industry_data_from_db"):
    """Clean up the industry data directory to start fresh."""
    try:
        if os.path.exists(data_dir):
            # Delete CSV files
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    os.remove(os.path.join(data_dir, file))
                    logger.info(f"Deleted {file}")
                
            # Delete models subdirectory if it exists
            models_dir = os.path.join(data_dir, "models")
            if os.path.exists(models_dir):
                for file in os.listdir(models_dir):
                    os.remove(os.path.join(models_dir, file))
                    logger.info(f"Deleted {file} from models directory")
                os.rmdir(models_dir)
                logger.info("Deleted models directory")
                
            logger.info(f"Cleaned up {data_dir} directory")
        else:
            os.makedirs(data_dir)
            logger.info(f"Created {data_dir} directory")
            
        return True
    except Exception as e:
        logger.error(f"Error cleaning up industry directory: {e}")
        return False

def inspect_database_structure(db_path="finance_data.db"):
    """Inspect the database structure to understand data formats."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        data_formats = {}
        table_counts = {}
        
        for table_name in [t[0] for t in tables]:
            # Get sample data format
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1;")
            columns = [description[0] for description in cursor.description]
            
            # Count rows
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            table_counts[table_name] = row_count
            
            # Store column info
            data_formats[table_name] = columns
            
            # Special handling for financial tables
            if table_name in ['financial_statements', 'balance_sheets', 'cash_flows']:
                # Check if 'metric_type' or 'type' column exists
                if 'metric_type' in columns:
                    # Get unique metric types
                    cursor.execute(f"SELECT DISTINCT metric_type FROM {table_name} LIMIT 20;")
                    metric_types = cursor.fetchall()
                    logger.info(f"{table_name} uses 'metric_type' column with values like: {metric_types[:5]}")
                elif 'type' in columns:
                    # Get unique types
                    cursor.execute(f"SELECT DISTINCT type FROM {table_name} LIMIT 20;")
                    types = cursor.fetchall()
                    logger.info(f"{table_name} uses 'type' column with values like: {types[:5]}")
        
        conn.close()
        
        logger.info("Database Structure Summary:")
        for table, cols in data_formats.items():
            logger.info(f"- {table}: {table_counts.get(table, 0)} rows, columns: {cols}")
            
        return data_formats, table_counts
    except Exception as e:
        logger.error(f"Error inspecting database: {e}")
        return {}, {}

def create_sample_training_data(db_path="finance_data.db", industry="Semiconductors", output_dir="industry_data_from_db"):
    """Create a sample training dataset for one industry to test the process."""
    try:
        conn = sqlite3.connect(db_path)
        
        # Get stocks in the specified industry
        industry_stocks = pd.read_sql_query(
            "SELECT stock_id FROM stock_info WHERE industry = ?",
            conn, 
            params=(industry,)
        )
        
        if industry_stocks.empty:
            logger.error(f"No stocks found in {industry} industry")
            return False
            
        logger.info(f"Found {len(industry_stocks)} stocks in {industry} industry")
        
        stock_ids = industry_stocks['stock_id'].tolist()
        
        if len(stock_ids) < 3:
            logger.warning(f"Too few stocks in {industry} industry")
            return False
        
        # Create records for each stock's financial data
        records = []
        
        for stock_id in stock_ids[:5]:  # Limit to 5 stocks for sample
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
            
            # Get dates with both financial statement and balance sheet data
            if not financial_stmt.empty and not balance_sheet.empty:
                dates = sorted(pd.Series(list(set(financial_stmt['date']) & set(balance_sheet['date']))))
                
                for date in dates:
                    record = {
                        'stock_id': stock_id,
                        'timestamp': pd.to_datetime(date),
                        'revenue': 0,
                        'operating_margin': 0,
                        'net_margin': 0,
                        'roa': 0,
                        'roe': 0,
                        'historical_growth_mean': 0.05,  # Sample values
                        'future_6m_return': 0.03,  # Sample values
                    }
                    
                    # Extract financial metrics using database columns
                    fs_date = financial_stmt[financial_stmt['date'] == date]
                    bs_date = balance_sheet[balance_sheet['date'] == date]
                    
                    # Check column format - use either 'metric_type' or 'type'
                    type_col = 'metric_type' if 'metric_type' in financial_stmt.columns else 'type'
                    
                    # Get revenue
                    for revenue_type in ['Revenue', 'OperatingRevenue']:
                        rev_rows = fs_date[fs_date[type_col] == revenue_type]
                        if not rev_rows.empty:
                            record['revenue'] = float(rev_rows['value'].iloc[0])
                            break
                    
                    # Get operating income and margin
                    for op_type in ['OperatingIncome', 'OperatingProfit']:
                        op_rows = fs_date[fs_date[type_col] == op_type]
                        if not op_rows.empty:
                            op_income = float(op_rows['value'].iloc[0])
                            if record['revenue'] > 0:
                                record['operating_margin'] = op_income / record['revenue']
                            break
                    
                    # Get net income and margin
                    for net_type in ['NetIncome', 'ProfitAfterTax']:
                        net_rows = fs_date[fs_date[type_col] == net_type]
                        if not net_rows.empty:
                            net_income = float(net_rows['value'].iloc[0])
                            if record['revenue'] > 0:
                                record['net_margin'] = net_income / record['revenue']
                            break
                    
                    # Add to records if we have the minimum required data
                    if record['revenue'] > 0:
                        records.append(record)
        
        # Create DataFrame and save to CSV
        if records:
            df = pd.DataFrame(records)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{industry.lower().replace(' ', '_')}_training.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"Created sample training file at {output_file} with {len(df)} records")
            return True
        else:
            logger.error("No valid records created")
            return False
        
    except Exception as e:
        logger.error(f"Error creating sample training data: {e}")
        return False

def main():
    """Main function to diagnose and fix data issues."""
    logger.info("Starting data format diagnosis and fix")
    
    # First create a database backup
    backup_database()
    
    # Inspect the database structure
    inspect_database_structure()
    
    # Clean up industry data directory
    cleanup_industry_dir()
    
    # Create sample training data for Semiconductors industry
    create_sample_training_data(industry="Semiconductors")
    create_sample_training_data(industry="Electronics")
    
    logger.info("Data diagnosis and fix completed")

if __name__ == "__main__":
    main()
