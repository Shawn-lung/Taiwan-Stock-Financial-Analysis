#!/usr/bin/env python3
import sqlite3
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inspect_database(db_path="finance_data.db"):
    """Inspect the database structure and sample data to help diagnose issues."""
    try:
        # Check if database file exists
        if not os.path.exists(db_path):
            logger.error(f"Database file {db_path} not found.")
            return
        
        logger.info(f"Inspecting database: {db_path}")
        
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logger.info(f"Found {len(tables)} tables in the database:")
        for table in tables:
            logger.info(f"  - {table[0]}")
        
        # Examine each table's structure and sample data
        for table_name in [t[0] for t in tables]:
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            logger.info(f"\nTable: {table_name}")
            logger.info(f"Columns ({len(columns)}):")
            for col in columns:
                logger.info(f"  - {col[1]}: {col[2]}")
            
            # Count rows
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            logger.info(f"Row count: {row_count}")
            
            # Get sample data
            if row_count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                sample = cursor.fetchall()
                logger.info(f"Sample data:")
                for row in sample:
                    logger.info(f"  {row}")
        
        # Special inspection for financial tables
        for table_name in ['financial_statements', 'balance_sheets', 'cash_flows']:
            if table_name in [t[0] for t in tables]:
                # Get unique stock IDs with data
                cursor.execute(f"SELECT DISTINCT stock_id FROM {table_name};")
                stock_ids = cursor.fetchall()
                logger.info(f"\n{table_name} has data for {len(stock_ids)} stocks")
                
                if len(stock_ids) > 0:
                    # Sample one stock's data structure
                    sample_stock = stock_ids[0][0]
                    df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE stock_id = ? LIMIT 10", conn, params=(sample_stock,))
                    logger.info(f"Sample data structure for {sample_stock} in {table_name}:")
                    logger.info(f"Columns: {df.columns.tolist()}")
                    
                    # Check if the table is in wide format (many columns) or long format (type, value)
                    if 'type' in df.columns and 'value' in df.columns:
                        logger.info(f"Table format: LONG format (type, value pairs)")
                        # Show unique metric types
                        cursor.execute(f"SELECT DISTINCT type FROM {table_name} LIMIT 20;")
                        types = [t[0] for t in cursor.fetchall()]
                        logger.info(f"Sample metric types: {types[:10]}{'...' if len(types) > 10 else ''}")
                    else:
                        logger.info(f"Table format: WIDE format (each metric as a column)")
                        # Log the metric columns
                        metric_cols = [col for col in df.columns if col not in ['stock_id', 'date']]
                        logger.info(f"Sample metric columns: {metric_cols[:10]}{'...' if len(metric_cols) > 10 else ''}")
        
        # Check for data between tables
        logger.info("\nChecking for stocks with data in multiple tables:")
        tables_to_check = []
        for table_name in ['financial_statements', 'balance_sheets', 'cash_flows', 'stock_prices']:
            if table_name in [t[0] for t in tables]:
                tables_to_check.append(table_name)
        
        if len(tables_to_check) >= 2:
            # Start with stocks from the first table
            query = f"SELECT DISTINCT stock_id FROM {tables_to_check[0]}"
            for table in tables_to_check[1:]:
                query = f"""
                    SELECT stock_id FROM ({query})
                    WHERE stock_id IN (SELECT DISTINCT stock_id FROM {table})
                """
            
            # Get stocks with data in all tables
            stocks_with_all_data = pd.read_sql_query(query, conn)
            logger.info(f"Found {len(stocks_with_all_data)} stocks with data in all financial tables.")
            
            if len(stocks_with_all_data) > 0:
                # Show sample stocks
                logger.info(f"Sample stocks with complete data: {stocks_with_all_data['stock_id'].tolist()[:5]}")
                
                # Check one sample stock in detail
                sample_stock = stocks_with_all_data['stock_id'].iloc[0]
                logger.info(f"\nDetailed inspection of sample stock {sample_stock}:")
                
                for table_name in tables_to_check:
                    df = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table_name} WHERE stock_id = ?", 
                                         conn, params=(sample_stock,))
                    row_count = df['count'].iloc[0]
                    logger.info(f"  - {table_name}: {row_count} rows")
                    
                    # Get sample row from each table
                    if row_count > 0:
                        sample_row = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE stock_id = ? LIMIT 1", 
                                                     conn, params=(sample_stock,))
                        logger.info(f"  - Sample data: {sample_row.to_dict('records')[0]}")
        
        conn.close()
        logger.info("\nDatabase inspection complete.")
        
    except Exception as e:
        logger.error(f"Error inspecting database: {e}")

if __name__ == "__main__":
    inspect_database("finance_data.db")
