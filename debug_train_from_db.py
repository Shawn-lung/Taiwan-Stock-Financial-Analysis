#!/usr/bin/env python3
"""
Debug and fix issues with training models from database
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import logging
from typing import Dict, List
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum detail
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def inspect_database_tables(db_path="finance_data.db"):
    """Inspect database tables structure and column formats"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        logger.info(f"Found {len(tables)} tables in the database")
        
        # Check financial tables specifically
        financial_tables = ['financial_statements', 'balance_sheets', 'cash_flows']
        for table in financial_tables:
            cursor.execute(f"SELECT * FROM {table} LIMIT 1")
            columns = [col[0] for col in cursor.description]
            logger.info(f"Table {table} columns: {columns}")
            
            # Check if table has type/metric_type column
            type_col = None
            if 'type' in columns:
                type_col = 'type'
            elif 'metric_type' in columns:
                type_col = 'metric_type'
            
            if type_col:
                cursor.execute(f"SELECT DISTINCT {type_col} FROM {table} LIMIT 10")
                types = cursor.fetchall()
                logger.info(f"Sample {type_col} values in {table}: {[t[0] for t in types]}")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error inspecting database tables: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def create_direct_training_data(db_path="finance_data.db", industry="Semiconductors"):
    """Extract financial metrics directly from database tables and create training dataset"""
    try:
        conn = sqlite3.connect(db_path)
        
        # Get stocks in the specified industry
        stocks = pd.read_sql_query(
            "SELECT stock_id, stock_name FROM stock_info WHERE industry = ?", 
            conn, 
            params=(industry,)
        )
        
        if stocks.empty:
            logger.error(f"No stocks found for industry: {industry}")
            return None
            
        logger.info(f"Found {len(stocks)} stocks in {industry} industry")
        logger.info(f"Sample stocks: {stocks['stock_id'].tolist()[:5]}")
        
        # Initialize list to store financial records
        all_records = []
        
        # Process each stock
        for _, stock in stocks.iterrows():
            stock_id = stock['stock_id']
            logger.info(f"Processing {stock_id} ({stock['stock_name']})")
            
            # Get financial statements for this stock
            fs_data = pd.read_sql_query(
                "SELECT * FROM financial_statements WHERE stock_id = ?",
                conn,
                params=(stock_id,)
            )
            
            # Get balance sheets for this stock
            bs_data = pd.read_sql_query(
                "SELECT * FROM balance_sheets WHERE stock_id = ?",
                conn,
                params=(stock_id,)
            )
            
            # Get cash flows for this stock
            cf_data = pd.read_sql_query(
                "SELECT * FROM cash_flows WHERE stock_id = ?",
                conn,
                params=(stock_id,)
            )
            
            # Get price data for this stock
            price_data = pd.read_sql_query(
                "SELECT * FROM stock_prices WHERE stock_id = ?",
                conn,
                params=(stock_id,)
            )
            
            # Log data availability
            logger.info(f"{stock_id} data: FS={len(fs_data)}, BS={len(bs_data)}, CF={len(cf_data)}, Price={len(price_data)}")
            
            # Skip if we don't have enough data
            if fs_data.empty or bs_data.empty:
                logger.warning(f"Insufficient data for {stock_id}, skipping")
                continue
                
            # Determine which column contains metric types
            type_col = None
            if 'type' in fs_data.columns:
                type_col = 'type'
                logger.info(f"Using 'type' column for metrics in {stock_id}")
            elif 'metric_type' in fs_data.columns:
                type_col = 'metric_type'
                logger.info(f"Using 'metric_type' column for metrics in {stock_id}")
            else:
                logger.error(f"No type column found in financial statements for {stock_id}")
                continue
            
            # Get unique report dates
            fs_dates = sorted(fs_data['date'].unique())
            logger.info(f"Found {len(fs_dates)} reporting periods for {stock_id}")
            
            # Process each reporting period
            for report_date in fs_dates:
                # Filter data for this date
                period_fs = fs_data[fs_data['date'] == report_date]
                period_bs = bs_data[bs_data['date'] == report_date] if not bs_data.empty else pd.DataFrame()
                period_cf = cf_data[cf_data['date'] == report_date] if not cf_data.empty else pd.DataFrame()
                
                # Skip if we're missing important data for this period
                if period_fs.empty or period_bs.empty:
                    continue
                
                # Create a record for this period
                record = {
                    'stock_id': stock_id,
                    'date': report_date,
                    'timestamp': pd.to_datetime(report_date),
                }
                
                # Extract revenue
                revenue = None
                for revenue_type in ['Revenue', 'OperatingRevenue', 'NetRevenue', 'TotalRevenue']:
                    rev_rows = period_fs[period_fs[type_col] == revenue_type]
                    if not rev_rows.empty:
                        try:
                            revenue = float(rev_rows['value'].iloc[0])
                            if revenue > 0:
                                break
                        except:
                            pass
                
                # Skip if no valid revenue
                if revenue is None or revenue <= 0:
                    continue
                    
                record['revenue'] = revenue
                
                # Extract operating income
                operating_income = None
                for op_type in ['OperatingIncome', 'OperatingProfit', 'GrossProfit']:
                    op_rows = period_fs[period_fs[type_col] == op_type]
                    if not op_rows.empty:
                        try:
                            operating_income = float(op_rows['value'].iloc[0])
                            break
                        except:
                            pass
                
                # Calculate operating margin
                if operating_income is not None:
                    record['operating_income'] = operating_income
                    record['operating_margin'] = operating_income / revenue
                
                # Extract net income
                net_income = None
                for net_type in ['NetIncome', 'ProfitAfterTax', 'NetProfit', 'NetIncomeLoss']:
                    net_rows = period_fs[period_fs[type_col] == net_type]
                    if not net_rows.empty:
                        try:
                            net_income = float(net_rows['value'].iloc[0])
                            break
                        except:
                            pass
                
                # Calculate net margin
                if net_income is not None:
                    record['net_income'] = net_income
                    record['net_margin'] = net_income / revenue
                
                # Extract balance sheet metrics
                # Total assets
                total_assets = None
                for asset_type in ['TotalAssets', 'Assets', 'ConsolidatedTotalAssets']:
                    asset_rows = period_bs[period_bs[type_col] == asset_type]
                    if not asset_rows.empty:
                        try:
                            total_assets = float(asset_rows['value'].iloc[0])
                            if total_assets > 0:
                                break
                        except:
                            pass
                
                if total_assets is not None and total_assets > 0:
                    record['total_assets'] = total_assets
                    # Calculate ROA
                    if net_income is not None:
                        record['roa'] = net_income / total_assets
                
                # Total equity
                total_equity = None
                for equity_type in ['TotalEquity', 'StockholdersEquity', 'Equity', 'TotalStockholdersEquity']:
                    equity_rows = period_bs[period_bs[type_col] == equity_type]
                    if not equity_rows.empty:
                        try:
                            total_equity = float(equity_rows['value'].iloc[0])
                            if total_equity > 0:
                                break
                        except:
                            pass
                
                if total_equity is not None and total_equity > 0:
                    record['total_equity'] = total_equity
                    # Calculate ROE
                    if net_income is not None:
                        record['roe'] = net_income / total_equity
                    
                    # Calculate debt-to-equity
                    if total_assets is not None:
                        total_liabilities = total_assets - total_equity
                        record['debt_to_equity'] = total_liabilities / total_equity
                        record['equity_to_assets'] = total_equity / total_assets
                
                # Add dummy future return (will be replaced with real data if available)
                record['future_6m_return'] = 0.05
                
                # Calculate real future return from price data if available
                if not price_data.empty:
                    # Convert report date to datetime
                    report_dt = pd.to_datetime(report_date)
                    
                    # Make sure price_data date is datetime
                    if not pd.api.types.is_datetime64_any_dtype(price_data['date']):
                        price_data['date'] = pd.to_datetime(price_data['date'])
                    
                    # Get prices after report date
                    future_prices = price_data[price_data['date'] >= report_dt]
                    
                    if not future_prices.empty:
                        # Get initial price (closest to report date)
                        start_price = future_prices.iloc[0]['close']
                        
                        # Get price 6 months later
                        future_date_6m = report_dt + pd.DateOffset(months=6)
                        future_prices_6m = price_data[price_data['date'] >= future_date_6m]
                        
                        if not future_prices_6m.empty:
                            future_price_6m = future_prices_6m.iloc[0]['close']
                            # Calculate return
                            if start_price > 0:
                                return_6m = (future_price_6m - start_price) / start_price
                                record['future_6m_return'] = return_6m
                
                # Add historical growth if we can calculate it
                if len(all_records) > 0:
                    # Check for previous records of this stock
                    prev_records = [r for r in all_records if r['stock_id'] == stock_id]
                    if prev_records:
                        growth_rates = []
                        for prev in prev_records:
                            if 'revenue' in prev and prev['revenue'] > 0:
                                growth_rate = (revenue - prev['revenue']) / prev['revenue']
                                growth_rates.append(growth_rate)
                        
                        if growth_rates:
                            record['historical_growth'] = np.mean(growth_rates)
                            record['historical_growth_mean'] = np.mean(growth_rates)
                            record['historical_growth_std'] = np.std(growth_rates) if len(growth_rates) > 1 else 0.1
                
                # Add record if it has minimum required fields
                min_fields = ['revenue']
                useful_fields = ['operating_margin', 'net_margin', 'roe', 'roa']
                
                if all(f in record for f in min_fields) and any(f in record for f in useful_fields):
                    all_records.append(record)
                    logger.debug(f"Added record for {stock_id} on {report_date}")
        
        # Create dataframe from all records
        if not all_records:
            logger.error(f"No valid financial records found for {industry}")
            return None
            
        df = pd.DataFrame(all_records)
        
        # Fill in missing historical growth
        if 'historical_growth_mean' not in df.columns:
            df['historical_growth_mean'] = 0.05  # Default value
        if 'historical_growth_std' not in df.columns:
            df['historical_growth_std'] = 0.02  # Default value
            
        logger.info(f"Created dataset with {len(df)} records for {industry}")
        
        # Create directory if it doesn't exist
        os.makedirs("industry_data_from_db", exist_ok=True)
        os.makedirs("industry_data_from_db/models", exist_ok=True)
        
        # Save to CSV
        output_file = f"industry_data_from_db/{industry.lower().replace(' ', '_')}_training.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved training data to {output_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating training data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def train_simple_industry_model(industry, data=None):
    """Train a simple model for an industry"""
    try:
        # Use provided data or load from CSV
        if data is None:
            csv_file = f"industry_data_from_db/{industry.lower().replace(' ', '_')}_training.csv"
            if not os.path.exists(csv_file):
                logger.error(f"Training data file not found: {csv_file}")
                return False
            
            data = pd.read_csv(csv_file)
        
        if data.empty:
            logger.error(f"No data available for {industry}")
            return False
            
        logger.info(f"Training model for {industry} with {len(data)} records")
        
        # Select features
        potential_features = [
            'revenue', 'operating_margin', 'net_margin', 'roa', 'roe',
            'historical_growth_mean', 'historical_growth', 'debt_to_equity'
        ]
        
        # Use features that exist in the data
        features = [f for f in potential_features if f in data.columns]
        
        if len(features) < 2:
            logger.error(f"Not enough features available for {industry}")
            return False
            
        logger.info(f"Using features: {features}")
        
        # Prepare features and target
        X = data[features].copy()
        
        # Ensure future_6m_return exists
        if 'future_6m_return' not in data.columns:
            logger.warning(f"No future_6m_return in data, using synthetic values")
            # Generate synthetic target based on features
            data['future_6m_return'] = 0.05 + 0.1 * data['operating_margin'] - 0.05 * np.random.random(len(data))
        
        y = data['future_6m_return']
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(0.05)  # Default expected return
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create simple model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(8, activation='relu', input_shape=(X.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        
        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        history = model.fit(
            X_scaled, y,
            epochs=50,
            batch_size=min(16, len(X)),
            validation_split=0.2 if len(X) > 10 else 0,
            verbose=1
        )
        
        # Save model and scaler
        model_dir = "industry_data_from_db/models"
        os.makedirs(model_dir, exist_ok=True)
        
        # FIX: Add .keras extension to model path
        model_path = os.path.join(model_dir, f"{industry.lower().replace(' ', '_')}_model.keras")
        scaler_path = os.path.join(model_dir, f"{industry.lower().replace(' ', '_')}_scaler.pkl")
        
        model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
        logger.info(f"Model saved to {model_path}")
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Training Loss for {industry}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join("industry_data_from_db", f"{industry.lower().replace(' ', '_')}_loss.png"))
        
        logger.info(f"Successfully trained model for {industry}")
        return True
        
    except Exception as e:
        logger.error(f"Error training model for {industry}: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to debug and fix training issues"""
    logger.info("Starting training debug process")
    
    # Step 1: Inspect database tables
    inspect_database_tables()
    
    # Step 2: Create training data for key industries
    industries_to_process = [
        "Semiconductors",
        "Electronics",
        "Computer Hardware",
        "Financial Services",
        "Telecommunications"
    ]
    
    successful_industries = []
    for industry in industries_to_process:
        logger.info(f"Processing {industry} industry")
        data = create_direct_training_data(industry=industry)
        
        if data is not None and not data.empty:
            # Step 3: Train model for this industry
            if train_simple_industry_model(industry, data):
                successful_industries.append(industry)
    
    # Summary
    logger.info("Training debug process completed")
    logger.info(f"Successfully processed {len(successful_industries)} industries: {successful_industries}")
    
    if successful_industries:
        logger.info("You can now run your main training script, which should work with these prepared datasets")

if __name__ == "__main__":
    main()
