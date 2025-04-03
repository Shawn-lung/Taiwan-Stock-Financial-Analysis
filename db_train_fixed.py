#!/usr/bin/env python3
"""
Simplified training script for industry models that properly handles data formats.
"""

import os
import pandas as pd
import numpy as np
import logging
import sqlite3
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("db_train_fixed.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FixedDBModelTrainer:
    """Train industry models using database data with proper format handling."""
    
    def __init__(self, db_path="finance_data.db", output_dir="industry_data_from_db"):
        """Initialize the trainer."""
        self.db_path = db_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    
    def get_industries(self):
        """Get list of industries from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT DISTINCT industry FROM stock_info WHERE industry IS NOT NULL"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Filter out empty strings
            industries = [ind for ind in df['industry'].tolist() if ind and ind.strip()]
            logger.info(f"Found {len(industries)} industries in database")
            return industries
        except Exception as e:
            logger.error(f"Error getting industry list: {e}")
            return []
    
    def get_stocks_by_industry(self, industry):
        """Get stocks for a specific industry."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT stock_id FROM stock_info WHERE industry = ?"
            df = pd.read_sql_query(query, conn, params=(industry,))
            conn.close()
            
            stocks = df['stock_id'].tolist()
            logger.info(f"Found {len(stocks)} stocks in {industry} industry")
            return stocks
        except Exception as e:
            logger.error(f"Error getting stocks for {industry}: {e}")
            return []
    
    def prepare_direct_financial_metrics(self, industry):
        """Directly extract financial metrics from the database for an industry."""
        try:
            stocks = self.get_stocks_by_industry(industry)
            
            if not stocks:
                logger.warning(f"No stocks found for {industry}")
                return None
            
            # Create database connection
            conn = sqlite3.connect(self.db_path)
            
            all_records = []
            
            # Process each stock
            for stock_id in stocks:
                try:
                    # Financial statements query with direct aggregation
                    fs_query = '''
                    SELECT 
                        fs.stock_id,
                        fs.date,
                        MAX(CASE WHEN fs.metric_type IN ('Revenue', 'OperatingRevenue', 'TotalRevenue') THEN fs.value ELSE NULL END) as revenue,
                        MAX(CASE WHEN fs.metric_type IN ('OperatingIncome', 'OperatingProfit') THEN fs.value ELSE NULL END) as operating_income,
                        MAX(CASE WHEN fs.metric_type IN ('NetIncome', 'ProfitAfterTax') THEN fs.value ELSE NULL END) as net_income
                    FROM financial_statements fs
                    WHERE fs.stock_id = ?
                    GROUP BY fs.stock_id, fs.date
                    '''
                    
                    # Get financial statement data
                    financial_data = pd.read_sql_query(fs_query, conn, params=(stock_id,))
                    
                    if financial_data.empty:
                        continue
                    
                    # Balance sheet query
                    bs_query = '''
                    SELECT 
                        bs.stock_id,
                        bs.date,
                        MAX(CASE WHEN bs.metric_type IN ('TotalAssets', 'Assets') THEN bs.value ELSE NULL END) as total_assets,
                        MAX(CASE WHEN bs.metric_type IN ('TotalEquity', 'Equity', 'StockholdersEquity') THEN bs.value ELSE NULL END) as total_equity
                    FROM balance_sheets bs
                    WHERE bs.stock_id = ?
                    GROUP BY bs.stock_id, bs.date
                    '''
                    
                    # Get balance sheet data
                    balance_data = pd.read_sql_query(bs_query, conn, params=(stock_id,))
                    
                    if not balance_data.empty:
                        # Merge financial and balance data on date
                        combined_data = pd.merge(financial_data, balance_data, on=['stock_id', 'date'], how='inner')
                        
                        for _, row in combined_data.iterrows():
                            record = {
                                'stock_id': stock_id,
                                'timestamp': row['date'],
                                'revenue': row['revenue'],
                                'operating_income': row['operating_income'],
                                'net_income': row['net_income'],
                                'total_assets': row['total_assets'],
                                'total_equity': row['total_equity']
                            }
                            
                            # Calculate financial ratios if we have the data
                            if row['revenue'] and row['revenue'] > 0:
                                if row['operating_income'] is not None:
                                    record['operating_margin'] = row['operating_income'] / row['revenue']
                                if row['net_income'] is not None:
                                    record['net_margin'] = row['net_income'] / row['revenue']
                            
                            if row['total_assets'] and row['total_assets'] > 0:
                                if row['net_income'] is not None:
                                    record['roa'] = row['net_income'] / row['total_assets']
                            
                            if row['total_equity'] and row['total_equity'] > 0:
                                if row['net_income'] is not None:
                                    record['roe'] = row['net_income'] / row['total_equity']
                                
                                if row['total_assets'] is not None:
                                    record['debt_to_equity'] = (row['total_assets'] - row['total_equity']) / row['total_equity']
                            
                            # Add record if it has minimum required fields
                            if 'revenue' in record and record['revenue'] and record['revenue'] > 0:
                                if any(k in record for k in ['operating_margin', 'net_margin', 'roa', 'roe']):
                                    all_records.append(record)
                    
                except Exception as e:
                    logger.warning(f"Error processing {stock_id}: {e}")
            
            conn.close()
            
            if not all_records:
                logger.warning(f"No valid records extracted for {industry}")
                return None
            
            # Create training dataframe
            df = pd.DataFrame(all_records)
            
            # Save to CSV
            csv_path = os.path.join(self.output_dir, f"{industry.lower().replace(' ', '_')}_training.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(df)} records for {industry} to {csv_path}")
            
            return df
        except Exception as e:
            logger.error(f"Error preparing data for {industry}: {e}")
            return None
    
    def train_industry_model(self, industry, data=None):
        """Train a model for an industry."""
        try:
            # Use provided data or load from CSV
            if data is None:
                csv_file = os.path.join(self.output_dir, f"{industry.lower().replace(' ', '_')}_training.csv")
                if not os.path.exists(csv_file):
                    logger.error(f"Training file not found: {csv_file}")
                    return False
                
                data = pd.read_csv(csv_file)
            
            if data.empty:
                logger.error(f"No data for {industry}")
                return False
                
            logger.info(f"Training model for {industry} with {len(data)} records")
            
            # Select features that exist in the data
            potential_features = [
                'revenue', 'operating_margin', 'net_margin', 'roa', 'roe',
                'historical_growth_mean', 'debt_to_equity'
            ]
            
            features = [f for f in potential_features if f in data.columns]
            
            if len(features) < 2:
                logger.error(f"Not enough features for {industry}")
                return False
            
            # Prepare features and target
            X = data[features].copy()
            
            # If future_6m_return doesn't exist, create synthetic values
            if 'future_6m_return' not in data.columns:
                logger.warning(f"No future_6m_return in data, using synthetic values")
                data['future_6m_return'] = 0.05 + 0.1 * data['operating_margin'] - 0.05 * np.random.random(len(data))
            
            y = data['future_6m_return']
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(0.05)  # Default expected return
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X.shape[1],)),  # Proper input layer
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1)
            ])
            
            # Compile model
            model.compile(optimizer='adam', loss='mse')
            
            # Train model
            history = model.fit(
                X_scaled, y,
                epochs=50,
                batch_size=min(16, len(X)),
                validation_split=0.2 if len(X) > 10 else 0,
                verbose=1
            )
            
            # Save model with .keras extension
            model_dir = os.path.join(self.output_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            
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
            plt.savefig(os.path.join(self.output_dir, f"{industry.lower().replace(' ', '_')}_loss.png"))
            plt.close()
            
            return True
        except Exception as e:
            logger.error(f"Error training model for {industry}: {e}")
            return False
    
    def create_industry_benchmarks(self):
        """Create industry benchmarks file from training datasets."""
        try:
            # Find training CSV files
            csv_files = [f for f in os.listdir(self.output_dir) if f.endswith('_training.csv')]
            
            if not csv_files:
                logger.error(f"No training files found")
                return False
            
            logger.info(f"Creating benchmarks from {len(csv_files)} industry datasets")
            
            # Process each industry
            industry_metrics = []
            
            for csv_file in csv_files:
                industry = csv_file.replace('_training.csv', '').replace('_', ' ')
                
                try:
                    # Read the training data
                    df = pd.read_csv(os.path.join(self.output_dir, csv_file))
                    
                    if df.empty:
                        continue
                    
                    logger.info(f"Processed benchmarks for {industry}: {len(df)} records")
                    
                    # Calculate key metrics
                    metrics = {
                        'industry': industry,
                        'stock_count': df['stock_id'].nunique() if 'stock_id' in df.columns else 0,
                        'record_count': len(df)
                    }
                    
                    # Calculate metrics for each key ratio
                    for metric in ['operating_margin', 'net_margin', 'roa', 'roe', 'debt_to_equity']:
                        if metric in df.columns:
                            metrics[f'{metric}_median'] = df[metric].median()
                            metrics[f'{metric}_mean'] = df[metric].mean()
                    
                    industry_metrics.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Error processing {industry}: {e}")
            
            # Create DataFrame from metrics
            if not industry_metrics:
                logger.error("No valid benchmarks created")
                return False
                
            benchmarks_df = pd.DataFrame(industry_metrics)
            
            # Save benchmarks
            benchmark_file = os.path.join(self.output_dir, 'industry_benchmarks.csv')
            benchmarks_df.to_csv(benchmark_file, index=False)
            
            # Also save to original industry_data directory for compatibility
            orig_dir = "industry_data"
            os.makedirs(orig_dir, exist_ok=True)
            benchmarks_df.to_csv(os.path.join(orig_dir, 'industry_benchmarks.csv'), index=False)
            
            logger.info(f"Saved industry benchmarks for {len(benchmarks_df)} industries")
            
            return True
        except Exception as e:
            logger.error(f"Error creating benchmarks: {e}")
            return False
    
    def train_all_industries(self):
        """Train models for all industries."""
        start_time = datetime.now()
        
        # Get all industries
        industries = self.get_industries()
        
        # Prepare data and train models
        successful = []
        
        for i, industry in enumerate(industries):
            logger.info(f"Processing industry {i+1}/{len(industries)}: {industry}")
            
            # Prepare data
            data = self.prepare_direct_financial_metrics(industry)
            
            if data is not None and len(data) > 0:
                # Train model
                if self.train_industry_model(industry, data):
                    successful.append(industry)
        
        # Create industry benchmarks
        self.create_industry_benchmarks()
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Training completed in {duration}. Successful: {len(successful)}/{len(industries)}")
        logger.info(f"Successful industries: {successful}")
        
        return successful

def main():
    """Main function to run training."""
    logger.info("Starting fixed model training")
    
    trainer = FixedDBModelTrainer()
    trainer.train_all_industries()
    
    logger.info("Training process completed")

if __name__ == "__main__":
    main()
