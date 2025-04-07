#!/usr/bin/env python3
"""
Comprehensive financial model training script for industry-specific models
with advanced features including hyperparameter tuning, detailed analytics,
and robust data processing.
"""

import os
import pandas as pd
import numpy as np
import logging
import sqlite3
import tensorflow as tf

# Enable eager execution to fix the "numpy() is only available when eager execution is enabled" error
tf.config.run_functions_eagerly(True)
# Enable debug mode for tf.data operations to fix the warning about tf.data functions
tf.data.experimental.enable_debug_mode()

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path

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

class DBModelTrainer:
    """Advanced model trainer for financial data with comprehensive analytics and robust processing."""
    
    def __init__(self, db_path="finance_data.db", output_dir="industry_data_from_db", 
                 config_path=None, use_gpu=True):
        """Initialize the model trainer with advanced configuration options.
        
        Args:
            db_path: Path to the SQLite database
            output_dir: Directory to save training data and models
            config_path: Path to config file for hyperparameters (optional)
            use_gpu: Whether to use GPU for training (if available)
        """
        self.db_path = db_path
        self.output_dir = output_dir
        
        # Set up directories
        self.model_dir = os.path.join(output_dir, "models")
        self.chart_dir = os.path.join(output_dir, "charts")
        self.analysis_dir = os.path.join(output_dir, "analysis")
        
        for directory in [output_dir, self.model_dir, self.chart_dir, self.analysis_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Configure GPU usage
        self.use_gpu = use_gpu
        if use_gpu:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"Found {len(gpus)} GPU(s), enabling GPU acceleration")
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                logger.warning("No GPU found, using CPU instead")
        else:
            logger.info("GPU usage disabled, using CPU")
            tf.config.set_visible_devices([], 'GPU')
        
        # Load configuration if provided, otherwise use defaults
        self.config = self._load_config(config_path)
        
        # List of common financial metrics to extract
        self.metrics_to_extract = [
            'revenue', 'operating_margin', 'net_margin', 'roa', 'roe', 
            'debt_to_equity', 'equity_to_assets', 'operating_income',
            'net_income', 'total_assets', 'total_equity', 'historical_growth'
        ]
    
    def _load_config(self, config_path):
        """Load training configuration from JSON file, or use defaults."""
        default_config = {
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.2,
            'learning_rate': 0.001,
            'dropout_rate': 0.3,
            'l2_regularization': 0.001,
            'early_stopping_patience': 15,
            'hidden_layers': [16, 8],
            'use_cross_validation': True,
            'cv_folds': 5,
            'min_delta': 0.0001
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    config = {**default_config, **user_config}
                    logger.info(f"Loaded configuration from {config_path}")
                    return config
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
        
        logger.info("Using default configuration")
        return default_config
    
    def get_industries(self):
        """Get list of all industries in the database with data quality indicators."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Query to get industries with stock count
            query = """
            SELECT industry, COUNT(DISTINCT stock_id) as stock_count
            FROM stock_info
            WHERE industry IS NOT NULL AND industry != ''
            GROUP BY industry
            ORDER BY stock_count DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Filter out empty strings and format results
            industries = []
            for _, row in df.iterrows():
                if row['industry'] and row['industry'].strip():
                    industries.append({
                        'name': row['industry'],
                        'stock_count': row['stock_count']
                    })
            
            logger.info(f"Found {len(industries)} industries in database")
            return industries
        except Exception as e:
            logger.error(f"Error getting industry list: {e}")
            return []
    
    def get_stocks_by_industry(self, industry):
        """Get stock IDs for a specific industry."""
        try:
            conn = sqlite3.connect(self.db_path)
            # Simplified query that only requests stock_id and stock_name which are known to exist
            query = """
            SELECT stock_id, stock_name 
            FROM stock_info
            WHERE industry = ?
            """
            df = pd.read_sql_query(query, conn, params=(industry,))
            conn.close()
            
            logger.info(f"Found {len(df)} stocks in {industry} industry")
            
            # Return just the stock_ids list since that's what's used in subsequent processing
            if not df.empty:
                return df['stock_id'].tolist()
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting stocks for {industry}: {e}")
            return []
    
    def prepare_direct_financial_metrics(self, industry):
        """Extract and prepare financial metrics from the database with advanced analytics."""
        try:
            stock_ids = self.get_stocks_by_industry(industry)
            
            if not stock_ids:
                logger.warning(f"No stocks found for {industry}")
                return None
            
            # Create database connection
            conn = sqlite3.connect(self.db_path)
            
            all_records = []
            stocks_processed = 0
            stocks_with_data = 0
            
            # Process each stock
            for stock_id in stock_ids:
                try:
                    stocks_processed += 1
                    
                    # Financial statements query with direct aggregation and multiple metric types
                    fs_query = """
                    SELECT 
                        fs.stock_id,
                        fs.date,
                        MAX(CASE WHEN fs.metric_type IN ('Revenue', 'OperatingRevenue', 'TotalRevenue', 'NetRevenue') THEN fs.value ELSE NULL END) as revenue,
                        MAX(CASE WHEN fs.metric_type IN ('OperatingIncome', 'OperatingProfit', 'EbitIncome') THEN fs.value ELSE NULL END) as operating_income,
                        MAX(CASE WHEN fs.metric_type IN ('NetIncome', 'ProfitAfterTax', 'NetProfit', 'NetIncomeLoss') THEN fs.value ELSE NULL END) as net_income,
                        MAX(CASE WHEN fs.metric_type IN ('GrossProfit', 'GrossMargin') THEN fs.value ELSE NULL END) as gross_profit,
                        MAX(CASE WHEN fs.metric_type IN ('Research&Development', 'ResearchAndDevelopment', 'R&D') THEN fs.value ELSE NULL END) as rd_expense
                    FROM financial_statements fs
                    WHERE fs.stock_id = ?
                    GROUP BY fs.stock_id, fs.date
                    ORDER BY fs.date
                    """
                    
                    # Get financial statement data
                    financial_data = pd.read_sql_query(fs_query, conn, params=(stock_id,))
                    
                    if financial_data.empty:
                        continue
                    
                    # Balance sheet query with multiple metric types
                    bs_query = """
                    SELECT 
                        bs.stock_id,
                        bs.date,
                        MAX(CASE WHEN bs.metric_type IN ('TotalAssets', 'Assets', 'ConsolidatedTotalAssets') THEN bs.value ELSE NULL END) as total_assets,
                        MAX(CASE WHEN bs.metric_type IN ('TotalEquity', 'Equity', 'StockholdersEquity', 'TotalStockholdersEquity') THEN bs.value ELSE NULL END) as total_equity,
                        MAX(CASE WHEN bs.metric_type IN ('TotalLiabilities', 'Liabilities', 'TotalDebt') THEN bs.value ELSE NULL END) as total_liabilities,
                        MAX(CASE WHEN bs.metric_type IN ('CurrentAssets') THEN bs.value ELSE NULL END) as current_assets,
                        MAX(CASE WHEN bs.metric_type IN ('CurrentLiabilities') THEN bs.value ELSE NULL END) as current_liabilities,
                        MAX(CASE WHEN bs.metric_type IN ('Cash', 'CashAndEquivalents', 'CashAndCashEquivalents') THEN bs.value ELSE NULL END) as cash
                    FROM balance_sheets bs
                    WHERE bs.stock_id = ?
                    GROUP BY bs.stock_id, bs.date
                    ORDER BY bs.date
                    """
                    
                    # Get balance sheet data
                    balance_data = pd.read_sql_query(bs_query, conn, params=(stock_id,))
                    
                    # Cash flow query
                    cf_query = """
                    SELECT 
                        cf.stock_id,
                        cf.date,
                        MAX(CASE WHEN cf.metric_type IN ('OperatingCashFlow', 'CashFromOperations') THEN cf.value ELSE NULL END) as operating_cash_flow,
                        MAX(CASE WHEN cf.metric_type IN ('CapitalExpenditure', 'Capex', 'InvestingCashFlow') THEN cf.value ELSE NULL END) as capex,
                        MAX(CASE WHEN cf.metric_type IN ('FreeCashFlow', 'FCF') THEN cf.value ELSE NULL END) as free_cash_flow
                    FROM cash_flows cf
                    WHERE cf.stock_id = ?
                    GROUP BY cf.stock_id, cf.date
                    ORDER BY cf.date
                    """
                    
                    # Get cash flow data
                    cash_flow_data = pd.read_sql_query(cf_query, conn, params=(stock_id,))
                    
                    # Get price data for calculating returns
                    price_query = """
                    SELECT date, close
                    FROM stock_prices
                    WHERE stock_id = ?
                    ORDER BY date
                    """
                    
                    price_data = pd.read_sql_query(price_query, conn, params=(stock_id,))
                    
                    if not balance_data.empty:
                        stocks_with_data += 1
                        
                        # Merge all the data sources on date
                        combined_data = pd.merge(financial_data, balance_data, on=['stock_id', 'date'], how='inner')
                        
                        if not cash_flow_data.empty:
                            combined_data = pd.merge(combined_data, cash_flow_data, on=['stock_id', 'date'], how='left')
                        
                        # Calculate growth rates if we have more than one period
                        if len(combined_data) > 1:
                            combined_data['revenue_prev'] = combined_data['revenue'].shift(1)
                            combined_data['revenue_growth'] = (combined_data['revenue'] - combined_data['revenue_prev']) / combined_data['revenue_prev']
                            
                            # Calculate historical growth statistics
                            growth_rates = combined_data['revenue_growth'].dropna().tolist()
                            if growth_rates:
                                historical_growth_mean = np.mean(growth_rates)
                                historical_growth_std = np.std(growth_rates) if len(growth_rates) > 1 else 0.1
                            else:
                                historical_growth_mean = None
                                historical_growth_std = None
                        else:
                            historical_growth_mean = None
                            historical_growth_std = None
                        
                        # Process each reporting period
                        for _, row in combined_data.iterrows():
                            # Create a record for this period
                            record = {
                                'stock_id': stock_id,
                                'timestamp': row['date'],
                                'revenue': row['revenue'],
                                'operating_income': row['operating_income'],
                                'net_income': row['net_income'],
                                'total_assets': row['total_assets'],
                                'total_equity': row['total_equity'],
                                'gross_profit': row.get('gross_profit'),
                                'rd_expense': row.get('rd_expense'),
                                'total_liabilities': row.get('total_liabilities'),
                                'current_assets': row.get('current_assets'),
                                'current_liabilities': row.get('current_liabilities'),
                                'cash': row.get('cash'),
                                'operating_cash_flow': row.get('operating_cash_flow'),
                                'capex': row.get('capex'),
                                'free_cash_flow': row.get('free_cash_flow'),
                                'revenue_growth': row.get('revenue_growth'),
                                'historical_growth_mean': historical_growth_mean,
                                'historical_growth_std': historical_growth_std
                            }
                            
                            # Calculate financial ratios if we have the data
                            if row['revenue'] and row['revenue'] > 0:
                                if row['operating_income'] is not None:
                                    record['operating_margin'] = row['operating_income'] / row['revenue']
                                if row['net_income'] is not None:
                                    record['net_margin'] = row['net_income'] / row['revenue']
                                if row.get('gross_profit') is not None:
                                    record['gross_margin'] = row['gross_profit'] / row['revenue']
                            
                            if row['total_assets'] and row['total_assets'] > 0:
                                if row['net_income'] is not None:
                                    record['roa'] = row['net_income'] / row['total_assets']
                                if row.get('operating_income') is not None:
                                    record['operating_roa'] = row['operating_income'] / row['total_assets']
                            
                            if row['total_equity'] and row['total_equity'] > 0:
                                if row['net_income'] is not None:
                                    record['roe'] = row['net_income'] / row['total_equity']
                                if row['total_assets'] is not None:
                                    record['equity_to_assets'] = row['total_equity'] / row['total_assets']
                                    
                                    if row.get('total_liabilities') is not None and row['total_liabilities'] > 0:
                                        record['debt_to_equity'] = row['total_liabilities'] / row['total_equity']
                            
                            # Calculate liquidity ratios
                            if row.get('current_liabilities') is not None and row['current_liabilities'] > 0:
                                if row.get('current_assets') is not None:
                                    record['current_ratio'] = row['current_assets'] / row['current_liabilities']
                                if row.get('cash') is not None:
                                    record['cash_ratio'] = row['cash'] / row['current_liabilities']
                            
                            # Calculate future returns if price data available
                            if not price_data.empty:
                                try:
                                    # Convert dates to datetime
                                    report_date = pd.to_datetime(row['date'])
                                    price_data['date'] = pd.to_datetime(price_data['date'])
                                    
                                    # Get prices after report date
                                    future_prices = price_data[price_data['date'] >= report_date]
                                    
                                    if not future_prices.empty:
                                        # Get initial price (closest to report date)
                                        start_price = future_prices.iloc[0]['close']
                                        
                                        # Calculate returns for different time periods
                                        for months in [3, 6, 12]:
                                            future_date = report_date + pd.DateOffset(months=months)
                                            future_prices_period = price_data[price_data['date'] >= future_date]
                                            
                                            if not future_prices_period.empty:
                                                future_price = future_prices_period.iloc[0]['close']
                                                if start_price > 0:
                                                    return_val = (future_price - start_price) / start_price
                                                    record[f'future_{months}m_return'] = return_val
                                except Exception as e:
                                    logger.debug(f"Error calculating returns for {stock_id}: {e}")
                            
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
            
            # Generate data quality metrics
            data_quality = {
                'industry': industry,
                'stocks_processed': stocks_processed,
                'stocks_with_data': stocks_with_data,
                'total_records': len(df),
                'unique_stocks': df['stock_id'].nunique(),
                'avg_records_per_stock': len(df) / df['stock_id'].nunique() if df['stock_id'].nunique() > 0 else 0,
                'feature_coverage': {col: (df[col].count() / len(df)) * 100 for col in df.columns},
                'timestamp_range': [df['timestamp'].min(), df['timestamp'].max()] if 'timestamp' in df.columns else None
            }
            
            # Save data quality metrics
            quality_path = os.path.join(self.analysis_dir, f"{industry.lower().replace(' ', '_')}_data_quality.json")
            with open(quality_path, 'w') as f:
                json.dump(data_quality, f, default=str, indent=2)
            
            # Save training data to CSV
            csv_path = os.path.join(self.output_dir, f"{industry.lower().replace(' ', '_')}_training.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(df)} records for {industry} to {csv_path}")
            
            # Create data distribution charts
            self._create_data_distribution_charts(df, industry)
            
            return df
        except Exception as e:
            logger.error(f"Error preparing data for {industry}: {e}")
            return None
    
    def _create_data_distribution_charts(self, df, industry):
        """Create charts showing distribution of key financial metrics."""
        try:
            key_metrics = [col for col in ['operating_margin', 'net_margin', 'roa', 'roe', 'debt_to_equity', 
                                         'revenue_growth', 'historical_growth_mean']
                         if col in df.columns]
            
            if not key_metrics:
                return
                
            # Create distribution charts
            fig, axs = plt.subplots(len(key_metrics), 1, figsize=(10, 4 * len(key_metrics)))
            if len(key_metrics) == 1:
                axs = [axs]
                
            for i, metric in enumerate(key_metrics):
                # Remove extreme outliers for better visualization (keep 1-99 percentile)
                data = df[metric].dropna()
                q1, q99 = data.quantile(0.01), data.quantile(0.99)
                filtered_data = data[(data >= q1) & (data <= q99)]
                
                sns.histplot(filtered_data, kde=True, ax=axs[i])
                axs[i].set_title(f'{metric.replace("_", " ").title()} Distribution')
                axs[i].axvline(filtered_data.mean(), color='red', linestyle='--', label=f'Mean: {filtered_data.mean():.4f}')
                axs[i].axvline(filtered_data.median(), color='green', linestyle='--', label=f'Median: {filtered_data.median():.4f}')
                axs[i].legend()
            
            plt.tight_layout()
            chart_path = os.path.join(self.chart_dir, f"{industry.lower().replace(' ', '_')}_distributions.png")
            plt.savefig(chart_path)
            plt.close()
            
            # Create correlation heatmap
            plt.figure(figsize=(12, 10))
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            correlation_cols = [col for col in numeric_cols if col not in ['stock_id'] and df[col].count() > df.shape[0] * 0.5]
            
            if len(correlation_cols) > 1:
                corr_matrix = df[correlation_cols].corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                           vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
                plt.title(f'Feature Correlation for {industry}')
                
                corr_path = os.path.join(self.chart_dir, f"{industry.lower().replace(' ', '_')}_correlation.png")
                plt.savefig(corr_path)
                plt.close()
        
        except Exception as e:
            logger.warning(f"Error creating charts for {industry}: {e}")
    
    def train_industry_model(self, industry, data=None):
        """Train a model for an industry with comprehensive evaluation."""
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
            
            # Select features that exist in the data with sufficient coverage
            potential_features = [
                'revenue', 'operating_margin', 'net_margin', 'roa', 'roe',
                'historical_growth_mean', 'debt_to_equity', 'equity_to_assets',
                'gross_margin', 'current_ratio', 'revenue_growth'
            ]
            
            # Filter for features with at least 50% coverage
            features = [f for f in potential_features if f in data.columns 
                      and data[f].count() > len(data) * 0.5]
            
            if len(features) < 2:
                logger.error(f"Not enough features for {industry}")
                return False
            
            logger.info(f"Using features: {features}")
            
            # Prepare features and target
            X = data[features].copy()
            
            # Choose target column - prefer 6m return if available, otherwise use synthetic
            target_options = ['future_6m_return', 'future_3m_return', 'future_12m_return']
            target_col = next((col for col in target_options if col in data.columns 
                              and data[col].count() > len(data) * 0.3), None)
            
            if target_col:
                logger.info(f"Using {target_col} as target variable")
                y = data[target_col]
            else:
                logger.warning(f"No future return data available, using synthetic values")
                # Create synthetic target based on financial metrics
                data['future_6m_return'] = (
                    0.05 + 
                    0.2 * data.get('operating_margin', 0) - 
                    0.1 * data.get('debt_to_equity', 0) + 
                    0.15 * data.get('roe', 0) -
                    0.05 * np.random.random(len(data))
                )
                y = data['future_6m_return']
            
            # Handle missing values and clean extreme values
            for col in X.columns:
                # Replace NaN values using median (more robust than mean)
                X[col] = X[col].fillna(X[col].median() if not X[col].empty else 0)
                
                # Replace infinity with large but finite values
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                
                # Handle remaining NaNs (if any after replacing infinities)
                X[col] = X[col].fillna(X[col].median() if not X[col].empty else 0)
                
                # Clip extreme values to reasonable range
                if X[col].dtype.kind in 'fc':  # If column is float or complex
                    q1, q99 = X[col].quantile(0.01), X[col].quantile(0.99)
                    iqr = q99 - q1
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q99 + 3 * iqr
                    X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Clean target variable
            y = y.fillna(y.median() if not y.empty else 0.05)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Use robust scaler to handle outliers
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Define model architecture with regularization
            model = self._create_model(X.shape[1])
            
            # Model metrics to monitor
            metrics = [
                tf.keras.metrics.MeanSquaredError(name='MSE'),
                tf.keras.metrics.MeanAbsoluteError(name='MAE')
            ]
            
            # Compile model with optimizer and loss
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
            model.compile(optimizer=optimizer, loss='mse', metrics=metrics)
            
            # Early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                min_delta=self.config['min_delta'],
                restore_best_weights=True
            )
            
            # Model checkpoint - FIX: Use proper file extension for save_weights_only=True
            checkpoint_path = os.path.join(
                self.model_dir, 
                f"{industry.lower().replace(' ', '_')}_checkpoint.weights.h5"
            )
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                save_weights_only=True
            )
            
            # Train mode, either with cross-validation or simple validation split
            if self.config['use_cross_validation'] and len(X) >= 50:
                logger.info(f"Using {self.config['cv_folds']}-fold cross-validation")
                
                # Store CV results
                cv_results = {
                    'train_loss': [],
                    'val_loss': [],
                    'test_loss': [],
                    'train_mae': [],
                    'val_mae': [],
                    'test_mae': []
                }
                
                # Initialize KFold
                kfold = KFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=42)
                
                # Perform cross-validation
                fold = 1
                for train_idx, val_idx in kfold.split(X_train_scaled):
                    logger.info(f"Training fold {fold}/{self.config['cv_folds']}")
                    
                    # Split data for this fold
                    X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
                    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Clone model for this fold
                    fold_model = self._create_model(X.shape[1])
                    
                    # Create a fresh optimizer for each fold to avoid the variable error
                    fold_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
                    fold_model.compile(optimizer=fold_optimizer, loss='mse', metrics=metrics)
                    
                    # Train fold
                    fold_history = fold_model.fit(
                        X_train_fold, y_train_fold,
                        epochs=self.config['epochs'],
                        batch_size=min(self.config['batch_size'], len(X_train_fold)),
                        validation_data=(X_val_fold, y_val_fold),
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    
                    # Evaluate fold
                    train_metrics = fold_model.evaluate(X_train_fold, y_train_fold, verbose=0)
                    val_metrics = fold_model.evaluate(X_val_fold, y_val_fold, verbose=0)
                    test_metrics = fold_model.evaluate(X_test_scaled, y_test, verbose=0)
                    
                    # Record metrics
                    cv_results['train_loss'].append(train_metrics[0])
                    cv_results['val_loss'].append(val_metrics[0])
                    cv_results['test_loss'].append(test_metrics[0])
                    
                    cv_results['train_mae'].append(train_metrics[1])
                    cv_results['val_mae'].append(val_metrics[1])
                    cv_results['test_mae'].append(test_metrics[1])
                    
                    fold += 1
                
                # Train final model on all training data
                # Create a fresh optimizer for the final model
                final_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
                model.compile(optimizer=final_optimizer, loss='mse', metrics=metrics)
                
                history = model.fit(
                    X_train_scaled, y_train,
                    epochs=self.config['epochs'],
                    batch_size=min(self.config['batch_size'], len(X_train)),
                    validation_split=0.2,
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=1
                )
                
                # Save CV results
                cv_summary = {
                    'train_loss_mean': np.mean(cv_results['train_loss']),
                    'train_loss_std': np.std(cv_results['train_loss']),
                    'val_loss_mean': np.mean(cv_results['val_loss']),
                    'val_loss_std': np.std(cv_results['val_loss']),
                    'test_loss_mean': np.mean(cv_results['test_loss']),
                    'test_loss_std': np.std(cv_results['test_loss']),
                    
                    'train_mae_mean': np.mean(cv_results['train_mae']),
                    'train_mae_std': np.std(cv_results['train_mae']),
                    'val_mae_mean': np.mean(cv_results['val_mae']),
                    'val_mae_std': np.std(cv_results['val_mae']),
                    'test_mae_mean': np.mean(cv_results['test_mae']),
                    'test_mae_std': np.std(cv_results['test_mae']),
                }
                
                cv_path = os.path.join(self.analysis_dir, f"{industry.lower().replace(' ', '_')}_cv_results.json")
                with open(cv_path, 'w') as f:
                    json.dump(cv_summary, f, indent=2)
                
                logger.info(f"Cross-validation results: Test MAE = {cv_summary['test_mae_mean']:.4f} ± {cv_summary['test_mae_std']:.4f}")
                
            else:
                # Train without CV for smaller datasets
                history = model.fit(
                    X_train_scaled, y_train,
                    epochs=self.config['epochs'],
                    batch_size=min(self.config['batch_size'], len(X_train)),
                    validation_split=self.config['validation_split'],
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=1
                )
            
            # Evaluate final model
            test_metrics = model.evaluate(X_test_scaled, y_test, verbose=0)
            train_metrics = model.evaluate(X_train_scaled, y_train, verbose=0)
            
            # Generate predictions for analysis
            y_pred = model.predict(X_test_scaled)
            
            # Calculate additional metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Save detailed evaluation metrics
            evaluation = {
                'industry': industry,
                'features_used': features,
                'n_samples': len(X),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_loss': float(train_metrics[0]),
                'test_loss': float(test_metrics[0]),
                'train_mae': float(train_metrics[1]),
                'test_mae': float(test_metrics[1]),
                'rmse': float(rmse),
                'r2': float(r2),
                'config': self.config,
                'feature_importance': self._calculate_feature_importance(model, X, features)
            }
            
            eval_path = os.path.join(self.analysis_dir, f"{industry.lower().replace(' ', '_')}_evaluation.json")
            with open(eval_path, 'w') as f:
                json.dump(evaluation, f, indent=2)
            
            # Plot training history
            self._plot_training_history(history, industry)
            
            # Plot predictions vs actual
            self._plot_predictions(y_test, y_pred, industry)
            
            # Save model and artifacts
            self._save_model_artifacts(model, scaler, features, industry, evaluation)
            
            logger.info(f"Model training completed for {industry}. Test MAE: {mae:.4f}, R²: {r2:.4f}")
            
            return True
        except Exception as e:
            logger.error(f"Error training model for {industry}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _create_model(self, input_size):
        """Create a neural network model based on configuration."""
        # L2 regularization
        regularizer = tf.keras.regularizers.l2(self.config['l2_regularization'])
        
        # Create model with configurable architecture
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=(input_size,)))
        
        # Hidden layers from config
        for units in self.config['hidden_layers']:
            model.add(tf.keras.layers.Dense(
                units=units,
                activation='relu',
                kernel_regularizer=regularizer
            ))
            model.add(tf.keras.layers.Dropout(self.config['dropout_rate']))
        
        # Output layer (single value for regression)
        model.add(tf.keras.layers.Dense(1))
        
        return model
    
    def _calculate_feature_importance(self, model, X, features):
        """Calculate feature importance using permutation importance method."""
        try:
            # Get baseline performance
            baseline = model.evaluate(X, y=None, verbose=0)[0]
            
            # Calculate importance for each feature
            importance = {}
            
            for i, feature in enumerate(features):
                # Make a copy of X
                X_permuted = X.copy()
                
                # Shuffle the feature to break its relationship with target
                X_permuted[feature] = np.random.permutation(X_permuted[feature])
                
                # Evaluate with the permuted feature
                permuted_score = model.evaluate(X_permuted, y=None, verbose=0)[0]
                
                # Calculate importance (increase in error)
                importance[feature] = float(permuted_score - baseline)
            
            # Normalize importance values
            max_importance = max(importance.values()) if importance else 1
            normalized_importance = {k: v/max_importance for k, v in importance.items()}
            
            return normalized_importance
        except Exception as e:
            logger.warning(f"Error calculating feature importance: {e}")
            return {}
    
    def _plot_training_history(self, history, industry):
        """Plot and save the model training history."""
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Loss Curves for {industry}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot MAE if available
        plt.subplot(1, 2, 2)
        if 'mae' in history.history:
            plt.plot(history.history['mae'], label='Training MAE')
        if 'val_mae' in history.history:
            plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title(f'MAE Curves for {industry}')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = os.path.join(self.chart_dir, f"{industry.lower().replace(' ', '_')}_training_history.png")
        plt.savefig(chart_path)
        plt.close()
    
    def _plot_predictions(self, y_true, y_pred, industry):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Calculate R² for plot
        r2 = r2_score(y_true, y_pred)
        
        plt.title(f'Actual vs Predicted Returns for {industry} (R² = {r2:.4f})')
        plt.xlabel('Actual Returns')
        plt.ylabel('Predicted Returns')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        chart_path = os.path.join(self.chart_dir, f"{industry.lower().replace(' ', '_')}_predictions.png")
        plt.savefig(chart_path)
        plt.close()
    
    def _save_model_artifacts(self, model, scaler, features, industry, evaluation):
        """Save model and related artifacts."""
        # Create standardized filename
        base_name = industry.lower().replace(' ', '_')
        
        # Save model with .keras extension
        model_path = os.path.join(self.model_dir, f"{base_name}_model.keras")
        scaler_path = os.path.join(self.model_dir, f"{base_name}_scaler.pkl")
        features_path = os.path.join(self.model_dir, f"{base_name}_features.json")
        summary_path = os.path.join(self.model_dir, f"{base_name}_summary.json")
        
        # Save model
        model.save(model_path)
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature list
        with open(features_path, 'w') as f:
            json.dump(features, f)
        
        # Create and save model card with key information
        model_card = {
            'industry': industry,
            'created_date': datetime.now().isoformat(),
            'model_type': 'Neural Network Regression',
            'features': features,
            'performance': {
                'mae': evaluation['test_mae'],
                'rmse': evaluation['rmse'],
                'r2': evaluation['r2']
            },
            'feature_importance': evaluation['feature_importance'],
            'config': self.config
        }
        
        with open(summary_path, 'w') as f:
            json.dump(model_card, f, indent=2)
            
        logger.info(f"Saved model and artifacts for {industry} to {model_path}")
    
    def create_industry_benchmarks(self):
        """Create comprehensive industry benchmarks from training datasets."""
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
                    
                    # Basic metrics
                    metrics = {
                        'industry': industry,
                        'stock_count': df['stock_id'].nunique() if 'stock_id' in df.columns else 0,
                        'record_count': len(df)
                    }
                    
                    # Calculate metrics for each key ratio with detailed percentiles
                    for metric in ['operating_margin', 'net_margin', 'roa', 'roe', 'debt_to_equity',
                                  'gross_margin', 'revenue_growth', 'historical_growth_mean']:
                        if metric in df.columns and df[metric].count() > 0:
                            # Remove extreme outliers for reliable statistics
                            filtered = df[metric].dropna()
                            if len(filtered) > 0:
                                q1, q3 = filtered.quantile(0.25), filtered.quantile(0.75)
                                iqr = q3 - q1
                                lower_bound, upper_bound = q1 - 3*iqr, q3 + 3*iqr
                                filtered = filtered[(filtered >= lower_bound) & (filtered <= upper_bound)]
                                
                                # Calculate statistics
                                metrics[f'{metric}_median'] = filtered.median()
                                metrics[f'{metric}_mean'] = filtered.mean()
                                metrics[f'{metric}_std'] = filtered.std()
                                metrics[f'{metric}_25th'] = filtered.quantile(0.25)
                                metrics[f'{metric}_75th'] = filtered.quantile(0.75)
                    
                    industry_metrics.append(metrics)
                    logger.info(f"Processed benchmarks for {industry}: {len(df)} records")
                    
                except Exception as e:
                    logger.error(f"Error processing {industry}: {e}")
            
            # Create DataFrame from metrics
            if not industry_metrics:
                logger.error("No valid benchmarks created")
                return False
                
            benchmarks_df = pd.DataFrame(industry_metrics)
            
            # Save benchmarks to multiple formats
            benchmark_file_csv = os.path.join(self.output_dir, 'industry_benchmarks.csv')
            benchmark_file_json = os.path.join(self.output_dir, 'industry_benchmarks.json')
            
            # Save as CSV
            benchmarks_df.to_csv(benchmark_file_csv, index=False)
            
            # Save as JSON (more portable format)
            benchmarks_df.to_json(benchmark_file_json, orient='records', indent=2)
            
            # Also save to original industry_data directory for compatibility
            orig_dir = "industry_data"
            os.makedirs(orig_dir, exist_ok=True)
            benchmarks_df.to_csv(os.path.join(orig_dir, 'industry_benchmarks.csv'), index=False)
            
            # Generate visualization of key metrics across industries
            self._create_benchmark_visualizations(benchmarks_df)
            
            logger.info(f"Saved industry benchmarks for {len(benchmarks_df)} industries")
            
            return True
        except Exception as e:
            logger.error(f"Error creating benchmarks: {e}")
            return False
    
    def _create_benchmark_visualizations(self, benchmarks_df):
        """Create visualizations comparing key metrics across industries."""
        try:
            # Filter to only include industries with sufficient data
            filtered_df = benchmarks_df[benchmarks_df['record_count'] >= 10]
            
            if filtered_df.empty:
                return
            
            # Key metrics to visualize
            metrics = ['operating_margin_median', 'net_margin_median', 'roa_median', 'roe_median']
            available_metrics = [m for m in metrics if m in filtered_df.columns]
            
            if not available_metrics:
                return
                
            # Sort industries by record count for more legible charts
            filtered_df = filtered_df.sort_values('record_count', ascending=False)
            
            # Limit to top 15 industries for readability
            if len(filtered_df) > 15:
                filtered_df = filtered_df.head(15)
            
            # Create bar charts for each metric
            for metric in available_metrics:
                base_metric = metric.replace('_median', '')
                
                plt.figure(figsize=(12, 8))
                
                # Create bar chart
                bars = plt.barh(filtered_df['industry'], filtered_df[metric])
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.002, 
                             bar.get_y() + bar.get_height()/2, 
                             f'{width:.2%}' if width < 1 else f'{width:.2f}',
                             va='center')
                
                plt.title(f'{base_metric.replace("_", " ").title()} by Industry')
                plt.xlabel(base_metric.replace('_', ' ').title())
                plt.tight_layout()
                
                # Save chart
                chart_path = os.path.join(self.chart_dir, f"benchmark_{base_metric}.png")
                plt.savefig(chart_path)
                plt.close()
            
            # Create a summary heatmap for key metrics
            if len(available_metrics) > 1:
                plt.figure(figsize=(14, 10))
                
                # Prepare data for heatmap
                heatmap_data = filtered_df.set_index('industry')[available_metrics]
                
                # Normalize data for better visualization
                normalized_data = heatmap_data.copy()
                for col in normalized_data.columns:
                    normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / (normalized_data[col].max() - normalized_data[col].min())
                
                # Create heatmap
                sns.heatmap(normalized_data, annot=heatmap_data, fmt=".2%", cmap="YlGnBu", linewidths=.5)
                plt.title('Industry Financial Metrics Comparison')
                plt.tight_layout()
                
                # Save chart
                chart_path = os.path.join(self.chart_dir, f"benchmark_heatmap.png")
                plt.savefig(chart_path)
                plt.close()
                
        except Exception as e:
            logger.warning(f"Error creating benchmark visualizations: {e}")
    
    def train_all_industries(self, min_stocks=5, min_records=20):
        """Train models for all industries with sufficient data."""
        start_time = datetime.now()
        logger.info(f"Starting comprehensive model training at {start_time}")
        
        # Get all industries with stock counts
        industries = self.get_industries()
        
        if not industries:
            logger.error("No industries found in database")
            return []
        
        # Filter industries with sufficient data
        industries_to_process = [ind['name'] for ind in industries if ind['stock_count'] >= min_stocks]
        
        logger.info(f"Found {len(industries_to_process)} industries with at least {min_stocks} stocks")
        
        # Prepare data and train models
        successful = []
        processed = 0
        
        for industry in industries_to_process:
            processed += 1
            logger.info(f"Processing industry {processed}/{len(industries_to_process)}: {industry}")
            
            # Prepare data
            data = self.prepare_direct_financial_metrics(industry)
            
            if data is not None and len(data) >= min_records:
                logger.info(f"Training model for {industry} with {len(data)} records")
                
                # Train model
                if self.train_industry_model(industry, data):
                    successful.append(industry)
                    logger.info(f"Successfully trained model for {industry}")
                else:
                    logger.warning(f"Failed to train model for {industry}")
            else:
                logger.warning(f"Insufficient data for {industry}, skipping")
        
        # Create industry benchmarks
        self.create_industry_benchmarks()
        
        # Generate summary report
        end_time = datetime.now()
        duration = end_time - start_time
        
        summary_report = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'industries_processed': processed,
            'models_trained': len(successful),
            'successful_industries': successful
        }
        
        summary_path = os.path.join(self.analysis_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        logger.info(f"Training completed in {duration}. Successful: {len(successful)}/{len(industries_to_process)}")
        logger.info(f"Successful industries: {successful}")
        
        return successful

def main():
    """Main function to run training."""
    logger.info("Starting comprehensive model training")
    
    # Create trainer with default settings
    trainer = DBModelTrainer()
    
    # Train models for all industries
    trainer.train_all_industries()
    
    logger.info("Training process completed")

if __name__ == "__main__":
    main()
