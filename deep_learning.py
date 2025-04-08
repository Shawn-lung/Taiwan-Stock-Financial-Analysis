import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Attention, concatenate # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from typing import List, Dict, Tuple, Union, Optional
import matplotlib.pyplot as plt
import datetime as dt  # Import as dt to avoid name collision

logger = logging.getLogger(__name__)

class DeepFinancialForecaster:
    """Deep learning-based financial forecasting models."""
    
    def __init__(self, sequence_length: int = 3):
        """Initialize the deep forecaster.
        
        Args:
            sequence_length: Number of time steps to use for sequence models
        """
        self.sequence_length = sequence_length
        self.min_sequence_length = 1  # Allow models to work with smaller datasets
        self.lstm_model = None
        self.attention_model = None
        self.hybrid_model = None
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        # Add counters to track filtered data points
        self.filtered_data_counts = {
            'missing_revenue': 0,
            'negative_revenue': 0,
            'missing_op_income': 0,
            'negative_op_income': 0,
            'valid_points': 0,
            'total_points': 0
        }
        
    def build_lstm_model(self, input_dim: int) -> Sequential:
        """Build an improved LSTM model with regularization to prevent overfitting."""
        model = Sequential([
            LSTM(64, activation='tanh', recurrent_activation='sigmoid',
                 return_sequences=True, input_shape=(self.sequence_length, input_dim),
                 recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.3),
            LSTM(32, activation='tanh', recurrent_activation='sigmoid',
                 recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.3),
            Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dense(1)
        ])
        
        # Use a lower learning rate to prevent memorization
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
        return model

    def build_attention_model(self, input_dim: int) -> Model:
        """Build an improved attention-based model with better regularization."""
        # Input layers
        sequence_input = Input(shape=(self.sequence_length, input_dim))
        
        # LSTM layers with regularization
        lstm_out = LSTM(64, return_sequences=True, 
                       recurrent_regularizer=tf.keras.regularizers.l2(0.01))(sequence_input)
        lstm_out = Dropout(0.3)(lstm_out)
        
        # Self-attention mechanism
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=32
        )(lstm_out, lstm_out)
        
        # Add layer normalization for more stable training
        attention_output = tf.keras.layers.LayerNormalization()(attention_output)
        
        # Combine attention with LSTM output
        x = concatenate([lstm_out, attention_output])
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = Dropout(0.3)(x)
        output = Dense(1)(x)
        
        # Create model with lower learning rate
        model = Model(inputs=sequence_input, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
        
        return model

    def _prepare_sequence_data(self, data: pd.DataFrame, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for LSTM training."""
        X_seq = []
        y_seq = []
        
        # Create sequences from the data
        for i in range(len(data) - self.sequence_length):
            X_seq.append(data.iloc[i:i+self.sequence_length].values)
            y_seq.append(target[i+self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
        
    def _prepare_industry_data(self, industry_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Process industry data with multiple companies to create robust training sequences.
        
        Args:
            industry_df: DataFrame containing financial data for multiple companies.
                         Expected to have 'stock_id' and time-based financial metrics.
        
        Returns:
            Tuple of (X_sequences, y_values) for model training
        """
        try:
            logger.info(f"Processing industry dataset with {len(industry_df)} records across multiple companies")
            
            # First, sort by stock_id and timestamp for proper time series organization
            if 'timestamp' in industry_df.columns:
                industry_df = industry_df.sort_values(['stock_id', 'timestamp'])
            
            # Create sequences by company
            all_X_sequences = []
            all_y_values = []
            
            # Track counts for logging
            companies_processed = 0
            total_sequences = 0
            
            # Process each company separately
            for stock_id, company_data in industry_df.groupby('stock_id'):
                # Skip if too few records for this company
                if len(company_data) < 2:  # Minimum: 1 for X and 1 for y
                    logger.debug(f"Skipping stock {stock_id} - insufficient data points ({len(company_data)})")
                    continue
                
                try:
                    # Extract features needed for prediction
                    features = pd.DataFrame({
                        'revenue_log': np.log(company_data['revenue'].clip(lower=1)),
                        'op_margin': company_data['operating_income'] / company_data['revenue'].clip(lower=1),
                        'revenue_growth': company_data['revenue_growth'] if 'revenue_growth' in company_data.columns 
                                         else company_data['revenue'].pct_change().fillna(0),
                        'margin_trend': None,  # Will fill below
                        'equity_to_assets': company_data['equity_to_assets'] if 'equity_to_assets' in company_data.columns else None
                    })
                    
                    # Calculate additional derived features
                    features['margin_trend'] = features['op_margin'].rolling(min(3, len(features))).mean().fillna(features['op_margin'])
                    features['revenue_growth_trend'] = features['revenue_growth'].rolling(min(3, len(features))).mean().fillna(0)
                    
                    # Drop any columns that are all NaN
                    features = features.dropna(axis=1, how='all')
                    
                    # Fill remaining NaNs with appropriate values
                    features = features.fillna(0)
                    
                    # Calculate target: next year's growth rate (shift revenue growth)
                    target = np.array(features['revenue_growth'].shift(-1).fillna(0))

                    # Determine effective sequence length for this company
                    effective_seq_length = min(self.sequence_length, len(features) - 1)
                    if effective_seq_length < 1:
                        logger.debug(f"Skipping stock {stock_id} - insufficient data for sequences")
                        continue
                        
                    # Create sequences for this company
                    X_seq = []
                    y_seq = []
                    
                    # Get most relevant feature columns
                    feature_cols = [col for col in features.columns if col not in ['revenue_growth']]
                    
                    # Create sequences
                    for i in range(len(features) - effective_seq_length):
                        X_seq.append(features[feature_cols].iloc[i:i+effective_seq_length].values)
                        y_seq.append(target[i+effective_seq_length])
                    
                    if len(X_seq) > 0:
                        all_X_sequences.extend(X_seq)
                        all_y_values.extend(y_seq)
                        total_sequences += len(X_seq)
                        companies_processed += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing company {stock_id}: {e}")
                    continue
            
            logger.info(f"Successfully processed {companies_processed} companies, creating {total_sequences} sequences")
            
            # Convert lists to numpy arrays
            if len(all_X_sequences) == 0:
                logger.warning("No valid sequences could be created from the industry data")
                return None, None
            
            # Standardize or normalize features across all companies
            X_array = np.array(all_X_sequences)
            y_array = np.array(all_y_values)
            
            # Reshape for scaling: (n_sequences, seq_length, n_features) -> (n_sequences * seq_length, n_features)
            n_sequences, seq_length, n_features = X_array.shape
            X_reshaped = X_array.reshape(-1, n_features)
            
            # Scale features
            X_scaled = self.scaler_x.fit_transform(X_reshaped)
            
            # Reshape back to 3D
            X_scaled = X_scaled.reshape(n_sequences, seq_length, n_features)
            
            # Scale targets
            y_scaled = self.scaler_y.fit_transform(y_array.reshape(-1, 1)).flatten()
            
            logger.info(f"Final industry dataset shape: X={X_scaled.shape}, y={y_scaled.shape}")
            
            return X_scaled, y_scaled
            
        except Exception as e:
            logger.error(f"Error in _prepare_industry_data: {e}")
            return None, None

    def prepare_financial_sequences(self, financials: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and prepare sequences from financial data."""        
        # Check if input is empty
        if financials is None or financials.empty:
            logger.warning("Empty financial data provided")
            return None, None
            
        try:
            # Handle industry dataset format (CSV-derived data)
            # Check for industry dataset format with columns like 'revenue', 'operating_income', etc.
            if isinstance(financials, pd.DataFrame) and 'revenue' in financials.columns:
                logger.info(f"Processing industry dataset format with {len(financials)} data points")
                
                # NEW: Check if we have multiple companies in the dataset (industry data)
                has_stock_id = 'stock_id' in financials.columns
                multiple_companies = has_stock_id and len(financials['stock_id'].unique()) > 1
                
                if multiple_companies:
                    logger.info(f"Found industry data with {len(financials['stock_id'].unique())} companies")
                    # Process differently for industry data with multiple companies
                    return self._prepare_industry_data(financials)
                
                # Already in correct format, just need to extract features
                features = []
                
                # Extract records from DataFrame
                for idx, row in financials.iterrows():
                    try:
                        self.filtered_data_counts['total_points'] += 1
                        
                        # Extract core metrics that we need
                        revenue = float(row['revenue']) if pd.notna(row['revenue']) else None
                        op_income = float(row['operating_income']) if pd.notna(row['operating_income']) else None
                        
                        # For net_income, try various possible column names
                        net_income = None
                        for col_name in ['net_income', 'income']:
                            if col_name in financials.columns and pd.notna(row[col_name]):
                                net_income = float(row[col_name])
                                break
                        
                        # Modified filtering logic to accept more data points
                        if revenue is None:
                            self.filtered_data_counts['missing_revenue'] += 1
                            continue
                            
                        if revenue <= 0:
                            self.filtered_data_counts['negative_revenue'] += 1
                            continue
                            
                        # Instead of skipping points with missing op_income, estimate it
                        if op_income is None:
                            self.filtered_data_counts['missing_op_income'] += 1
                            # Estimate operating income based on industry averages (5-15% of revenue)
                            op_income = revenue * 0.08  # Using 8% as a reasonable default
                        
                        # Track negative operating income but don't filter it out
                        if op_income < 0:
                            self.filtered_data_counts['negative_op_income'] += 1
                            # We keep negative operating income points - they're valid data!
                        
                        features.append({
                            'revenue': revenue,
                            'op_income': op_income,
                            'net_income': net_income if net_income is not None else op_income * 0.8,
                            'op_margin': op_income / revenue
                        })
                        self.filtered_data_counts['valid_points'] += 1
                    except Exception as e:
                        logger.debug(f"Error processing row {idx}: {e}")
                
                logger.info(f"Successfully extracted {len(features)} data points from industry dataset format")
                logger.info(f"Filtered data counts: {self.filtered_data_counts}")
            else:
                # Extract relevant financial metrics from yfinance-style input
                features = []
                dates = sorted(financials.columns) if hasattr(financials, 'columns') else []
                logger.info(f"Preparing financial data with {len(dates)} time periods")
                
                # Fix orientation of financials DataFrame if needed
                if hasattr(financials, 'index') and len(financials.index) > 0 and isinstance(financials.index[0], (str, pd.Timestamp, dt.datetime)) and 'Total Revenue' not in financials.index:
                    logger.info("Transposing financial data to ensure metrics are in rows")
                    financials = financials.T
                
                for date in dates:
                    try:
                        self.filtered_data_counts['total_points'] += 1
                        
                        # More flexible revenue extraction
                        revenue = None
                        for rev_key in ['Total Revenue', 'Revenue', 'TotalRevenue', 'OperatingRevenue']:
                            if rev_key in financials.index:
                                revenue = financials.loc[rev_key, date]
                                if pd.notna(revenue) and revenue > 0:
                                    revenue = float(revenue)
                                    break
                        
                        # More flexible operating income extraction
                        op_income = None
                        for op_key in ['Operating Income', 'OperatingIncome', 'OperatingProfit', 'EBIT']:
                            if op_key in financials.index:
                                op_income = financials.loc[op_key, date]
                                if pd.notna(op_income):
                                    op_income = float(op_income)
                                    break
                        
                        # More flexible net income extraction
                        net_income = None
                        for net_key in ['Net Income', 'NetIncome', 'ProfitAfterTax', 'NetEarnings']:
                            if net_key in financials.index:
                                net_income = financials.loc[net_key, date]
                                if pd.notna(net_income):
                                    net_income = float(net_income)
                                    break
                        
                        if revenue and op_income and net_income:
                            features.append({
                                'revenue': revenue,
                                'op_income': op_income,
                                'net_income': net_income,
                                'op_margin': op_income / revenue
                            })
                    except Exception as e:
                        logger.debug(f"Error processing financial data for {date}: {e}")
            
            # Adjust sequence length for small datasets - this is the key change
            original_seq_length = self.sequence_length
            
            # ENHANCED DATA HANDLING FOR VERY SMALL DATASETS
            if len(features) < 2:
                # With just 1 data point, generate synthetic data to enable training
                if len(features) == 1:
                    logger.info("Only 1 data point available - generating synthetic data to enable training")
                    base_data = features[0]
                    
                    # Create synthetic historical data from the single point
                    for i in range(3):  # Generate 3 synthetic past points
                        # Create variations of the data with small adjustments
                        variation_factor = 0.9 + (i * 0.05)  # 0.9, 0.95, 1.0
                        synthetic_point = {
                            'revenue': base_data['revenue'] * variation_factor,
                            'op_income': base_data['op_income'] * variation_factor,
                            'net_income': base_data['net_income'] * variation_factor if base_data['net_income'] else base_data['op_income'] * 0.8 * variation_factor,
                            'op_margin': base_data['op_margin']  # Keep margin similar
                        }
                        features.insert(0, synthetic_point)  # Insert as older data
                    
                    logger.info(f"Expanded dataset from 1 to {len(features)} points using synthetic data")
                    
                    # Ensure sequence length is set correctly for the expanded data
                    # We need to set this to 1 since we'll have 3 sequences of length 1
                    self.sequence_length = 1
                else:
                    logger.warning(f"Insufficient financial data: {len(features)} points available, minimum 1 needed")
                    # Restore original sequence length
                    self.sequence_length = original_seq_length
                    return None, None
            elif len(features) < self.sequence_length + 1:
                # Try with smaller sequence length if we have at least 2 data points
                logger.warning(f"Limited data available ({len(features)} points). Adapting sequence length from {self.sequence_length} to 1.")
                self.sequence_length = 1
            
            logger.info(f"Processing with sequence length {self.sequence_length} using {len(features)} financial data points")
            
            # Convert to DataFrame and calculate growth rates
            df = pd.DataFrame(features)
            df['revenue_growth'] = df['revenue'].pct_change().fillna(0)
            df['op_income_growth'] = df['op_income'].pct_change().fillna(0)
            df['net_income_growth'] = df['net_income'].pct_change().fillna(0)
            
            # Add more meaningful features
            df['revenue_log'] = np.log(df['revenue'].clip(lower=1))
            df['revenue_growth_trend'] = df['revenue_growth'].rolling(min(3, len(df))).mean().fillna(0)
            df['margin_trend'] = df['op_margin'].rolling(min(3, len(df))).mean().fillna(df['op_margin'])
            
            # Log summary statistics
            logger.info(f"Historical revenue growth stats: mean={df['revenue_growth'].mean():.2%}, std={df['revenue_growth'].std():.2%}")
            
            # Normalize features
            feature_cols = ['revenue_log', 'op_margin', 'revenue_growth', 'revenue_growth_trend', 'margin_trend']
            X = df[feature_cols].copy()
            X_scaled = pd.DataFrame(self.scaler_x.fit_transform(X), columns=feature_cols)
            
            # Target: next year's growth rate
            y = np.array(df['revenue_growth'].shift(-1).fillna(0))
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            # Prepare sequences
            sequences = self._prepare_sequence_data(X_scaled, y_scaled)
            
            # Restore original sequence length
            self.sequence_length = original_seq_length
            
            return sequences
        except Exception as e:
            logger.error(f"Error preparing financial sequences: {e}")
            return None, None
        
    def train_growth_forecaster(self, financials: pd.DataFrame, validation_split: float = 0.2) -> bool:
        """Train deep learning models on historical financial data."""
        try:
            # Convert to DataFrame if financials is a dictionary
            if isinstance(financials, dict):
                # If it's a simple dict with financial metrics, convert to a single-row DataFrame
                if 'revenue' in financials or 'historical_revenue' in financials:
                    financials_df = pd.DataFrame([financials])
                else:
                    # It might be a dict with keys as dates and values as metrics
                    # Try to reconstruct a DataFrame that makes sense for our analysis
                    logger.info("Converting dictionary to DataFrame for deep learning training")
                    try:
                        if 'income_statement' in financials:
                            financials_df = financials['income_statement']
                        else:
                            # Fall back to creating a DataFrame from the dict
                            financials_df = pd.DataFrame.from_dict(financials)
                    except Exception as e:
                        logger.error(f"Failed to convert dict to DataFrame: {e}")
                        return False
            else:
                financials_df = financials
            
            # Prepare sequence data
            X_seq, y_seq = self.prepare_financial_sequences(financials_df)
            if X_seq is None or len(X_seq) < 1:  # Just need at least one sample
                logger.warning("Insufficient data for deep learning training")
                return False
            
            # Get input dimensions
            _, _, input_dim = X_seq.shape
            logger.info(f"Training deep models with {len(X_seq)} sequences of {self.sequence_length} timesteps and {input_dim} features")
            
            # Build models
            self.lstm_model = self.build_lstm_model(input_dim)
            self.attention_model = self.build_attention_model(input_dim)
            
            # Set up early stopping with more patience for small datasets
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,  # Increased patience for small datasets
                restore_best_weights=True,
            )
            
            # For very small datasets, skip validation to use all data for training
            use_validation = len(X_seq) >= 3
            val_split = min(validation_split, 0.2) if use_validation else 0.0
            
            # Train LSTM model
            logger.info("Training LSTM model...")
            lstm_history = self.lstm_model.fit(
                X_seq, y_seq,
                epochs=150,  # Increased epochs for small datasets
                batch_size=max(1, min(4, len(X_seq))),  # Smaller batch size for small datasets
                validation_split=val_split,
                callbacks=[early_stopping] if use_validation else None,
                verbose=0
            )
            
            # Train attention model
            logger.info("Training attention model...")
            attention_history = self.attention_model.fit(
                X_seq, y_seq,
                epochs=150,  # Increased epochs for small datasets
                batch_size=max(1, min(4, len(X_seq))),
                validation_split=val_split,
                callbacks=[early_stopping] if use_validation else None,
                verbose=0
            )
            
            # Evaluate models
            lstm_loss = self.lstm_model.evaluate(X_seq, y_seq, verbose=0)
            attention_loss = self.attention_model.evaluate(X_seq, y_seq, verbose=0)
            logger.info(f"LSTM Model Loss: {lstm_loss:.4f}")
            logger.info(f"Attention Model Loss: {attention_loss:.4f}")
            
            # Validate if models are learning properly by checking predictions
            lstm_preds = self.lstm_model.predict(X_seq, verbose=0).flatten()
            attn_preds = self.attention_model.predict(X_seq, verbose=0).flatten()
            
            # Check variance in predictions to ensure model is not outputting constants
            lstm_var = np.var(lstm_preds)
            attn_var = np.var(attn_preds)
            logger.info(f"LSTM prediction variance: {lstm_var:.6f}, Attention prediction variance: {attn_var:.6f}")
            
            # For small datasets, we're less strict about variance requirements
            min_variance_threshold = 1e-6 if len(X_seq) >= 3 else 1e-8
            
            if lstm_var < min_variance_threshold and attn_var < min_variance_threshold:
                logger.warning("Models are producing near-constant outputs - they may not be learning properly")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error training deep learning models: {e}")
            return False

    def predict_future_growth(self, financials: pd.DataFrame, forecast_years: int = 5) -> List[float]:
        """Predict future growth rates using ensemble of deep learning models."""
        try:
            # Reset filtered data counters for this prediction
            self.filtered_data_counts = {
                'missing_revenue': 0,
                'negative_revenue': 0,
                'missing_op_income': 0,
                'negative_op_income': 0,
                'valid_points': 0,
                'total_points': 0
            }
            
            # First train the models
            success = self.train_growth_forecaster(financials)
            
            # Log data filtering information
            if self.filtered_data_counts['total_points'] > 0:
                logger.info(f"Data filtering summary:")
                logger.info(f"  - Total data points: {self.filtered_data_counts['total_points']}")
                logger.info(f"  - Valid data points: {self.filtered_data_counts['valid_points']}")
                logger.info(f"  - Missing revenue: {self.filtered_data_counts['missing_revenue']}")
                logger.info(f"  - Negative/zero revenue: {self.filtered_data_counts['negative_revenue']}")
                logger.info(f"  - Missing operating income (estimated): {self.filtered_data_counts['missing_op_income']}")
                logger.info(f"  - Negative operating income: {self.filtered_data_counts['negative_op_income']}")
            
            # Check if models trained successfully
            if not success or self.lstm_model is None or self.attention_model is None:
                logger.warning("Deep learning model training failed, falling back to industry growth patterns")
                industry_type = self._detect_industry_from_financials(financials)
                return self._generate_industry_baseline_growth(industry_type, forecast_years)
                
            # Prepare sequence data for prediction
            X_seq, _ = self.prepare_financial_sequences(financials)
            if X_seq is None or len(X_seq) == 0:
                logger.warning("Could not prepare sequence data for prediction")
                return self._generate_industry_baseline_growth(self._detect_industry_from_financials(financials), forecast_years)
            
            # Get predictions from trained models
            last_sequence = X_seq[-1:].copy()
            growth_predictions = []
            
            # Generate predictions for each year
            for year in range(forecast_years):
                # Get predictions from both models
                lstm_pred = self.lstm_model.predict(last_sequence, verbose=0)[0][0]
                attn_pred = self.attention_model.predict(last_sequence, verbose=0)[0][0]
                
                # Weighted average (trust LSTM more in early years)
                lstm_weight = max(0.7 - (year * 0.1), 0.3)
                ensemble_pred = lstm_pred * lstm_weight + attn_pred * (1 - lstm_weight)
                
                # Convert to actual growth rate
                growth_rate = float(self.scaler_y.inverse_transform(
                    np.array([ensemble_pred]).reshape(-1, 1)
                )[0][0])
                
                # Store prediction
                growth_predictions.append(growth_rate)
                
                # Update sequence for next year's prediction
                if year < forecast_years - 1:
                    try:
                        # Create updated data point
                        new_data_point = last_sequence[0, -1].copy().reshape(1, -1)
                        growth_idx = min(2, new_data_point.shape[1] - 1)
                        new_data_point[0, growth_idx] = ensemble_pred
                        
                        # Create new sequence
                        last_sequence = np.concatenate([
                            last_sequence[0, 1:].reshape(1, self.sequence_length - 1, -1),
                            new_data_point.reshape(1, 1, -1)
                        ], axis=1)
                    except Exception as e:
                        logger.warning(f"Error updating sequence: {e}")
            
            # Check for constant pattern issues (especially 30% pattern)
            constant_pattern = all(abs(g - 0.3) < 0.02 for g in growth_predictions)
            high_values = all(g > 0.2 for g in growth_predictions)
            
            # Apply fixes if problematic patterns detected
            if constant_pattern or high_values:
                logger.warning(f"Detected problematic growth pattern: {[f'{x:.2%}' for x in growth_predictions]}")
                industry_type = self._detect_industry_from_financials(financials)
                
                # Get reasonable first year growth
                first_year = growth_predictions[0]
                if constant_pattern or first_year > 0.3:
                    # Use industry-specific cap instead
                    industry_caps = {
                        'tech': 0.25,
                        'semiconductor': 0.28,
                        'healthcare': 0.15,
                        'utilities': 0.07,
                        'finance': 0.12,
                        'consumer': 0.10,
                        'telecom': 0.05,
                        'default': 0.15
                    }
                    first_year = min(first_year, industry_caps.get(industry_type, 0.15))
                
                # Apply industry-specific decay
                decay_rates = {
                    'tech': 0.75,
                    'semiconductor': 0.70,
                    'healthcare': 0.80,
                    'utilities': 0.90,
                    'finance': 0.80,
                    'consumer': 0.85,
                    'telecom': 0.90,
                    'energy': 0.80,
                    'default': 0.80
                }
                
                # Create realistic decay pattern
                revised_predictions = [first_year]
                decay_rate = decay_rates.get(industry_type, 0.80)
                
                for i in range(1, forecast_years):
                    # Apply stronger decay in early years
                    decay_factor = decay_rate - (0.05 * min(i, 2))
                    next_val = revised_predictions[-1] * decay_factor
                    
                    # Add small variation
                    variation = np.random.normal(0, 0.01)
                    next_val = max(0.01, next_val + variation)
                    
                    # Apply industry-specific minimum
                    min_rate = 0.02 if industry_type in ['utilities', 'telecom'] else 0.03
                    next_val = max(min_rate, next_val)
                    
                    revised_predictions.append(next_val)
                
                logger.info(f"Revised DL predictions: {[f'{x:.2%}' for x in revised_predictions]}")
                return revised_predictions
            
            # Final sanity check to ensure declining pattern in later years
            for i in range(1, len(growth_predictions)):
                if growth_predictions[i] > growth_predictions[i-1] * 0.95:
                    # Enforce at least 5% decline each year
                    growth_predictions[i] = growth_predictions[i-1] * 0.95
            
            logger.info(f"Final DL growth predictions: {[f'{x:.2%}' for x in growth_predictions]}")
            return growth_predictions
            
        except Exception as e:
            logger.error(f"Error predicting with deep learning: {e}")
            # Fallback to varied declining pattern
            return self._generate_industry_baseline_growth(
                self._detect_industry_from_financials(financials), 
                forecast_years
            )

    def _generate_industry_baseline_growth(self, industry: str, years: int = 5) -> List[float]:
        """Generate industry-specific growth pattern when models fail."""
        # Try to map 'default' to a better industry based on any available info
        if industry == 'default' or industry is None:
            # Try to extract ticker information from financials if available
            ticker = None
            if hasattr(self, 'financials') and isinstance(self.financials, dict) and 'stock_code' in self.financials:
                ticker = self.financials['stock_code']
            elif hasattr(self, 'financials') and hasattr(self.financials, 'attrs') and 'stock_code' in self.financials.attrs:
                ticker = self.financials.attrs['stock_code']
                
            # If ticker found, try to derive industry from it (especially Taiwan stocks)
            if ticker and isinstance(ticker, str):
                if '.TW' in ticker:
                    base_number = ticker.split('.')[0]
                    # Basic industry mapping
                    if base_number in ['2330', '2454', '2379', '2337', '2308', '2303', '2409', '2344', '2351']:
                        logger.info(f"Mapping {ticker} from 'default' to 'semiconductor' in baseline generator")
                        industry = 'semiconductor'
                    elif base_number.startswith('23') or base_number in ['2317', '2356']:
                        logger.info(f"Mapping {ticker} from 'default' to 'tech' in baseline generator")
                        industry = 'tech'
                    elif base_number in ['2412', '3045', '4904', '4977']:
                        logger.info(f"Mapping {ticker} from 'default' to 'telecom' in baseline generator")
                        industry = 'telecom'
                    elif base_number.startswith('26') or base_number.startswith('27'):
                        logger.info(f"Mapping {ticker} from 'default' to 'finance' in baseline generator")
                        industry = 'finance'
        
        # Industry parameters
        industry_params = {
            'tech': {'base': 0.15, 'decay': 0.80, 'floor': 0.04},
            'semiconductor': {'base': 0.18, 'decay': 0.75, 'floor': 0.04},
            'healthcare': {'base': 0.10, 'decay': 0.85, 'floor': 0.03},
            'utilities': {'base': 0.04, 'decay': 0.90, 'floor': 0.02},
            'finance': {'base': 0.07, 'decay': 0.85, 'floor': 0.03},
            'telecom': {'base': 0.03, 'decay': 0.90, 'floor': 0.01},
            'consumer': {'base': 0.06, 'decay': 0.85, 'floor': 0.02},
            'energy': {'base': 0.08, 'decay': 0.80, 'floor': 0.02},
            'default': {'base': 0.10, 'decay': 0.85, 'floor': 0.03},
            'electronics': {'base': 0.14, 'decay': 0.78, 'floor': 0.04},
            'computer_hardware': {'base': 0.12, 'decay': 0.80, 'floor': 0.03}, 
            'retail': {'base': 0.08, 'decay': 0.82, 'floor': 0.02},
            'materials': {'base': 0.06, 'decay': 0.85, 'floor': 0.02},
            'industrial': {'base': 0.07, 'decay': 0.82, 'floor': 0.02}
        }
        
        # Get parameters for this industry - with more flexible matching
        industry_key = industry.lower() if industry else 'default'
        params = None
        
        # Try exact match first
        if industry_key in industry_params:
            params = industry_params[industry_key]
        else:
            # Try partial/fuzzy matching
            for key in industry_params:
                if key in industry_key or industry_key in key:
                    logger.info(f"Using growth parameters for '{key}' instead of '{industry_key}'")
                    params = industry_params[key]
                    break
        
        # Fallback to default if still no match
        if params is None:
            logger.info(f"No matching industry parameters for '{industry_key}', using default")
            params = industry_params['default']
        
        # Generate realistic growth pattern
        result = [params['base']]
        for i in range(1, years):
            # Apply decay
            decay = params['decay'] ** (1 + 0.2 * min(i, 3))
            next_val = max(params['floor'], result[-1] * decay)
            
            # Add small variation
            variation = np.random.normal(0, 0.01)
            next_val = max(params['floor'], next_val + variation)
            
            result.append(next_val)
        
        logger.info(f"Generated industry-baseline pattern for {industry_key}: {[f'{x:.2%}' for x in result]}")
        return result

    def _detect_industry_from_financials(self, financials: pd.DataFrame) -> str:
        """Detect the likely industry based on financial metrics and ticker patterns."""
        if financials is None:
            return 'default'
        
        try:
            # Extract ticker if available
            stock_code = None
            # Initialize metrics dictionary to avoid "referenced before assignment" error
            metrics = {}
            
            # Handle dictionary input
            if isinstance(financials, dict):
                if 'stock_code' in financials:
                    stock_code = financials['stock_code']
                elif 'ticker' in financials:
                    stock_code = financials['ticker']
                
                # We can't use index-based detection with a dict, so we'll extract metrics directly
                if 'gross_margin' in financials:
                    metrics['gross_margin'] = financials['gross_margin']
                elif 'operating_margin' in financials:
                    metrics['op_margin'] = financials['operating_margin']
                elif 'historical_operating_income' in financials and 'historical_revenue' in financials:
                    # Try to calculate operating margin from historical data
                    op_income = financials['historical_operating_income'][-1] if financials['historical_operating_income'] else 0
                    revenue = financials['historical_revenue'][-1] if financials['historical_revenue'] else 1
                    if revenue > 0:
                        metrics['op_margin'] = op_income / revenue
            
            # Handle DataFrame input
            elif hasattr(financials, 'attrs') and 'stock_code' in financials.attrs:
                stock_code = financials.attrs['stock_code']
                
                # Financial ratio-based detection for DataFrame
                
                # Check for high gross margins
                if 'Gross Margin' in financials.index:
                    gross_margin = np.mean([float(x) for x in financials.loc['Gross Margin'] if pd.notna(x)])
                    metrics['gross_margin'] = gross_margin
                
                # Check for operating margins
                if 'Operating Margin' in financials.index:
                    op_margin = np.mean([float(x) for x in financials.loc['Operating Margin'] if pd.notna(x)])
                    metrics['op_margin'] = op_margin
            
            # Expanded Taiwan stock classification based on stock code
            if stock_code and isinstance(stock_code, str):
                # Extract base number for Taiwan stocks
                base_number = None
                if '.' in stock_code:
                    parts = stock_code.split('.')
                    if len(parts) >= 2 and parts[1] in ['TW', 'TWO']:
                        base_number = parts[0]
                        
                        # Taiwan semiconductors - expanded list
                        if base_number in ['2330', '2454', '2379', '2337', '2308', '2303', '2409', '2344', '2351', '2408',
                                         '3707', '5347', '3105', '3545', '6239']:
                            return 'semiconductor'
                            
                        # Taiwan tech hardware - expanded list
                        if base_number in ['2317', '2382', '2354', '2353', '2474', '2357', '2324', '2327', '2356',
                                         '2377', '2395', '2376', '3231', '2301']:
                            return 'tech'
                            
                        # Taiwan telecom
                        if base_number in ['2412', '3045', '4904', '4977', '2406', '4977']:
                            return 'telecom'
                            
                        # Taiwan financial
                        if (base_number.startswith('26') or 
                            (base_number.startswith('27') and len(base_number) == 4) or
                            base_number in ['2801', '2809', '2812', '2823', '2834', '2836', '2838', '2845', '2867']):
                            return 'finance'
                        
                        # Taiwan utilities
                        if base_number.startswith('9') and len(base_number) == 4:
                            return 'utilities'
                            
                        # Taiwan electronics
                        if (base_number.startswith('23') or 
                            base_number in ['6271', '6411', '6488', '8069', '8271', '2458', '2368']):
                            return 'electronics'
                            
                        # Taiwan food & beverage
                        if base_number in ['1101', '1216', '1301', '1326', '1333', '1789', '1909', '9945']:
                            return 'consumer'
                            
                        # Taiwan healthcare
                        if base_number in ['1476', '1762', '4103', '4107', '4119', '4133', '4137', '4142', '4144', '4164']:
                            return 'healthcare'
                            
                        # Taiwan chemicals - adding chemical companies
                        if base_number in ['1301', '1303', '1304', '1309', '1313', '1314', '1319', '1321', '1323', 
                                         '1324', '1338', '1702', '1708', '1710', '1713', '1717', '1718', '1722', 
                                         '1725', '1726', '1730', '1762', '4763', '4767', '4768', '4722', '4720',
                                         '4737', '4744', '4746', '4755', '4762', '4764', '4766', '4772', '4774']:
                            logger.info(f"Identified {stock_code} as chemicals industry")
                            return 'chemicals'
            
            # Use ratios to identify industry when ticker-based detection fails
            if 'gross_margin' in metrics and 'op_margin' in metrics:
                gm = metrics['gross_margin']
                om = metrics['op_margin']
                
                if gm > 0.60 and om > 0.25:  # High margins
                    return 'tech'
                elif gm > 0.40 and om > 0.15:
                    return 'healthcare'
                elif gm < 0.25 and om < 0.10:
                    return 'energy'
                elif gm > 0.30 and om < 0.15:
                    return 'consumer'
            
            return 'default'
            
        except Exception as e:
            logger.warning(f"Error detecting industry: {e}")
            return 'default'
    
    def visualize_predictions(self, historical_growth: List[float], predicted_growth: List[float]):
        """Visualize historical vs predicted growth rates."""
        years_hist = range(1, len(historical_growth) + 1)
        years_pred = range(len(historical_growth) + 1, len(historical_growth) + len(predicted_growth) + 1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(years_hist, historical_growth, marker='o', label='Historical Growth')
        plt.plot(years_pred, predicted_growth, marker='x', linestyle='--', label='DL Predicted Growth')
        
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.xlabel('Year')
        plt.ylabel('Growth Rate')
        plt.title('Deep Learning Growth Rate Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt
