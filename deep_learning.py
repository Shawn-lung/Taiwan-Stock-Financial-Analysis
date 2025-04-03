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
import datetime

logger = logging.getLogger(__name__)

class DeepFinancialForecaster:
    """Deep learning-based financial forecasting models."""
    
    def __init__(self, sequence_length: int = 3):
        """Initialize the deep forecaster.
        
        Args:
            sequence_length: Number of time steps to use for sequence models
        """
        self.sequence_length = sequence_length
        self.lstm_model = None
        self.attention_model = None
        self.hybrid_model = None
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
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
        
    def prepare_financial_sequences(self, financials: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and prepare sequences from financial data."""
        # Extract relevant financial metrics
        features = []
        dates = sorted(financials.columns)
        logger.info(f"Preparing financial data with {len(dates)} time periods")
        
        # Fix orientation of financials DataFrame if needed
        if isinstance(financials.index[0], (str, pd.Timestamp, datetime)) and 'Total Revenue' not in financials.index:
            logger.info("Transposing financial data to ensure metrics are in rows")
            financials = financials.T
        
        for date in dates:
            try:
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
        
        # Enhanced validation and logging
        if len(features) < self.sequence_length + 1:
            logger.warning(f"Insufficient financial data: {len(features)} points available, {self.sequence_length + 1} needed")
            return None, None
        logger.info(f"Successfully extracted {len(features)} financial data points")
        
        # Convert to DataFrame and calculate growth rates
        df = pd.DataFrame(features)
        df['revenue_growth'] = df['revenue'].pct_change().fillna(0)
        df['op_income_growth'] = df['op_income'].pct_change().fillna(0)
        df['net_income_growth'] = df['net_income'].pct_change().fillna(0)
        
        # Add more meaningful features
        df['revenue_log'] = np.log(df['revenue'].clip(lower=1))
        df['revenue_growth_trend'] = df['revenue_growth'].rolling(3).mean().fillna(0)
        df['margin_trend'] = df['op_margin'].rolling(3).mean().fillna(df['op_margin'])
        
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
        return self._prepare_sequence_data(X_scaled, y_scaled)
        
    def train_growth_forecaster(self, financials: pd.DataFrame, validation_split: float = 0.2) -> bool:
        """Train deep learning models on historical financial data."""
        try:
            # Prepare sequence data
            X_seq, y_seq = self.prepare_financial_sequences(financials)
            if X_seq is None or len(X_seq) < 3:  # Need enough samples
                logger.warning("Insufficient data for deep learning training")
                return False
            
            # Get input dimensions
            _, _, input_dim = X_seq.shape
            logger.info(f"Training deep models with {len(X_seq)} sequences of {self.sequence_length} timesteps and {input_dim} features")
            
            # Build models
            self.lstm_model = self.build_lstm_model(input_dim)
            self.attention_model = self.build_attention_model(input_dim)
            
            # Set up early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
            )
            
            # Train LSTM model
            logger.info("Training LSTM model...")
            lstm_history = self.lstm_model.fit(
                X_seq, y_seq,
                epochs=100,
                batch_size=min(8, len(X_seq)),  # Ensure batch size <= number of samples
                validation_split=min(validation_split, 0.5),  # Limit validation split for small datasets
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Train attention model
            logger.info("Training attention model...")
            attention_history = self.attention_model.fit(
                X_seq, y_seq,
                epochs=100,
                batch_size=min(8, len(X_seq)),
                validation_split=min(validation_split, 0.5),
                callbacks=[early_stopping],
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
            
            if lstm_var < 1e-6 and attn_var < 1e-6:
                logger.warning("Models are producing near-constant outputs - they may not be learning properly")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error training deep learning models: {e}")
            return False

    def predict_future_growth(self, financials: pd.DataFrame, forecast_years: int = 5) -> List[float]:
        """Predict future growth rates using ensemble of deep learning models."""
        try:
            # First train the models
            success = self.train_growth_forecaster(financials)
            
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

    def _generate_industry_baseline_growth(self, industry: str, years: int) -> List[float]:
        """Generate industry-specific growth pattern when models fail."""
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
            'default': {'base': 0.10, 'decay': 0.85, 'floor': 0.03}
        }
        
        # Get parameters for this industry
        params = industry_params.get(industry, industry_params['default'])
        
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
        
        logger.info(f"Generated industry-baseline pattern for {industry}: {[f'{x:.2%}' for x in result]}")
        return result

    def _detect_industry_from_financials(self, financials: pd.DataFrame) -> str:
        """Detect the likely industry based on financial metrics and ticker patterns."""
        if financials is None:
            return 'default'
        
        try:
            # Extract ticker if available
            stock_code = None
            if hasattr(financials, 'attrs') and 'stock_code' in financials.attrs:
                stock_code = financials.attrs['stock_code']
            
            # Taiwan stock classification
            if stock_code and isinstance(stock_code, str):
                if '.' in stock_code:
                    parts = stock_code.split('.')
                    if len(parts) >= 2 and parts[1] in ['TW', 'TWO']:
                        base_number = parts[0]
                        
                        # Taiwan semiconductors
                        if base_number in ['2330', '2454', '2379', '2337', '2308', '2303', '2409', '2344', '2351', '2408']:
                            return 'semiconductor'
                            
                        # Taiwan tech hardware
                        if base_number in ['2317', '2382', '2354', '2353', '2474', '2357', '2324', '2327', '2356']:
                            return 'tech'
                            
                        # Taiwan telecom
                        if base_number in ['2412', '3045', '4904', '4977']:
                            return 'telecom'
                            
                        # Taiwan financial
                        if base_number.startswith('26') or (base_number.startswith('27') and len(base_number) == 4):
                            return 'finance'
                        
                        # Taiwan utilities
                        if base_number.startswith('9') and len(base_number) == 4:
                            return 'utilities'
                            
            # Financial ratio-based detection
            metrics = {}
            
            # Check for high gross margins
            if 'Gross Margin' in financials.index:
                gross_margin = np.mean([float(x) for x in financials.loc['Gross Margin'] if pd.notna(x)])
                metrics['gross_margin'] = gross_margin
            
            # Check for operating margins
            if 'Operating Margin' in financials.index:
                op_margin = np.mean([float(x) for x in financials.loc['Operating Margin'] if pd.notna(x)])
                metrics['op_margin'] = op_margin
            
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
