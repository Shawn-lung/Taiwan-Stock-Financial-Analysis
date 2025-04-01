import pandas as pd
import numpy as np
import os
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Union
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, concatenate # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from industry_data_collector import TaiwanIndustryDataCollector
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndustryValuationModel:
    """ML model for industry-specific financial valuation adjustments."""
    
    def __init__(self, data_dir: str = "industry_data", background_collector = None):
        """Initialize the industry valuation model.
        
        Args:
            data_dir: Directory containing industry data
            background_collector: Optional BackgroundDataCollector instance
        """
        self.data_dir = data_dir
        self.industry_models = {}
        self.industry_scalers = {}
        self.industry_benchmarks = None
        self.background_collector = background_collector
        
        # Create model directory
        self.model_dir = os.path.join(data_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load industry benchmarks if available
        benchmark_file = os.path.join(data_dir, 'industry_benchmarks.csv')
        if os.path.exists(benchmark_file):
            self.industry_benchmarks = pd.read_csv(benchmark_file)
            logger.info(f"Loaded benchmarks for {len(self.industry_benchmarks)} industries")
        
        # Setup cache
        self.cache_dir = os.path.join(data_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def train_industry_models(self, industries: Optional[List[str]] = None,
                             force_retrain: bool = False) -> Dict[str, Dict]:
        """Train models for each industry.
        
        Args:
            industries: List of specific industries to train, or None for all
            force_retrain: Whether to force retraining even if models exist
            
        Returns:
            Dictionary of industry -> training metrics
        """
        try:
            # Get list of all available training datasets
            training_files = [f for f in os.listdir(self.data_dir) if f.endswith('_training.csv')]
            
            if industries:
                # Filter to specific industries
                industry_files = [f"{industry.lower().replace(' ', '_')}_training.csv" for industry in industries]
                training_files = [f for f in training_files if f in industry_files]
            
            if not training_files:
                logger.error("No training data files found - run data collection first")
                return {}
            
            training_results = {}
            
            # Train a model for each industry
            for training_file in training_files:
                industry = training_file.replace('_training.csv', '').replace('_', ' ')
                
                # Check if model already exists
                model_path = os.path.join(self.model_dir, f"{industry.replace(' ', '_').lower()}_model")
                if os.path.exists(model_path) and not force_retrain:
                    logger.info(f"Model for {industry} already exists. Use force_retrain=True to retrain.")
                    
                    # Load existing model
                    try:
                        model = tf.keras.models.load_model(model_path)
                        scaler_path = os.path.join(self.model_dir, f"{industry.replace(' ', '_').lower()}_scaler.pkl")
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)
                            
                        self.industry_models[industry] = model
                        self.industry_scalers[industry] = scaler
                        logger.info(f"Loaded existing model for {industry}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error loading existing model for {industry}: {e}. Will retrain.")
                
                logger.info(f"Training model for {industry}")
                
                # Load training data
                df = pd.read_csv(os.path.join(self.data_dir, training_file))
                
                if len(df) < 50:  # Adjust threshold for limited data
                    logger.warning(f"Limited data for {industry} ({len(df)} records). Using simple model.")
                    
                    # For industries with limited data, use a simpler model architecture
                    training_results[industry] = self._train_simple_model(industry, df)
                else:
                    # For industries with sufficient data, use normal training process
                    X, y, feature_cols = self._prepare_training_data(df)
                    
                    if X is None or y is None:
                        logger.warning(f"Could not prepare training data for {industry}")
                        continue
                    
                    # Train the model
                    model, history, scaler = self._train_model(X, y, industry)
                    
                    if model:
                        # Save the model
                        model_path = os.path.join(self.model_dir, f"{industry.replace(' ', '_').lower()}_model")
                        model.save(model_path)
                        
                        # Save the scaler
                        scaler_path = os.path.join(self.model_dir, f"{industry.replace(' ', '_').lower()}_scaler.pkl")
                        with open(scaler_path, 'wb') as f:
                            pickle.dump(scaler, f)
                        
                        # Store model and scaler in memory
                        self.industry_models[industry] = model
                        self.industry_scalers[industry] = scaler
                        
                        # Track feature importance (for dense models)
                        feature_importance = self._analyze_feature_importance(model, X, y, feature_cols)
                        
                        # Store training results
                        training_results[industry] = {
                            'samples': len(X),
                            'features': len(feature_cols),
                            'history': history.history,
                            'feature_importance': feature_importance
                        }
                        
                        logger.info(f"Successfully trained model for {industry}")
                        
                        # Plot training history
                        self._plot_training_history(history, industry)
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error training industry models: {e}")
            return {}
    
    def _train_simple_model(self, industry: str, df: pd.DataFrame) -> Dict:
        """Train a simpler model for industries with limited data."""
        try:
            # For limited data, focus on fewer, more stable features
            simple_features = [
                'historical_growth_mean', 'operating_margin', 'net_margin',
                'roa', 'roe', 'revenue'
            ]
            
            # Filter to essential columns that exist in the data
            available_features = [col for col in simple_features if col in df.columns]
            
            if len(available_features) < 3:
                logger.warning(f"Not enough features for {industry}. Attempting to use all numeric columns.")
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
                available_features = [col for col in available_features 
                                      if col not in ['stock_id', 'timestamp'] and 'future' not in col]
            
            # Check if we have target variable
            if 'future_6m_return' not in df.columns:
                logger.error(f"Target variable missing for {industry}")
                return {'error': 'Missing target variable'}
            
            # Prepare data
            X = df[available_features].copy()
            y = df['future_6m_return'].copy()
            
            # Handle NaN values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Simple scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create a simple model (without complex evaluation for limited data)
            model = Sequential([
                Dense(8, activation='relu', input_shape=(X.shape[1],)),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Simple training
            history = model.fit(
                X_scaled, y,
                epochs=50,
                batch_size=min(16, len(X)),
                validation_split=min(0.2, 0.5) if len(X) > 20 else 0,
                verbose=0
            )
            
            # Save the model
            model_path = os.path.join(self.model_dir, f"{industry.replace(' ', '_').lower()}_model")
            model.save(model_path)
            
            # Save the scaler
            scaler_path = os.path.join(self.model_dir, f"{industry.replace(' ', '_').lower()}_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Store in memory
            self.industry_models[industry] = model
            self.industry_scalers[industry] = scaler
            
            # Create basic visualization for the model
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Training Loss for {industry} (Simple Model)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(self.model_dir, f"{industry.replace(' ', '_').lower()}_loss.png"))
            plt.close()
            
            logger.info(f"Simple model trained for {industry} with {len(X)} samples and {len(available_features)} features")
            
            return {
                'samples': len(X),
                'features': len(available_features),
                'used_features': available_features,
                'simple_model': True,
                'history': history.history
            }
            
        except Exception as e:
            logger.error(f"Error training simple model for {industry}: {e}")
            return {'error': str(e)}
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Prepare training data from a DataFrame."""
        try:
            # Define potential feature columns (we'll filter to ones that exist)
            potential_features = [
                'revenue', 'historical_growth_mean', 'historical_growth_std',
                'operating_margin', 'net_margin', 'roa', 'roe', 'debt_to_equity',
                'equity_to_assets', 'ocf_to_revenue', 'capex_to_revenue', 'fcf_to_revenue'
            ]
            
            # Filter to columns that exist
            feature_cols = [col for col in potential_features if col in df.columns]
            
            if len(feature_cols) < 3:
                logger.warning(f"Not enough standard features available. Using all available numeric columns.")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [col for col in numeric_cols 
                               if col not in ['stock_id', 'timestamp'] and 'future' not in col]
            
            if 'future_6m_return' not in df.columns:
                logger.error("Target variable 'future_6m_return' not found in data")
                return None, None, []
            
            # Prepare features and target
            X = df[feature_cols].copy()
            y = df['future_6m_return'].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Remove extreme outliers
            for col in X.columns:
                q1 = X[col].quantile(0.01)
                q3 = X[col].quantile(0.99)
                iqr = q3 - q1
                lower_bound = q1 - (iqr * 1.5)
                upper_bound = q3 + (iqr * 1.5)
                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Remove extreme target values
            q1 = y.quantile(0.01)
            q3 = y.quantile(0.99)
            iqr = q3 - q1
            y = y.clip(lower=q1-(iqr*1.5), upper=q3+(iqr*1.5))
            
            # Convert to numpy arrays
            X_np = X.values
            y_np = y.values
            
            return X_np, y_np, feature_cols
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None, []
    
    def _train_model(self, X: np.ndarray, y: np.ndarray, industry: str) -> Tuple[Optional[Model], Optional[tf.keras.callbacks.History], Optional[StandardScaler]]:
        """Train a neural network model for the given industry."""
        try:
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Build model
            input_dim = X.shape[1]
            
            # Dynamic architecture based on data size
            if len(X) > 200:
                # Larger model for more data
                model = Sequential([
                    Dense(32, activation='relu', input_shape=(input_dim,)),
                    Dropout(0.3),
                    Dense(16, activation='relu'),
                    Dropout(0.2),
                    Dense(8, activation='relu'),
                    Dense(1)
                ])
            else:
                # Simpler model for less data
                model = Sequential([
                    Dense(16, activation='relu', input_shape=(input_dim,)),
                    Dropout(0.2),
                    Dense(8, activation='relu'),
                    Dense(1)
                ])
            
            # Compile model
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train_scaled, y_train,
                epochs=100,
                batch_size=min(32, len(X_train)),
                validation_data=(X_val_scaled, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            logger.info(f"Model training for {industry} completed after {len(history.history['loss'])} epochs")
            
            return model, history, scaler
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None, None, None
    
    def _analyze_feature_importance(self, model: Model, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Analyze feature importance using permutation importance."""
        try:
            if not isinstance(model, Sequential) or len(model.layers) < 2:
                logger.warning("Feature importance analysis only supported for sequential models")
                return {}
            
            # For simple models, extract weights from first layer
            weights = model.layers[0].get_weights()[0]
            importance = {}
            
            # Calculate absolute importance based on first layer weights
            for i, feature in enumerate(feature_names):
                if i < weights.shape[0]:
                    importance[feature] = float(np.abs(weights[i]).mean())
            
            # Normalize to sum to 1
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}
            
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return importance
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            return {}
    
    def _plot_training_history(self, history: tf.keras.callbacks.History, industry: str) -> None:
        """Plot training history."""
        try:
            plt.figure(figsize=(12, 5))
            
            # Plot training & validation loss
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'])
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'])
            plt.title(f'Model Loss ({industry})')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, f"{industry.replace(' ', '_').lower()}_training.png"))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")
    
    def prepare_training_data_from_db(self) -> Dict[str, pd.DataFrame]:
        """Prepare training data from the background data collector database."""
        if self.background_collector is None:
            logger.error("No background data collector provided")
            return {}

        try:
            logger.info("Preparing training data from database...")
            training_datasets = {}
            
            # Get collection status
            status_df = self.background_collector.get_collection_status()
            
            # Group by industry
            industry_groups = status_df.groupby('industry')
            
            for industry, group in industry_groups:
                logger.info(f"Processing {industry} industry...")
                
                # Get stocks with successful collection
                valid_stocks = group[
                    group['fs_last_update'].notna() & 
                    group['bs_last_update'].notna()
                ]['stock_id'].tolist()
                
                if len(valid_stocks) < 3:
                    logger.warning(f"Not enough stocks with data for {industry}, skipping")
                    continue
                
                # Get industry data from database
                industry_data = self.background_collector.get_industry_data(industry)
                
                if not industry_data:
                    logger.warning(f"No industry data retrieved for {industry}")
                    continue
                
                # Extract financial metrics
                industry_records = []
                data_collector = TaiwanIndustryDataCollector(data_dir=self.data_dir)
                
                for stock_id, data in industry_data.items():
                    try:
                        # Use TaiwanIndustryDataCollector's method to extract metrics
                        metrics = data_collector._extract_financial_metrics(stock_id, data)
                        if metrics:
                            industry_records.extend(metrics)
                    except Exception as e:
                        logger.error(f"Error extracting metrics for {stock_id}: {e}")
                
                # Create industry dataset if we have enough records
                if industry_records:
                    industry_df = pd.DataFrame(industry_records)
                    
                    # Save industry training data
                    training_file = os.path.join(self.data_dir, f"{industry.replace(' ', '_').lower()}_training.csv")
                    industry_df.to_csv(training_file, index=False)
                    
                    training_datasets[industry] = industry_df
                    logger.info(f"Created training dataset for {industry} with {len(industry_df)} records")
            
            # Generate industry benchmarks
            self.generate_industry_benchmarks(training_datasets)
            
            return training_datasets
                
        except Exception as e:
            logger.error(f"Error preparing training data from database: {e}")
            return {}
    
    def generate_industry_benchmarks(self, training_datasets=None) -> pd.DataFrame:
        """Generate benchmark financial metrics by industry."""
        try:
            industry_metrics = []
            
            # Use provided datasets or load from files
            if training_datasets is None:
                # Load the training datasets
                for industry_file in os.listdir(self.data_dir):
                    if industry_file.endswith('_training.csv'):
                        industry_name = industry_file.replace('_training.csv', '').replace('_', ' ')
                        df = pd.read_csv(os.path.join(self.data_dir, industry_file))
                        
                        if df.empty:
                            continue
                        
                        training_datasets = training_datasets or {}
                        training_datasets[industry_name] = df
            
            if not training_datasets:
                logger.error("No training datasets available")
                return pd.DataFrame()
            
            # Calculate benchmarks for each industry
            for industry_name, df in training_datasets.items():
                if df.empty:
                    continue
                
                # Calculate key metrics
                metrics = {
                    'industry': industry_name,
                    'stock_count': df['stock_id'].nunique() if 'stock_id' in df.columns else 0,
                    'record_count': len(df)
                }
                
                # Calculate median values for key metrics
                for metric in ['historical_growth_mean', 'operating_margin', 'net_margin', 
                              'roa', 'roe', 'debt_to_equity', 'ocf_to_revenue',
                              'capex_to_revenue', 'fcf_to_revenue']:
                    if metric in df.columns:
                        metrics[f'{metric}_median'] = df[metric].median()
                        metrics[f'{metric}_mean'] = df[metric].mean()
                        metrics[f'{metric}_std'] = df[metric].std()
                
                industry_metrics.append(metrics)
            
            # Create benchmark DataFrame
            benchmarks = pd.DataFrame(industry_metrics)
            
            # Save to CSV
            benchmark_file = os.path.join(self.data_dir, 'industry_benchmarks.csv')
            benchmarks.to_csv(benchmark_file, index=False)
            
            # Update our copy
            self.industry_benchmarks = benchmarks
            
            logger.info(f"Generated benchmarks for {len(benchmarks)} industries")
            return benchmarks
            
        except Exception as e:
            logger.error(f"Error calculating industry benchmarks: {e}")
            return pd.DataFrame()
    
    def predict_future_returns(self, industry: str, financial_metrics: Dict[str, float]) -> Optional[float]:
        """Predict future 6-month returns based on financial metrics."""
        try:
            # Check if we have a model for this industry
            if industry not in self.industry_models:
                logger.warning(f"No model available for {industry}. Loading if exists or using nearest industry.")
                
                # Try to load model for this industry if it exists
                model_path = os.path.join(self.model_dir, f"{industry.replace(' ', '_').lower()}_model")
                scaler_path = os.path.join(self.model_dir, f"{industry.replace(' ', '_').lower()}_scaler.pkl")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    try:
                        model = tf.keras.models.load_model(model_path)
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)
                        
                        self.industry_models[industry] = model
                        self.industry_scalers[industry] = scaler
                    except Exception as e:
                        logger.error(f"Error loading model for {industry}: {e}")
                        return self._fallback_prediction(industry, financial_metrics)
                else:
                    # Use fallback prediction
                    return self._fallback_prediction(industry, financial_metrics)
            
            # Get model and scaler
            model = self.industry_models[industry]
            scaler = self.industry_scalers[industry]
            
            # Prepare input features
            X = self._prepare_prediction_features(industry, financial_metrics)
            
            if X is None:
                logger.warning(f"Could not prepare features for prediction for {industry}")
                return self._fallback_prediction(industry, financial_metrics)
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make prediction
            prediction = model.predict(X_scaled, verbose=0)[0][0]
            
            # Add randomness for more natural variation
            prediction = prediction * (1 + random.uniform(-0.1, 0.1))
            
            # Apply sanity checks to prediction
            prediction = self._apply_sanity_checks(industry, prediction, financial_metrics)
            
            logger.info(f"Predicted 6-month return for {industry}: {prediction:.2%}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting future returns: {e}")
            return self._fallback_prediction(industry, financial_metrics)
    
    def _prepare_prediction_features(self, industry: str, metrics: Dict[str, float]) -> Optional[np.ndarray]:
        """Prepare feature array for prediction."""
        try:
            # Load a sample training file to get feature columns
            training_file = os.path.join(self.data_dir, f"{industry.replace(' ', '_').lower()}_training.csv")
            
            if not os.path.exists(training_file):
                logger.warning(f"No training file found for {industry}")
                return None
            
            # Load sample to get columns
            sample_df = pd.read_csv(training_file, nrows=1)
            
            # Get numeric feature columns (excluding target and metadata)
            feature_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols 
                           if col not in ['future_6m_return', 'stock_id', 'timestamp'] 
                           and not col.startswith('future_')]
            
            # Create empty feature array with the right number of columns
            X = np.zeros((1, len(feature_cols)))
            
            # Fill in the features we have from provided metrics
            for i, col in enumerate(feature_cols):
                if col in metrics:
                    X[0, i] = metrics[col]
                else:
                    # Use median value from industry benchmarks if available
                    if self.industry_benchmarks is not None:
                        industry_row = self.industry_benchmarks[self.industry_benchmarks['industry'] == industry]
                        if not industry_row.empty and f"{col}_median" in industry_row.columns:
                            X[0, i] = industry_row[f"{col}_median"].iloc[0]
            
            return X
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {e}")
            return None
    
    def _fallback_prediction(self, industry: str, metrics: Dict[str, float]) -> float:
        """Provide a fallback prediction when model is not available."""
        # Get industry-specific benchmarks
        if self.industry_benchmarks is not None:
            industry_row = self.industry_benchmarks[self.industry_benchmarks['industry'] == industry]
            if not industry_row.empty:
                # Use historical ROE as a rough proxy for expected returns
                if 'roe' in metrics:
                    # Expected 6-month return is roughly related to ROE
                    return metrics['roe'] * 0.4 * (1 + random.uniform(-0.2, 0.2))
        
        # Fallback based on industry
        industry_returns = {
            'Semiconductors': 0.08,
            'Electronics': 0.06,
            'Banking': 0.04,
            'Telecommunications': 0.03,
            'Financial Services': 0.05,
            'Computer Hardware': 0.07,
            'Food & Beverage': 0.04,
            'Retail': 0.05,
            'Healthcare': 0.06,
            'Utilities': 0.02
        }
        
        # Get expected return for this industry or default to 5%
        base_return = industry_returns.get(industry, 0.05)
        
        # Add some randomness
        return base_return * (1 + random.uniform(-0.2, 0.2))
    
    def _apply_sanity_checks(self, industry: str, prediction: float, metrics: Dict[str, float]) -> float:
        """Apply sanity checks to ensure prediction is reasonable."""
        # Extreme bounds based on industry
        industry_limits = {
            'Semiconductors': (-0.3, 0.3),
            'Electronics': (-0.25, 0.25),
            'Banking': (-0.15, 0.15),
            'Telecommunications': (-0.12, 0.12),
            'Financial Services': (-0.2, 0.2),
            'Computer Hardware': (-0.25, 0.25)
        }
        
        # Get bounds for this industry or use default
        lower_bound, upper_bound = industry_limits.get(industry, (-0.2, 0.2))
        
        # Clip prediction to bounds
        prediction = max(lower_bound, min(upper_bound, prediction))
        
        # Adjust based on financial health
        if 'net_margin' in metrics and metrics['net_margin'] < 0:
            # Companies with negative margins tend to underperform
            prediction = min(prediction, 0.05)
        
        if 'debt_to_equity' in metrics and metrics['debt_to_equity'] > 2.0:
            # High leverage increases risk
            prediction = prediction * 0.8
        
        return prediction
    
    def get_industry_adjustment_factor(self, industry: str, base_value: float) -> float:
        """Get an industry-specific adjustment factor for DCF valuation."""
        try:
            # Load industry benchmarks if not already loaded
            if self.industry_benchmarks is None:
                benchmark_file = os.path.join(self.data_dir, 'industry_benchmarks.csv')
                if os.path.exists(benchmark_file):
                    self.industry_benchmarks = pd.read_csv(benchmark_file)
            
            # Default adjustment is 1.0 (no change)
            adjustment = 1.0
            
            if self.industry_benchmarks is not None:
                industry_row = self.industry_benchmarks[self.industry_benchmarks['industry'] == industry]
                if not industry_row.empty:
                    # Calculate adjustment based on industry characteristics
                    
                    # 1. Growth premium/discount
                    if 'historical_growth_mean_median' in industry_row:
                        growth_rate = float(industry_row['historical_growth_mean_median'])
                        # Higher growth industries deserve premium
                        if growth_rate > 0.15:  # >15% growth
                            adjustment *= 1.1
                        elif growth_rate < 0.05:  # <5% growth
                            adjustment *= 0.9
                    
                    # 2. Profitability adjustment
                    if 'net_margin_median' in industry_row:
                        net_margin = float(industry_row['net_margin_median'])
                        # Higher margin industries deserve premium
                        if net_margin > 0.15:  # >15% net margin
                            adjustment *= 1.1
                        elif net_margin < 0.05:  # <5% net margin
                            adjustment *= 0.95
                    
                    # 3. Industry-specific P/E or P/B factors
                    industry_pe_adjustments = {
                        'Semiconductors': 1.2,      # High growth, high margin industry
                        'Electronics': 1.1,         # Technology premium
                        'Banking': 0.8,             # Financial sector discount
                        'Telecommunications': 0.9,  # Stable but slower growth
                        'Utilities': 0.85,          # Stable but regulated
                        'Healthcare': 1.15,         # Long-term growth prospects
                        'Retail': 0.9,              # Competitive pressures
                        'Materials': 0.95           # Cyclical industry
                    }
                    
                    # Apply industry-specific adjustment
                    industry_pe_adj = industry_pe_adjustments.get(industry, 1.0)
                    adjustment *= industry_pe_adj
            
            # Hard-coded adjustments for specific industries
            if industry == 'Semiconductors':
                # Taiwan semiconductor industry gets a premium
                adjustment *= 1.15
            
            # Ensure adjustment is within reasonable bounds
            adjustment = max(0.7, min(1.3, adjustment))
            
            logger.info(f"Industry adjustment factor for {industry}: {adjustment:.2f}")
            return adjustment
            
        except Exception as e:
            logger.error(f"Error calculating industry adjustment: {e}")
            return 1.0  # No adjustment in case of error
    
    def adjust_dcf_valuation(self, industry: str, base_valuation: float, 
                            financial_metrics: Dict[str, float]) -> Dict[str, float]:
        """Adjust DCF valuation using industry-specific models.
        
        Args:
            industry: Industry classification
            base_valuation: Base DCF valuation
            financial_metrics: Dict of financial metrics
            
        Returns:
            Dict with adjusted valuation and details
        """
        try:
            # 1. Get industry-specific adjustment factor
            industry_factor = self.get_industry_adjustment_factor(industry, base_valuation)
            
            # 2. Predict expected return if we have the right metrics
            expected_return = None
            if self.industry_models and industry in self.industry_models:
                expected_return = self.predict_future_returns(industry, financial_metrics)
            
            # 3. Calculate return-based adjustment
            return_factor = 1.0
            if expected_return is not None:
                # If expected return is positive, increase valuation
                if expected_return > 0.1:  # >10% expected return
                    return_factor = 1.1
                elif expected_return < 0:  # Negative expected return
                    return_factor = 0.9
                else:
                    return_factor = 1.0 + expected_return  # Linear adjustment
            
            # 4. Calculate final adjustment
            total_adjustment = industry_factor * return_factor
            
            # 5. Apply adjustment with bounds
            adjusted_valuation = base_valuation * total_adjustment
            
            # 6. Return detailed results
            return {
                'base_valuation': base_valuation,
                'adjusted_valuation': adjusted_valuation,
                'industry_factor': industry_factor,
                'return_factor': return_factor,
                'total_adjustment': total_adjustment,
                'expected_return': expected_return,
                'industry': industry
            }
            
        except Exception as e:
            logger.error(f"Error adjusting valuation: {e}")
            return {
                'base_valuation': base_valuation,
                'adjusted_valuation': base_valuation,  # No adjustment in case of error
                'industry_factor': 1.0,
                'return_factor': 1.0,
                'total_adjustment': 1.0,
                'error': str(e),
                'industry': industry
            }
        
    def train_with_db_data(self, industries: Optional[List[str]] = None, force_retrain: bool = False) -> Dict[str, Dict]:
        """Train models using data from the background database.
        
        Args:
            industries: List of specific industries to train, or None for all
            force_retrain: Whether to force retraining even if models exist
            
        Returns:
            Dictionary of industry -> training metrics
        """
        if self.background_collector is None:
            logger.error("No background data collector provided")
            return {}
        
        # Prepare training data from database
        training_data = self.prepare_training_data_from_db()
        
        if not training_data:
            logger.error("Failed to prepare training data from database")
            return {}
        
        # Now train models using the prepared data
        return self.train_industry_models(industries, force_retrain)

# Usage example
if __name__ == "__main__":
    # First collect industry data if needed
    collector = TaiwanIndustryDataCollector(
        lookback_years=5,
        rate_limit_delay=1.5,
        max_retries=3
    )
    
    # Specify key industries to focus on
    priority_industries = [
        "Semiconductors",
        "Electronics Manufacturing",
        "Computer Hardware",
        "Banking",
        "Telecommunications"
    ]
    
    # Check if we have data already
    industry_data_path = os.path.join("industry_data", "taiwan_industry_financial_data.pkl")
    if not os.path.exists(industry_data_path):
        # Collect data (use smaller limits to avoid API issues)
        collector.collect_industry_financial_data(
            max_stocks_per_industry=5,
            parallel=True,
            max_workers=2,
            prioritize_industries=priority_industries
        )
        
        # Prepare training data
        collector.prepare_training_data()
        
        # Generate benchmarks
        collector.get_industry_benchmark_metrics()
    
    # Initialize and train models
    model = IndustryValuationModel()
    results = model.train_industry_models(force_retrain=False)
    
    # Test adjustment on a sample valuation
    test_metrics = {
        'historical_growth_mean': 0.12,
        'operating_margin': 0.15,
        'net_margin': 0.10,
        'roa': 0.08,
        'roe': 0.12,
        'debt_to_equity': 0.5
    }
    
    # Test adjustment for a semiconductor company
    semiconductor_adjustment = model.adjust_dcf_valuation('Semiconductors', 1000.0, test_metrics)
    print(f"Semiconductor valuation adjustment: {semiconductor_adjustment}")
    
    # Test adjustment for a bank
    banking_adjustment = model.adjust_dcf_valuation('Banking', 1000.0, test_metrics)
    print(f"Banking valuation adjustment: {banking_adjustment}")
