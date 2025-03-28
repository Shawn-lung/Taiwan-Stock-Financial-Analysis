import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
import yfinance as yf
import logging
from data_fetcher import FinancialDataFetcher
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import zscore

logger = logging.getLogger(__name__)

class GrowthPredictor:
    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.data_fetcher = FinancialDataFetcher()
        self.stock = yf.Ticker(stock_code)
        
        # Modify model parameters for better stability
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=200,
                max_depth=3,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                max_features='sqrt'
            ))
        ])
        self.feature_selector = SelectKBest(f_regression, k=5)
        self.available_years = 0
        self.training_samples = 0
        self.feature_importance = None
        self.cv_scores = None
        self.selected_features = None
        self.growth_pipeline = self._create_pipeline()
        self.capex_pipeline = self._create_pipeline()
        self.wc_pipeline = self._create_pipeline()
        self.depr_pipeline = self._create_pipeline()
        self.tax_pipeline = self._create_pipeline()
        
    def _create_pipeline(self):
        """Create a new pipeline for each factor."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=200,
                max_depth=3,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                max_features='sqrt'
            ))
        ])

    def calculate_financial_ratios(self, financials, balance_sheet):
        """Enhanced ratio calculation with smoothing."""
        try:
            revenue_metric = next((m for m in ['Total Revenue', 'Revenue'] if m in financials.index), None)
            if not revenue_metric:
                return {}

            ratios = {}
            dates = sorted(financials.columns)
            
            # Calculate rolling metrics
            for i, date in enumerate(dates):
                try:
                    revenue = float(financials.loc[revenue_metric, date])
                    if revenue <= 0:
                        continue

                    # Store raw values
                    ratios.setdefault('revenue', []).append(revenue)
                    
                    # Operating metrics
                    if 'Operating Income' in financials.index:
                        op_income = float(financials.loc['Operating Income', date])
                        ratios.setdefault('operating_income', []).append(op_income)
                        ratios.setdefault('operating_margin', []).append(op_income / revenue)
                    
                    # Growth metrics (with smoothing)
                    if i > 0:
                        growth = (revenue - ratios['revenue'][-2]) / ratios['revenue'][-2]
                        ratios.setdefault('revenue_growth', []).append(growth)
                        
                        # Exponential smoothing for growth
                        if len(ratios.get('revenue_growth', [])) > 1:
                            alpha = 0.7  # Smoothing factor
                            prev_smoothed = ratios['smooth_growth'][-1]
                            smoothed_growth = alpha * growth + (1 - alpha) * prev_smoothed
                            ratios.setdefault('smooth_growth', []).append(smoothed_growth)
                        else:
                            ratios.setdefault('smooth_growth', []).append(growth)
                    
                    # Efficiency metrics
                    if 'Total Assets' in balance_sheet.index:
                        assets = float(balance_sheet.loc['Total Assets', date])
                        if assets > 0:
                            ratios.setdefault('asset_turnover', []).append(revenue / assets)
                            ratios.setdefault('asset_growth', []).append(
                                (assets - float(balance_sheet.loc['Total Assets', dates[i-1]])) / float(balance_sheet.loc['Total Assets', dates[i-1]])
                                if i > 0 else 0
                            )

                except Exception as e:
                    logger.debug(f"Error calculating ratios for {date}: {e}")
                    continue

            # Calculate additional features
            if len(ratios.get('revenue_growth', [])) > 3:
                ratios['growth_volatility'] = pd.Series(ratios['revenue_growth']).rolling(4).std().tolist()
                ratios['growth_momentum'] = pd.Series(ratios['revenue_growth']).rolling(4).mean().diff().tolist()
                ratios['growth_acceleration'] = pd.Series(ratios['growth_momentum']).diff().tolist()

            return ratios
        except Exception as e:
            logger.error(f"Error in ratio calculation: {e}")
            return {}

    def prepare_features(self):
        """Prepare features with enhanced engineering."""
        try:
            financial_data = self.data_fetcher.get_financial_data(self.stock_code)
            if not financial_data:
                return None, None

            financials = financial_data['income_statement']
            revenue = financials.loc['Total Revenue']
            op_income = financials.loc['Operating Income']
            
            data = []
            dates = sorted(financials.columns)
            lookback = 3  # Use 3 years lookback
            
            for i in range(lookback, len(dates)-1):
                try:
                    features = {}
                    
                    # Calculate trailing metrics
                    current_revenue = revenue[dates[i]]
                    trailing_revenues = [revenue[dates[j]] for j in range(i-lookback+1, i+1)]
                    trailing_op_income = [op_income[dates[j]] for j in range(i-lookback+1, i+1)]
                    
                    # Growth metrics
                    yearly_growth = [(trailing_revenues[j+1] - trailing_revenues[j]) / trailing_revenues[j] 
                                   for j in range(len(trailing_revenues)-1)]
                    
                    # Operating metrics
                    margins = [op/rev for op, rev in zip(trailing_op_income, trailing_revenues)]
                    
                    # Feature engineering
                    features['recent_growth'] = yearly_growth[-1]
                    features['avg_growth'] = np.mean(yearly_growth)
                    features['growth_trend'] = yearly_growth[-1] - yearly_growth[0]
                    features['growth_volatility'] = np.std(yearly_growth)
                    
                    features['current_margin'] = margins[-1]
                    features['margin_trend'] = margins[-1] - margins[0]
                    features['margin_volatility'] = np.std(margins)
                    
                    features['size'] = np.log(current_revenue)
                    features['size_momentum'] = np.log(current_revenue / trailing_revenues[-2])
                    
                    # Target: Next year's growth
                    next_revenue = revenue[dates[i+1]]
                    target = (next_revenue - current_revenue) / current_revenue
                    
                    # Include point if values are reasonable
                    if all(abs(x) < 2 for x in yearly_growth + [target]):
                        data.append((features, target))
                        
                except Exception as e:
                    logger.debug(f"Error processing point {i}: {e}")
                    continue

            if not data:
                return None, None

            X = pd.DataFrame([d[0] for d in data])
            y = np.array([d[1] for d in data])

            # Remove extreme outliers only
            z_scores = np.abs(zscore(y))
            mask = z_scores < 3
            X = X[mask]
            y = y[mask]

            self.training_samples = len(X)
            logger.info(f"Generated {self.training_samples} valid training samples")
            logger.info(f"Features: {list(X.columns)}")
            logger.info(f"Historical growth rates: {[f'{g:.1%}' for g in y]}")
            
            return X, y

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None, None

    def prepare_factor_features(self, data_type='growth'):
        """Prepare features for different factor predictions."""
        try:
            financial_data = self.data_fetcher.get_financial_data(self.stock_code)
            if not financial_data:
                return None, None

            financials = financial_data['income_statement']
            balance = financial_data['balance_sheet']
            cashflow = financial_data['cash_flow']
            
            # Calculate factor-specific features
            if data_type == 'growth':
                return self._prepare_growth_features(financials)
            elif data_type == 'capex':
                return self._prepare_capex_features(financials, cashflow)
            elif data_type == 'wc':
                return self._prepare_wc_features(financials, balance)
            elif data_type == 'depr':
                return self._prepare_depr_features(financials, cashflow)
            elif data_type == 'tax':
                return self._prepare_tax_features(financials)
            
        except Exception as e:
            logger.error(f"Error preparing {data_type} features: {e}")
            return None, None

    def _prepare_growth_features(self, financials):
        """Prepare features for revenue growth prediction."""
        try:
            revenue = financials.loc['Total Revenue']
            op_income = financials.loc['Operating Income']
            dates = sorted(financials.columns)
            data = []
            lookback = 3

            for i in range(lookback, len(dates)-1):
                try:
                    features = {}
                    current_rev = revenue[dates[i]]
                    prev_revenues = [revenue[dates[j]] for j in range(i-lookback, i+1)]
                    growth_rates = [(prev_revenues[j+1] - prev_revenues[j])/prev_revenues[j] for j in range(len(prev_revenues)-1)]
                    
                    features['recent_growth'] = growth_rates[-1]
                    features['avg_growth'] = np.mean(growth_rates)
                    features['growth_volatility'] = np.std(growth_rates)
                    features['size'] = np.log(current_rev)
                    features['margin'] = op_income[dates[i]] / current_rev
                    
                    target = (revenue[dates[i+1]] - current_rev) / current_rev
                    if abs(target) < 1.0:  # Filter extreme values
                        data.append((features, target))
                except Exception as e:
                    continue

            if not data:
                return None, None
            return pd.DataFrame([d[0] for d in data]), np.array([d[1] for d in data])
        except Exception as e:
            logger.error(f"Error preparing growth features: {e}")
            return None, None

    def _normalize_metric_name(self, df, metrics):
        """Find the first matching metric from a list of possible names."""
        for metric in metrics:
            if metric in df.index:
                logger.info(f"Found metric: {metric}")
                return metric
        return None

    def _prepare_capex_features(self, financials, cashflow):
        """Prepare features for CAPEX prediction with enhanced metric detection."""
        try:
            revenue = financials.loc['Total Revenue']
            
            # Try different possible CAPEX metric names with logging
            capex_metrics = [
                'Capital Expenditure',
                'PropertyAndPlantAndEquipment',
                'AcquisitionOfPropertyPlantAndEquipment',
                'Purchase Of Property Plant And Equipment',
                'NetCapitalExpenditure'
            ]
            
            logger.info(f"Available cashflow metrics: {cashflow.index.tolist()}")
            capex_metric = self._normalize_metric_name(cashflow, capex_metrics)
            
            if not capex_metric:
                logger.error("Could not find CAPEX metric in available metrics")
                return None, None
                
            capex = cashflow.loc[capex_metric]
            dates = sorted(set(financials.columns) & set(cashflow.columns))
            data = []
            lookback = 2

            for i in range(lookback, len(dates)-1):
                try:
                    features = {}
                    current_rev = revenue[dates[i]]
                    current_capex = abs(capex[dates[i]])
                    
                    features['capex_to_revenue'] = current_capex / current_rev
                    features['revenue_growth'] = (current_rev - revenue[dates[i-1]]) / revenue[dates[i-1]]
                    features['size'] = np.log(current_rev)
                    
                    next_capex = abs(capex[dates[i+1]])
                    target = (next_capex/revenue[dates[i+1]]) / (current_capex/current_rev) - 1
                    
                    if abs(target) < 0.5:
                        data.append((features, target))
                except Exception as e:
                    logger.debug(f"Error processing CAPEX point {i}: {e}")
                    continue

            if not data:
                return None, None
                
            return pd.DataFrame([d[0] for d in data]), np.array([d[1] for d in data])
            
        except Exception as e:
            logger.error(f"Error preparing capex features: {e}")
            return None, None

    def _prepare_wc_features(self, financials, balance):
        """Prepare features for working capital prediction."""
        try:
            revenue = financials.loc['Total Revenue']
            
            # Try different possible asset/liability metric names
            asset_metrics = [
                'Current Assets',
                'CurrentAssets',
                'Total Current Assets'
            ]
            liability_metrics = [
                'Current Liabilities',
                'CurrentLiabilities',
                'Total Current Liabilities'
            ]
            
            asset_metric = self._normalize_metric_name(balance, asset_metrics)
            liability_metric = self._normalize_metric_name(balance, liability_metrics)
            
            if not asset_metric or not liability_metric:
                logger.error("Could not find working capital metrics")
                return None, None
                
            current_assets = balance.loc[asset_metric]
            current_liab = balance.loc[liability_metric]
            dates = sorted(set(financials.columns) & set(balance.columns))
            data = []
            lookback = 2

            for i in range(lookback, len(dates)-1):
                try:
                    features = {}
                    current_rev = revenue[dates[i]]
                    wc = current_assets[dates[i]] - current_liab[dates[i]]
                    
                    features['wc_to_revenue'] = wc / current_rev
                    features['revenue_growth'] = (current_rev - revenue[dates[i-1]]) / revenue[dates[i-1]]
                    features['size'] = np.log(current_rev)
                    
                    next_wc = current_assets[dates[i+1]] - current_liab[dates[i+1]]
                    target = (next_wc/revenue[dates[i+1]]) / (wc/current_rev) - 1
                    if abs(target) < 0.5:
                        data.append((features, target))
                except Exception:
                    continue

            if not data:
                return None, None
            return pd.DataFrame([d[0] for d in data]), np.array([d[1] for d in data])
        except Exception as e:
            logger.error(f"Error preparing wc features: {e}")
            return None, None

    def _prepare_depr_features(self, financials, cashflow):
        """Prepare features for depreciation prediction."""
        try:
            revenue = financials.loc['Total Revenue']
            depr = cashflow.loc['Depreciation']
            dates = sorted(set(financials.columns) & set(cashflow.columns))
            data = []
            lookback = 2

            for i in range(lookback, len(dates)-1):
                try:
                    features = {}
                    current_rev = revenue[dates[i]]
                    current_depr = abs(depr[dates[i]])
                    
                    features['depr_to_revenue'] = current_depr / current_rev
                    features['revenue_growth'] = (current_rev - revenue[dates[i-1]]) / revenue[dates[i-1]]
                    features['size'] = np.log(current_rev)
                    
                    next_depr = abs(depr[dates[i+1]])
                    target = (next_depr/revenue[dates[i+1]]) / (current_depr/current_rev) - 1
                    if abs(target) < 0.5:
                        data.append((features, target))
                except Exception:
                    continue

            if not data:
                return None, None
            return pd.DataFrame([d[0] for d in data]), np.array([d[1] for d in data])
        except Exception as e:
            logger.error(f"Error preparing depr features: {e}")
            return None, None

    def _prepare_tax_features(self, financials):
        """Prepare features for tax rate prediction."""
        try:
            revenue = financials.loc['Total Revenue']
            op_income = financials.loc['Operating Income']
            tax = financials.loc['Tax Provision']
            dates = sorted(financials.columns)
            data = []
            lookback = 2

            for i in range(lookback, len(dates)-1):
                try:
                    features = {}
                    current_rev = revenue[dates[i]]
                    current_tax = tax[dates[i]]
                    current_op = op_income[dates[i]]
                    
                    features['tax_to_opincome'] = current_tax / current_op if current_op > 0 else 0
                    features['margin'] = current_op / current_rev
                    features['size'] = np.log(current_rev)
                    
                    next_tax = tax[dates[i+1]]
                    next_op = op_income[dates[i+1]]
                    target = (next_tax/next_op if next_op > 0 else 0) - (current_tax/current_op if current_op > 0 else 0)
                    if abs(target) < 0.2:  # Tax rates shouldn't change too dramatically
                        data.append((features, target))
                except Exception:
                    continue

            if not data:
                return None, None
            return pd.DataFrame([d[0] for d in data]), np.array([d[1] for d in data])
        except Exception as e:
            logger.error(f"Error preparing tax features: {e}")
            return None, None

    def train(self):
        """Train model and report performance metrics."""
        try:
            X, y = self.prepare_features()
            if X is None or y is None or len(X) < 4:
                return False

            # Store column names as selected features
            self.selected_features = X.columns.tolist()

            # Use recent data for validation
            train_size = int(len(X) * 0.8)
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_test = X[train_size:]
            y_test = y[train_size:]
            
            # Train model
            self.pipeline.fit(X_train, y_train)
            
            # Calculate performance metrics
            train_score = self.pipeline.score(X_train, y_train)
            test_predictions = self.pipeline.predict(X_test)
            prediction_errors = np.abs(test_predictions - y_test)
            mean_error = np.mean(prediction_errors)
            
            logger.info(f"Train R² score: {train_score:.4f}")
            logger.info(f"Mean absolute prediction error: {mean_error:.1%}")
            
            # Store feature importance
            importances = self.pipeline.named_steps['regressor'].feature_importances_
            self.feature_importance = pd.DataFrame({
                'feature': self.selected_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logger.info("Feature importance:")
            for idx, row in self.feature_importance.iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.3f}")
            
            self.cv_scores = [train_score, 1 - mean_error]
            return True

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def predict_growth(self):
        """Predict next year's growth rate."""
        try:
            X, _ = self.prepare_features()
            if X is None or X.empty:
                logger.error("No features available for prediction")
                return None
            
            # Use all features since they're already selected in prepare_features
            X_current = X.iloc[-1:]
            X_scaled = self.pipeline.named_steps['scaler'].transform(X_current)
            prediction = self.pipeline.named_steps['regressor'].predict(X_scaled)[0]
            
            # Bound prediction between -20% and +50%
            bounded_prediction = max(min(prediction, 0.5), -0.2)
            
            logger.info(f"Raw growth prediction: {prediction:.2%}")
            logger.info(f"Bounded growth prediction: {bounded_prediction:.2%}")
            
            return bounded_prediction
            
        except Exception as e:
            logger.error(f"Error predicting growth: {e}")
            return None

    def predict_all_factors(self, forecast_years=5, terminal_growth=0.025):
        """Predict factors for user-specified years with individual ML predictions."""
        try:
            predictions = {
                'growth_rates': [],
                'capex_factors': [],
                'wc_factors': [],
                'depr_factors': [],
                'tax_factors': []
            }
            
            X_growth, y_growth = self.prepare_factor_features('growth')
            if X_growth is not None and len(X_growth) >= 4:
                self.growth_pipeline.fit(X_growth, y_growth)
                last_features = X_growth.iloc[-1:].copy()
                
                for year in range(forecast_years):
                    # Make prediction for current year
                    current_pred = float(self.growth_pipeline.predict(last_features)[0])
                    
                    # Add small random variation
                    variation = np.random.normal(0, 0.005)  # 0.5% variation
                    current_pred *= (1 + variation)
                    
                    # Bound between terminal growth and max growth
                    current_pred = max(min(current_pred, 0.5), terminal_growth)
                    predictions['growth_rates'].append(current_pred)
                    
                    # Update features for next prediction
                    # This simulates how recent growth affects future predictions
                    if 'recent_growth' in last_features:
                        last_features['recent_growth'] = current_pred
                    if 'avg_growth' in last_features:
                        prev_growths = predictions['growth_rates'][-3:] if len(predictions['growth_rates']) >= 3 else [current_pred]
                        last_features['avg_growth'] = np.mean(prev_growths)
                
                # Predict other factors
                for factor_type, pipeline in [
                    ('capex', self.capex_pipeline),
                    ('wc', self.wc_pipeline),
                    ('depr', self.depr_pipeline),
                    ('tax', self.tax_pipeline)
                ]:
                    X, y = self.prepare_factor_features(factor_type)
                    if X is not None and len(X) >= 4:
                        pipeline.fit(X, y)
                        last_feat = X.iloc[-1:].copy()
                        
                        for year in range(forecast_years):
                            pred = pipeline.predict(last_feat)[0]
                            variation = np.random.normal(0, 0.002)
                            final_pred = pred * (1 + variation)
                            
                            if factor_type in ['capex', 'wc']:
                                final_pred = max(min(final_pred, 0.3), -0.3)
                            else:  # tax, depr
                                final_pred = max(min(final_pred, 0.2), -0.2)
                                
                            predictions[f'{factor_type}_factors'].append(float(final_pred))
                            
                            # Update features for next prediction if needed
                            if 'revenue_growth' in last_feat:
                                last_feat['revenue_growth'] = predictions['growth_rates'][year]
                    else:
                        predictions[f'{factor_type}_factors'].extend([0.0] * forecast_years)
                
                logger.info(f"ML Predictions for {forecast_years} years:")
                for factor, values in predictions.items():
                    logger.info(f"{factor}: {[f'{x:.1%}' for x in values]}")
                
                predictions['forecast_years'] = forecast_years
                return predictions
            
            return None

        except Exception as e:
            logger.error(f"Error in predictions: {e}")
            return None

    def _decay_factor(self, pipeline, X, year, decay_rate):
        """Calculate decayed factor with small random variation."""
        try:
            if X is not None and len(X) >= 4:
                initial = pipeline.predict(X.iloc[-1:])[0]
                decayed = initial * (decay_rate ** year)
                variation = np.random.normal(0, 0.01)  # 1% variation
                return float(decayed * (1 + variation))
            return 0.0
        except Exception:
            return 0.0

    def get_prediction_confidence(self) -> float:
        """Calculate confidence score based on data quality and model performance."""
        try:
            # Start with base confidence from number of training samples
            sample_confidence = min(self.training_samples / 10, 1.0)  # Max confidence at 10+ samples
            
            # Add model performance confidence if available
            if self.cv_scores is not None and not np.isnan(self.cv_scores).any():
                model_confidence = max(np.mean(self.cv_scores), 0)  # R² score can be negative
            else:
                model_confidence = 0.3  # Default confidence if no CV scores
                
            # Calculate final confidence
            confidence = (sample_confidence + model_confidence) / 2
            
            # Bound between 0.3 and 0.9
            return min(max(confidence, 0.3), 0.9)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.3
