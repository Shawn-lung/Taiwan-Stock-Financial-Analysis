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
        """Prepare features for revenue ratio prediction."""
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
                    
                    # Calculate revenue ratios instead of growth rates
                    ratios = [prev_revenues[j+1]/prev_revenues[j] for j in range(len(prev_revenues)-1)]
                    
                    features['recent_ratio'] = ratios[-1]
                    features['avg_ratio'] = np.mean(ratios)
                    features['ratio_volatility'] = np.std(ratios)
                    features['size'] = np.log(current_rev)
                    features['margin'] = op_income[dates[i]] / current_rev
                    
                    # Target is next year's revenue ratio
                    next_revenue = revenue[dates[i+1]]
                    target = next_revenue / current_rev
                    
                    # Filter out extreme ratios
                    if 0.5 < target < 2.0 and all(0.5 < r < 2.0 for r in ratios):
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
        """Prepare features for CAPEX/Revenue ratio prediction."""
        try:
            revenue = financials.loc['Total Revenue']
            capex_metrics = [
                'Capital Expenditure',
                'PropertyAndPlantAndEquipment',
                'AcquisitionOfPropertyPlantAndEquipment',
                'Purchase Of Property Plant And Equipment',
                'NetCapitalExpenditure'
            ]
            
            # Log available metrics for debugging
            logger.info(f"Available cashflow metrics: {cashflow.index.tolist()}")
            capex_metric = self._normalize_metric_name(cashflow, capex_metrics)
            
            if not capex_metric:
                logger.error("Could not find CAPEX metric")
                return None, None
                
            capex = cashflow.loc[capex_metric]
            dates = sorted(set(financials.columns) & set(cashflow.columns))
            logger.info(f"Available years for CAPEX data: {len(dates)}")
            
            data = []
            lookback = 2

            for i in range(lookback, len(dates)-1):
                try:
                    features = {}
                    current_rev = revenue[dates[i]]
                    current_capex = abs(capex[dates[i]])
                    current_ratio = current_capex / current_rev
                    
                    # Log the training data points
                    logger.info(f"CAPEX data point {i}: Year={dates[i]}, Ratio={current_ratio:.2%}")
                    
                    features['current_capex_ratio'] = current_ratio
                    features['size'] = np.log(current_rev)
                    
                    if i > 0:
                        prev_ratio = abs(capex[dates[i-1]]) / revenue[dates[i-1]]
                        features['prev_capex_ratio'] = prev_ratio
                        features['ratio_trend'] = current_ratio - prev_ratio
                    
                    next_capex = abs(capex[dates[i+1]])
                    next_rev = revenue[dates[i+1]]
                    target = next_capex / next_rev
                    
                    if 0.01 <= target <= 0.4:
                        data.append((features, target))
                        
                except Exception as e:
                    logger.debug(f"Error processing CAPEX point {i}: {e}")
                    continue

            if not data:
                logger.warning("No valid CAPEX training data")
                return None, None
                
            logger.info(f"Generated {len(data)} CAPEX training samples")
            return pd.DataFrame([d[0] for d in data]), np.array([d[1] for d in data])
            
        except Exception as e:
            logger.error(f"Error preparing capex features: {e}")
            return None, None

    def _prepare_wc_features(self, financials, balance):
        """Prepare features for working capital to revenue ratio prediction."""
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
            logger.info(f"Available years for WC data: {len(dates)}")
            
            data = []
            lookback = 2

            for i in range(lookback, len(dates)-1):
                try:
                    features = {}
                    current_rev = revenue[dates[i]]
                    wc = current_assets[dates[i]] - current_liab[dates[i]]
                    current_ratio = wc / current_rev
                    
                    # Log the training data point
                    logger.info(f"WC data point {i}: Year={dates[i]}, Ratio={current_ratio:.2%}")
                    
                    features['wc_to_revenue'] = current_ratio
                    features['size'] = np.log(current_rev)
                    
                    if i > 0:
                        prev_wc = current_assets[dates[i-1]] - current_liab[dates[i-1]]
                        prev_ratio = prev_wc / revenue[dates[i-1]]
                        features['prev_wc_ratio'] = prev_ratio
                        features['ratio_trend'] = current_ratio - prev_ratio
                    
                    next_wc = current_assets[dates[i+1]] - current_liab[dates[i+1]]
                    next_rev = revenue[dates[i+1]]
                    # Target is the WC/Revenue ratio for next year
                    target = next_wc / next_rev
                    
                    # Filter reasonable values
                    if -0.3 <= target <= 0.5:
                        data.append((features, target))
                        
                except Exception as e:
                    logger.debug(f"Error processing WC point {i}: {e}")
                    continue

            if not data:
                logger.warning("No valid WC training data")
                return None, None
                
            logger.info(f"Generated {len(data)} WC training samples")
            return pd.DataFrame([d[0] for d in data]), np.array([d[1] for d in data])
            
        except Exception as e:
            logger.error(f"Error preparing WC features: {e}")
            return None, None

    def _prepare_depr_features(self, financials, cashflow):
        """Prepare features for depreciation to revenue ratio prediction."""
        try:
            revenue = financials.loc['Total Revenue']
            
            # Try different possible depreciation metric names
            depr_metrics = [
                'Depreciation',
                'Depreciation & Amortization',
                'DepreciationAndAmortization'
            ]
            
            logger.info(f"Available cashflow metrics: {cashflow.index.tolist()}")
            depr_metric = self._normalize_metric_name(cashflow, depr_metrics)
            
            if not depr_metric:
                logger.error("Could not find Depreciation metric")
                return None, None
                
            depr = cashflow.loc[depr_metric]
            dates = sorted(set(financials.columns) & set(cashflow.columns))
            logger.info(f"Available years for Depreciation data: {len(dates)}")
            
            data = []
            lookback = 2

            for i in range(lookback, len(dates)-1):
                try:
                    features = {}
                    current_rev = revenue[dates[i]]
                    current_depr = abs(depr[dates[i]])
                    current_ratio = current_depr / current_rev
                    
                    logger.info(f"Depr data point {i}: Year={dates[i]}, Ratio={current_ratio:.2%}")
                    
                    features['depr_to_revenue'] = current_ratio
                    features['size'] = np.log(current_rev)
                    
                    if i > 0:
                        prev_ratio = abs(depr[dates[i-1]]) / revenue[dates[i-1]]
                        features['prev_depr_ratio'] = prev_ratio
                        features['ratio_trend'] = current_ratio - prev_ratio
                    
                    next_depr = abs(depr[dates[i+1]])
                    next_rev = revenue[dates[i+1]]
                    target = next_depr / next_rev
                    
                    if 0.01 <= target <= 0.2:
                        data.append((features, target))
                        
                except Exception as e:
                    logger.debug(f"Error processing depreciation point {i}: {e}")
                    continue

            if not data:
                logger.warning("No valid depreciation training data")
                return None, None
                
            logger.info(f"Generated {len(data)} depreciation training samples")
            return pd.DataFrame([d[0] for d in data]), np.array([d[1] for d in data])
            
        except Exception as e:
            logger.error(f"Error preparing depreciation features: {e}")
            return None, None

    def _prepare_tax_features(self, financials):
        """Prepare features for tax rate prediction."""
        try:
            revenue = financials.loc['Total Revenue']
            
            # Try different possible tax and income metric names
            tax_metrics = ['Tax Provision', 'Income Tax Expense']
            income_metrics = ['Operating Income', 'Pretax Income', 'Income Before Tax']
            
            logger.info(f"Available income metrics: {financials.index.tolist()}")
            tax_metric = self._normalize_metric_name(financials, tax_metrics)
            income_metric = self._normalize_metric_name(financials, income_metrics)
            
            if not tax_metric or not income_metric:
                logger.error("Could not find tax or income metrics")
                return None, None
                
            tax = financials.loc[tax_metric]
            op_income = financials.loc[income_metric]
            dates = sorted(financials.columns)
            logger.info(f"Available years for Tax data: {len(dates)}")
            
            data = []
            lookback = 2

            for i in range(lookback, len(dates)-1):
                try:
                    features = {}
                    current_rev = revenue[dates[i]]
                    current_tax = abs(tax[dates[i]])
                    current_op = abs(op_income[dates[i]])
                    
                    if current_op <= 0:
                        continue
                        
                    current_rate = current_tax / current_op
                    
                    logger.info(f"Tax data point {i}: Year={dates[i]}, Rate={current_rate:.2%}")
                    
                    features['tax_rate'] = current_rate
                    features['size'] = np.log(current_rev)
                    features['margin'] = current_op / current_rev
                    
                    if i > 0:
                        prev_op = abs(op_income[dates[i-1]])
                        prev_tax = abs(tax[dates[i-1]])
                        if prev_op > 0:
                            prev_rate = prev_tax / prev_op
                            features['prev_tax_rate'] = prev_rate
                            features['rate_trend'] = current_rate - prev_rate
                    
                    next_tax = abs(tax[dates[i+1]])
                    next_op = abs(op_income[dates[i+1]])
                    
                    if next_op <= 0:
                        continue
                        
                    target = next_tax / next_op
                    
                    if 0.1 <= target <= 0.4:
                        data.append((features, target))
                        
                except Exception as e:
                    logger.debug(f"Error processing tax point {i}: {e}")
                    continue

            if not data:
                logger.warning("No valid tax training data")
                return None, None
                
            logger.info(f"Generated {len(data)} tax training samples")
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
        """Predict factors with revenue growth and ratio-based predictions."""
        try:
            predictions = {
                'growth_rates': [],
                'capex_factors': [],
                'wc_factors': [],
                'depr_factors': [],
                'tax_factors': []
            }
            
            # First predict revenue growth rates
            X_growth, y_growth = self.prepare_factor_features('growth')
            if X_growth is not None and len(X_growth) >= 3:
                self.growth_pipeline.fit(X_growth, y_growth)
                last_features = X_growth.iloc[-1:].copy()
                
                historical_ratio = float(y_growth.mean())
                historical_std = float(y_growth.std())
                logger.info(f"Growth - Historical mean: {historical_ratio:.2%}, std: {historical_std:.2%}")
                
                # For high-growth companies, adjust bounds to allow higher growth persistence
                max_possible_growth = min(historical_ratio + 1.5*historical_std, 1.5)  # Increased max growth cap
                
                # Predict first year directly with the model
                ratio_pred = float(self.growth_pipeline.predict(last_features)[0])
                small_variation = np.random.normal(0, 0.01)  
                ratio_pred = ratio_pred * (1 + small_variation)
                
                # Calculate year-specific bounds
                min_ratio = max(0.95, historical_ratio - 2*historical_std)
                max_ratio = min(max_possible_growth, 1.5)  # Increased max ratio
                
                # Apply bounds
                ratio_pred = max(min(ratio_pred, max_ratio), min_ratio)
                first_year_growth = ratio_pred - 1.0
                predictions['growth_rates'].append(float(first_year_growth))
                
                # Update features for next iteration
                if 'recent_ratio' in last_features:
                    last_features['recent_ratio'] = ratio_pred
                
                # For remaining years, create a more gradual curve towards long-term rate
                if forecast_years > 1:
                    # Start with the first year growth and decline more gradually
                    start_growth = first_year_growth
                    end_growth = max(terminal_growth * 1.5, terminal_growth + 0.02)
                    
                    # Use a more stretched curve for companies with high historical growth
                    if historical_ratio > 0.5:  # For companies with >50% historical growth
                        # Create a slower decay curve
                        for i in range(forecast_years-1):
                            # Use a flatter decay function for high-growth companies
                            progress = ((i + 1) / (forecast_years - 1)) ** 0.3  # Slower decay (was 0.5)
                            
                            # Calculate intermediate growth with slower decay
                            current_growth = start_growth - progress * (start_growth - end_growth)
                            
                            # Add small variation
                            variation = np.random.normal(0, 0.005)
                            adjusted_growth = current_growth * (1 + variation)
                            
                            predictions['growth_rates'].append(float(adjusted_growth))
                    else:
                        # Use standard decay for normal growth companies
                        for i in range(forecast_years-1):
                            progress = ((i + 1) / (forecast_years - 1)) ** 0.5
                            current_growth = start_growth - progress * (start_growth - end_growth)
                            variation = np.random.normal(0, 0.005)
                            adjusted_growth = current_growth * (1 + variation)
                            predictions['growth_rates'].append(float(adjusted_growth))
            else:
                logger.warning("Insufficient growth data, using default pattern with smooth decay")
                # Use smooth declining growth pattern
                initial_growth = 0.15  # Starting growth rate
                min_growth = max(terminal_growth * 1.2, 0.03)  # Minimum growth rate with buffer
                
                # Create a smooth declining curve
                for year in range(forecast_years):
                    # Calculate progress
                    progress = (year / (forecast_years - 1)) ** 0.5 if forecast_years > 1 else 1
                    
                    # Calculate growth with smooth decay
                    growth = initial_growth - progress * (initial_growth - min_growth)
                    
                    # Add small random variation
                    variation = np.random.normal(0, 0.005)
                    growth = growth * (1 + variation)
                    
                    predictions['growth_rates'].append(float(growth))
            
            # Process each factor type with dynamic prediction and stricter bounds
            factor_types = [
                ('capex', self.capex_pipeline, 0.05, 0.25),  # Reduced max cap from 0.3 to 0.25
                ('wc', self.wc_pipeline, -0.05, 0.15),      # Reduced max from 0.25 to 0.15
                ('depr', self.depr_pipeline, 0.02, 0.15),
                ('tax', self.tax_pipeline, 0.15, 0.35)
            ]
            
            for factor_type, pipeline, min_bound, max_bound in factor_types:
                X, y = self.prepare_factor_features(factor_type)
                
                if X is not None and len(X) >= 3:
                    pipeline.fit(X, y)
                    last_feat = X.iloc[-1:].copy()
                    
                    # Calculate historical statistics
                    historical_mean = float(y.mean())
                    historical_std = float(y.std())
                    logger.info(f"{factor_type} - Training samples: {len(X)}, Mean: {historical_mean:.2%}, Std: {historical_std:.2%}")
                    
                    # Generate predictions with tighter bounds
                    factor_preds = []
                    for year in range(forecast_years):
                        # Predict the ratio directly
                        ratio_pred = float(pipeline.predict(last_feat)[0])
                        
                        # Scale variation down over time for more stability
                        variation = np.random.normal(0, 0.01 + 0.005 * year)
                        ratio_pred = ratio_pred * (1 + variation)
                        
                        # Use tighter bounds for year 1 based on historical data
                        # Then gradually move toward industry norms for later years
                        year_weight = min(year / 2, 1.0)  # 0, 0.5, 1.0, 1.0, 1.0
                        industry_norm = {'capex': 0.10, 'wc': 0.05, 'depr': 0.05, 'tax': 0.20}[factor_type]
                        
                        # Blend historical mean with industry norm based on year
                        blended_mean = historical_mean * (1 - year_weight) + industry_norm * year_weight
                        
                        # Tighten std bounds for later years (more predictable)
                        std_factor = max(1.0 - (year * 0.1), 0.6)
                        effective_std = historical_std * std_factor
                        
                        # Calculate safe bounds
                        upper_bound = min(blended_mean + 1.5 * effective_std, max_bound)
                        lower_bound = max(blended_mean - 1.5 * effective_std, min_bound)
                        
                        # Special handling for CAPEX ratio - it should be higher in growth years
                        if factor_type == 'capex' and year < len(predictions['growth_rates']):
                            growth_rate = predictions['growth_rates'][year]
                            if growth_rate > 0.15:  # For high growth years
                                # Scale CAPEX with growth rate
                                growth_adjustment = min((growth_rate - 0.15) * 0.5, 0.1)
                                upper_bound = min(upper_bound + growth_adjustment, max_bound)
                                lower_bound = min(lower_bound + growth_adjustment, upper_bound - 0.05)
                        
                        # Ensure tax rates remain within reasonable bounds
                        if factor_type == 'tax':
                            upper_bound = min(upper_bound, 0.35)
                            lower_bound = max(lower_bound, 0.15)
                        
                        # Constrain the prediction
                        ratio_pred = max(min(ratio_pred, upper_bound), lower_bound)
                        
                        factor_preds.append(float(ratio_pred))
                        
                        # Update features for next prediction
                        if 'current_capex_ratio' in last_feat and factor_type == 'capex':
                            if 'prev_capex_ratio' in last_feat:
                                last_feat['prev_capex_ratio'] = last_feat['current_capex_ratio']
                            last_feat['current_capex_ratio'] = ratio_pred
                    
                    predictions[f'{factor_type}_factors'] = factor_preds
                else:
                    # Use industry defaults with slight variations
                    logger.warning(f"Insufficient {factor_type} data, using defaults with variation")
                    default_base = {'capex': 0.1, 'wc': 0.05, 'depr': 0.05, 'tax': 0.20}[factor_type]
                    defaults = []
                    
                    for year in range(forecast_years):
                        # Less variation for defaults to improve stability
                        variation = np.random.normal(0, 0.005 + 0.005 * year)
                        val = default_base * (1 + variation)
                        
                        # Make CAPEX scale with growth in early years
                        if factor_type == 'capex' and year < len(predictions['growth_rates']):
                            growth_rate = predictions['growth_rates'][year]
                            if growth_rate > 0.1:
                                # Add small adjustment for growth
                                growth_adj = min((growth_rate - 0.1) * 0.3, 0.05)
                                val += growth_adj
                        
                        val = max(min(val, max_bound), min_bound)
                        defaults.append(val)
                        
                    predictions[f'{factor_type}_factors'] = defaults
            
            # Additional sanity check: for very high growth companies, ensure WC doesn't drain too much cash
            if any(g > 0.25 for g in predictions['growth_rates']):
                # Use more conservative WC targets to avoid cash drain
                wc_adjustments = [min(r, 0.1) for r in predictions['wc_factors']]
                predictions['wc_factors'] = wc_adjustments
            
            logger.info("Final ML predictions:")
            for factor, values in predictions.items():
                if factor != 'forecast_years':
                    logger.info(f"{factor}: {[f'{x:.2%}' for x in values]}")
            
            predictions['forecast_years'] = forecast_years
            return predictions
                
        except Exception as e:
            logger.error(f"Error in predictions: {e}", exc_info=True)
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
