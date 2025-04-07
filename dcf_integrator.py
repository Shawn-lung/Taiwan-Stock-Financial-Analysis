import logging
from util.db_data_provider import DBFinancialDataProvider
from dcf_model import DCFModel
from ml_predictor import GrowthPredictor
from deep_learning import DeepFinancialForecaster
from industry_valuation_model import IndustryValuationModel
from typing import Dict
import pandas as pd
import yfinance as yf
from data_fetcher import FinancialDataFetcher  # Import FinancialDataFetcher to use its utilities

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegratedValuationModel:
    """Integrate DCF, ML, DL, and industry-specific valuation adjustments."""
    
    def __init__(self, use_ml: bool = True, use_dl: bool = True, use_industry: bool = True, db_path: str = "finance_data.db"):
        """Initialize the integrated valuation model.
        
        Args:
            use_ml: Whether to use ML predictions for growth factors
            use_dl: Whether to use deep learning predictions
            use_industry: Whether to apply industry-specific adjustments
            db_path: Path to the SQLite database file
        """
        self.use_ml = use_ml
        self.use_dl = use_dl
        self.use_industry = use_industry
        self.db_path = db_path
        self.db_provider = DBFinancialDataProvider(db_path)
        
        # Initialize industry model if needed - always load pretrained models
        if self.use_industry:
            self.industry_model = IndustryValuationModel(load_pretrained=True)
            logger.info("Using pre-trained industry valuation models")
        
    def run_valuation(self, ticker: str, industry: str = None) -> Dict:
        """Run a comprehensive valuation with all available models.
        
        Args:
            ticker: Stock ticker symbol
            industry: Optional industry classification (detected if not provided)
            
        Returns:
            Dictionary with valuation results from all models
        """
        results = {
            'ticker': ticker,
            'models': {}
        }
        
        # 1. Standard DCF calculation
        standard_dcf = DCFModel(
            stock_code=ticker,
            forecast_years=5,
            perpetual_growth_rate=0.025
        )
        standard_price = standard_dcf.calculate_stock_price()
        results['models']['standard_dcf'] = standard_price
        
        # 2. ML-enhanced DCF if requested
        if self.use_ml:
            try:
                predictor = GrowthPredictor(ticker)
                ml_predictions = predictor.predict_all_factors(forecast_years=5)
                
                if ml_predictions:
                    ml_dcf = DCFModel(
                        stock_code=ticker,
                        forecast_years=5,
                        perpetual_growth_rate=0.025,
                        manual_growth_rates=ml_predictions['growth_rates'],
                        manual_capex_factors=ml_predictions['capex_factors'], 
                        manual_wc_factors=ml_predictions['wc_factors'],
                        manual_depr_factors=ml_predictions['depr_factors'],
                        manual_tax_factors=ml_predictions['tax_factors']
                    )
                    
                    ml_price = ml_dcf.calculate_stock_price()
                    results['models']['ml_dcf'] = ml_price
                    results['ml_predictions'] = ml_predictions
            except Exception as e:
                logger.error(f"Error in ML valuation: {e}")
        
        # 3. Add Deep Learning component if requested
        if self.use_dl and self.use_ml and 'ml_predictions' in results:
            try:
                # Get enhanced financial data from multiple sources
                enhanced_data = self.enhance_financial_data(ticker)
                
                if enhanced_data is not None and len(enhanced_data) > 2:  # Ensure we have enough data points
                    logger.info(f"Using enhanced dataset with {len(enhanced_data)} records for deep learning")
                    
                    # Create deep learning forecaster
                    deep_forecaster = DeepFinancialForecaster()
                    
                    # Get DL growth predictions using the enhanced data
                    dl_growth_predictions = deep_forecaster.predict_future_growth(enhanced_data, forecast_years=5)
                else:
                    # Fallback to standard data if enhanced data is insufficient
                    logger.warning(f"Enhanced data insufficient, using standard financial data for {ticker}")
                    
                    # Get financial data from DCF model
                    financial_data = standard_dcf.get_financial_data()
                    
                    # Add stock_code attribute
                    if isinstance(financial_data, dict):
                        financial_data['stock_code'] = ticker
                    else:
                        if hasattr(financial_data, 'attrs'):
                            financial_data.attrs['stock_code'] = ticker
                        else:
                            try:
                                financial_data = financial_data.copy()
                                financial_data.attrs = {'stock_code': ticker}
                            except:
                                financial_data = {'data': financial_data, 'stock_code': ticker}
                    
                    # Get DL growth predictions
                    deep_forecaster = DeepFinancialForecaster()
                    dl_growth_predictions = deep_forecaster.predict_future_growth(financial_data, forecast_years=5)
                
                if dl_growth_predictions:
                    # Create an ensemble of ML + DL predictions
                    ensemble_growth = []
                    for i in range(5):
                        # Gradually increase DL weight in later years
                        dl_weight = min(0.3 + (i * 0.1), 0.7)
                        ml_weight = 1.0 - dl_weight
                        blended = (results['ml_predictions']['growth_rates'][i] * ml_weight + 
                                 dl_growth_predictions[i] * dl_weight)
                        ensemble_growth.append(blended)
                    
                    # Create ensemble DCF model
                    ensemble_dcf = DCFModel(
                        stock_code=ticker,
                        forecast_years=5,
                        perpetual_growth_rate=0.025,
                        manual_growth_rates=ensemble_growth,
                        manual_capex_factors=results['ml_predictions']['capex_factors'], 
                        manual_wc_factors=results['ml_predictions']['wc_factors'],
                        manual_depr_factors=results['ml_predictions']['depr_factors'],
                        manual_tax_factors=results['ml_predictions']['tax_factors']
                    )
                    
                    # Calculate ensemble price
                    ensemble_price = ensemble_dcf.calculate_stock_price()
                    results['models']['ml_dl_ensemble_dcf'] = ensemble_price
                    results['dl_predictions'] = dl_growth_predictions
            except Exception as e:
                logger.error(f"Error in DL valuation: {e}")
        
        # 4. Add industry-specific adjustments if requested
        if self.use_industry:
            try:
                # Detect industry if not provided
                if industry is None:
                    # Try to detect from financial data
                    financial_data = standard_dcf.get_financial_data()
                    deep_forecaster = DeepFinancialForecaster()
                    industry = deep_forecaster._detect_industry_from_financials(financial_data)
                
                # Get financial metrics for adjustment
                metrics = self._extract_financial_metrics(standard_dcf)
                
                # Adjust each valuation model
                for model_name, base_price in results['models'].items():
                    if base_price is not None:
                        adjustment = self.industry_model.adjust_dcf_valuation(
                            industry=industry,
                            base_valuation=base_price,
                            financial_metrics=metrics
                        )
                        # Store adjustment details
                        adjustment_key = f"{model_name}_industry_adjusted"
                        results[adjustment_key] = adjustment
                
                # Store detected industry
                results['detected_industry'] = industry
                
            except Exception as e:
                logger.error(f"Error applying industry adjustments: {e}")
        
        return results
    
    def _extract_financial_metrics(self, dcf_model: DCFModel) -> Dict:
        """Extract financial metrics from DCF model for industry adjustment."""
        metrics = {}
        
        try:
            # Get key financial metrics from DCF model's data
            financial_data = dcf_model.get_financial_data()
            if financial_data is None:
                return metrics
            
            # Revenue growth
            growth_rates = dcf_model.calculate_historical_growth_rates()
            if growth_rates:
                metrics['historical_growth_mean'] = sum(growth_rates) / len(growth_rates)
            
            # Get last year's data
            if financial_data and hasattr(financial_data, 'iloc'):
                # Try to get operating margin
                if 'Operating Margin' in financial_data.index:
                    metrics['operating_margin'] = float(financial_data.loc['Operating Margin'].iloc[-1])
                elif 'Operating Income' in financial_data.index and 'Total Revenue' in financial_data.index:
                    op_income = float(financial_data.loc['Operating Income'].iloc[-1])
                    revenue = float(financial_data.loc['Total Revenue'].iloc[-1])
                    if revenue > 0:
                        metrics['operating_margin'] = op_income / revenue
                
                # Try to get net margin
                if 'Net Margin' in financial_data.index:
                    metrics['net_margin'] = float(financial_data.loc['Net Margin'].iloc[-1])
                elif 'Net Income' in financial_data.index and 'Total Revenue' in financial_data.index:
                    net_income = float(financial_data.loc['Net Income'].iloc[-1])
                    revenue = float(financial_data.loc['Total Revenue'].iloc[-1])
                    if revenue > 0:
                        metrics['net_margin'] = net_income / revenue
                
                # Try to get ROE
                if 'ROE' in financial_data.index:
                    metrics['roe'] = float(financial_data.loc['ROE'].iloc[-1])
                elif 'Net Income' in financial_data.index and 'Total Equity' in financial_data.index:
                    net_income = float(financial_data.loc['Net Income'].iloc[-1])
                    equity = float(financial_data.loc['Total Equity'].iloc[-1])
                    if equity > 0:
                        metrics['roe'] = net_income / equity
                
                # Try to get debt-to-equity
                if 'Total Debt' in financial_data.index and 'Total Equity' in financial_data.index:
                    debt = float(financial_data.loc['Total Debt'].iloc[-1])
                    equity = float(financial_data.loc['Total Equity'].iloc[-1])
                    if equity > 0:
                        metrics['debt_to_equity'] = debt / equity
        
        except Exception as e:
            logger.error(f"Error extracting financial metrics: {e}")
        
        return metrics

    def enhance_financial_data(self, ticker: str) -> pd.DataFrame:
        """Fetch financial data from database and enhance it with additional data sources.
        
        This method addresses the problem of having insufficient data points for deep learning,
        by combining data from both the database and yfinance.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Enhanced DataFrame with more historical financial data
        """
        try:
            logger.info(f"Enhancing financial data for {ticker} from multiple sources")
            
            # Step 1: Get data from database
            db_data = self.db_provider.get_financial_data(ticker)
            logger.info(f"Database provided {len(db_data)} records for {ticker}")
            
            # Step 2: Get data from yfinance using the FinancialDataFetcher
            data_fetcher = FinancialDataFetcher()
            yf_data = data_fetcher.get_financial_data(ticker, force_refresh=False)
            
            has_yf_data = yf_data and 'income_statement' in yf_data and not yf_data['income_statement'].empty
            logger.info(f"YFinance data {'available' if has_yf_data else 'not available'} for {ticker}")
            if has_yf_data:
                logger.info(f"YFinance provided data with shape: {yf_data['income_statement'].shape}")
            
            # Step 3: Convert into a consistent format for deep learning
            combined_data = {}
            
            # Process DB data first
            if db_data and len(db_data) > 0:
                # Extract key metrics we need for deep learning
                metrics = []
                for record in db_data:
                    # Skip records with missing essential data
                    if not record.get('revenue') or not record.get('operating_income'):
                        continue
                        
                    metrics.append({
                        'timestamp': record.get('date'),
                        'revenue': record.get('revenue'),
                        'operating_income': record.get('operating_income'),
                        'net_income': record.get('net_income'),
                        'stock_id': ticker
                    })
                
                if metrics:
                    combined_data.update({
                        'db': pd.DataFrame(metrics)
                    })
            
            # Process yfinance data next
            if yf_data and 'income_statement' in yf_data and not yf_data['income_statement'].empty:
                income_stmt = yf_data['income_statement']
                
                # Ensure correct orientation (dates in columns)
                if isinstance(income_stmt.index[0], str):
                    # Data is already correctly oriented
                    metrics = []
                    
                    for col in income_stmt.columns:
                        revenue = None
                        op_income = None
                        net_income = None
                        
                        # Try to extract revenue
                        for rev_key in ['Total Revenue', 'Revenue']:
                            if rev_key in income_stmt.index:
                                val = income_stmt.loc[rev_key, col]
                                if pd.notna(val) and val > 0:
                                    revenue = float(val)
                                    break
                        
                        # Try to extract operating income
                        if 'Operating Income' in income_stmt.index:
                            val = income_stmt.loc['Operating Income', col]
                            if pd.notna(val):
                                op_income = float(val)
                        
                        # Try to extract net income
                        for ni_key in ['Net Income', 'Net Income Common Shareholders']:
                            if ni_key in income_stmt.index:
                                val = income_stmt.loc[ni_key, col]
                                if pd.notna(val):
                                    net_income = float(val)
                                    break
                        
                        # Only add if we have the essential metrics
                        if revenue is not None and op_income is not None:
                            metrics.append({
                                'timestamp': col,  # Date is in column
                                'revenue': revenue,
                                'operating_income': op_income,
                                'net_income': net_income,
                                'stock_id': ticker
                            })
                    
                    if metrics:
                        combined_data.update({
                            'yfinance': pd.DataFrame(metrics)
                        })
            
            # Merge data from both sources
            if not combined_data:
                logger.warning(f"No financial data found for {ticker} from any source")
                return None
                
            # Combine dataframes from different sources
            dfs = []
            for source, df in combined_data.items():
                if not df.empty:
                    # Add source identifier
                    df['source'] = source
                    dfs.append(df)
            
            # Merge all dataframes
            if not dfs:
                return None
                
            merged_df = pd.concat(dfs, ignore_index=True)
            
            # Remove duplicates if any (prefer DB data)
            if 'timestamp' in merged_df.columns:
                merged_df.sort_values(['timestamp', 'source'], key=lambda x: x.map({'db': 0, 'yfinance': 1}) if x.name == 'source' else x, inplace=True)
                merged_df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
            
            logger.info(f"Enhanced data now has {len(merged_df)} records for {ticker}")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error enhancing financial data: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize the integrated model
    valuation_model = IntegratedValuationModel(use_ml=True, use_dl=True, use_industry=True)
    
    # Run valuation for a Taiwan semiconductor company
    result = valuation_model.run_valuation("2330.TW", "Semiconductors")
    
    # Print results
    print(f"\nValuation Results for {result['ticker']}:")
    print("-" * 50)
    
    # Print base valuations
    print("\nBase Valuations:")
    for model, price in result['models'].items():
        print(f"  {model.replace('_', ' ').title()}: {price:.2f}")
    
    # Print industry-adjusted valuations if available
    for key in result:
        if key.endswith('_industry_adjusted'):
            model_name = key.replace('_industry_adjusted', '').replace('_', ' ').title()
            adj = result[key]
            print(f"\n{model_name} with Industry Adjustment:")
            print(f"  Base: {adj['base_valuation']:.2f}")
            print(f"  Adjusted: {adj['adjusted_valuation']:.2f}")
            print(f"  Adjustment Factor: {adj['total_adjustment']:.2f}")
            if 'expected_return' in adj and adj['expected_return'] is not None:
                print(f"  Expected 6-month Return: {adj['expected_return']:.2%}")
