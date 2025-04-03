import logging
from util.db_data_provider import DBFinancialDataProvider
from dcf_model import DCFModel
from ml_predictor import GrowthPredictor
from deep_learning import DeepFinancialForecaster
from industry_valuation_model import IndustryValuationModel
from typing import Dict

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
        
        # Initialize industry model if needed
        if self.use_industry:
            self.industry_model = IndustryValuationModel()
        
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
                # Get financial data
                financial_data = standard_dcf.get_financial_data()
                
                # Create deep learning forecaster
                deep_forecaster = DeepFinancialForecaster()
                
                # Add stock info as an attribute to help with industry detection
                if hasattr(financial_data, 'attrs'):
                    financial_data.attrs['stock_code'] = ticker
                else:
                    financial_data = financial_data.copy()
                    financial_data.attrs = {'stock_code': ticker}
                
                # Get DL growth predictions
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
