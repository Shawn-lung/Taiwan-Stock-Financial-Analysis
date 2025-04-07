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
import os
import numpy as np

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
                
                # Calculate industry-wide average growth rates
                if industry:
                    industry_growth_stats = self.calculate_industry_growth_rates(industry, years=5)
                    if industry_growth_stats:
                        results['industry_growth_stats'] = industry_growth_stats
                
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

    def calculate_industry_growth_rates(self, industry: str, years: int = 5) -> Dict:
        """Calculate the average revenue growth rates for an entire industry.
        
        Args:
            industry: Industry name to analyze
            years: Number of years to forecast
            
        Returns:
            Dictionary with industry growth statistics including:
            - average_growth_rates: List of average growth rates for each future year
            - company_count: Number of companies with valid data used in calculation
            - growth_dispersion: Standard deviation of growth rates
            - growth_range: Min and max growth rates
        """
        try:
            logger.info(f"Calculating industry-wide growth rates for {industry}")
            
            if not hasattr(self, 'industry_model') or self.industry_model is None:
                self.industry_model = IndustryValuationModel(load_pretrained=True)
                logger.info("Initialized industry model for growth rate calculation")
            
            # Check if industry name needs standardization
            industry_lower = industry.lower()
            
            # Get industry training data that has revenue information
            industry_file = os.path.join(
                self.industry_model.data_dir, 
                f"{industry_lower.replace(' ', '_')}_training.csv"
            )
            
            # Debug: Check if file exists
            if not os.path.exists(industry_file):
                logger.warning(f"No industry data file found for {industry} at {industry_file}")
                
                # Try to find any matching industry file
                potential_files = []
                for filename in os.listdir(self.industry_model.data_dir):
                    if filename.endswith('_training.csv'):
                        potential_files.append(filename)
                
                if potential_files:
                    logger.info(f"Available industry files: {', '.join(potential_files)}")
                    
                    # Try to find a close match
                    for file in potential_files:
                        if industry_lower in file.lower():
                            industry_file = os.path.join(self.industry_model.data_dir, file)
                            logger.info(f"Found potential match: {file}")
                            break
                
                # If still no match, use industry benchmarks if available
                if not os.path.exists(industry_file):
                    logger.warning(f"Using fallback from industry benchmarks for {industry}")
                    return self._get_fallback_industry_growth(industry)
                
            # Load industry data
            industry_df = pd.read_csv(industry_file)
            logger.info(f"Loaded industry data for {industry} with {len(industry_df)} records")
            logger.info(f"Industry data columns: {industry_df.columns.tolist()}")
            
            # Check if key columns exist
            required_cols = ['stock_id', 'timestamp', 'revenue']
            missing_cols = [col for col in required_cols if col not in industry_df.columns]
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}")
                
                # Try to find alternative column names
                if 'stock_id' in missing_cols and 'symbol' in industry_df.columns:
                    industry_df['stock_id'] = industry_df['symbol']
                    missing_cols.remove('stock_id')
                
                if 'timestamp' in missing_cols and 'date' in industry_df.columns:
                    industry_df['timestamp'] = industry_df['date']
                    missing_cols.remove('timestamp')
                
                if missing_cols:
                    logger.warning(f"Cannot proceed with missing columns: {missing_cols}")
                    return self._get_fallback_industry_growth(industry)
            
            # Get unique companies in this industry
            companies = industry_df['stock_id'].unique() if 'stock_id' in industry_df.columns else []
            company_count = len(companies)
            
            if company_count == 0:
                logger.warning(f"No companies found in industry data for {industry}")
                return self._get_fallback_industry_growth(industry)
                
            logger.info(f"Found {company_count} companies in {industry} industry")
            
            # Calculate historical growth rates for each company
            company_growth_rates = []
            
            # First try to use historical_growth if available
            historical_growth_found = False
            if 'historical_growth' in industry_df.columns:
                growth_values = industry_df['historical_growth'].dropna().values
                if len(growth_values) >= 5:
                    company_growth_rates = growth_values.tolist()
                    historical_growth_found = True
                    logger.info(f"Using {len(growth_values)} historical_growth values from dataset")
            
            if 'historical_growth_mean' in industry_df.columns:
                growth_values = industry_df['historical_growth_mean'].dropna().values
                if len(growth_values) >= 3:
                    company_growth_rates = growth_values.tolist()
                    historical_growth_found = True
                    logger.info(f"Using {len(growth_values)} historical_growth_mean values from dataset")
            
            # If not enough data from historical_growth, calculate from revenue
            companies_with_growth = 0
            if len(company_growth_rates) < 5 and 'revenue' in industry_df.columns:
                logger.info(f"Calculating growth rates from revenue data for {industry}")
                
                # Group by company to analyze growth rates
                for company in companies:
                    company_data = industry_df[industry_df['stock_id'] == company].sort_values('timestamp')
                    
                    # Need at least 2 data points to calculate growth
                    if len(company_data) < 2:
                        continue
                        
                    # Calculate year-over-year growth rates
                    revenues = company_data['revenue'].values
                    company_growth = []
                    
                    for i in range(1, len(revenues)):
                        if revenues[i-1] > 0:  # Avoid division by zero
                            growth_rate = (revenues[i] - revenues[i-1]) / revenues[i-1]
                            # Filter out extreme values
                            if -0.5 <= growth_rate <= 2.0:  # Filter very extreme outliers
                                company_growth.append(growth_rate)
                    
                    if company_growth:
                        company_growth_rates.extend(company_growth)
                        companies_with_growth += 1
                
                logger.info(f"Calculated growth rates for {companies_with_growth} out of {company_count} companies")
            
            # Process growth rates
            if not company_growth_rates:
                logger.warning(f"No valid growth rates found for {industry}")
                return self._get_fallback_industry_growth(industry)
                
            logger.info(f"Total growth data points: {len(company_growth_rates)}")
            
            # Remove outliers using IQR method
            growth_array = np.array(company_growth_rates)
            
            # Log original stats
            original_mean = growth_array.mean()
            original_median = np.median(growth_array)
            logger.info(f"Original growth stats - Mean: {original_mean:.1%}, Median: {original_median:.1%}")
            
            # Filter outliers
            q1, q3 = np.percentile(growth_array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            filtered_growth = growth_array[(growth_array >= lower_bound) & (growth_array <= upper_bound)]
            
            if len(filtered_growth) == 0:
                logger.warning(f"No valid growth rates after filtering outliers for {industry}")
                # Use original array if filtering removed everything
                filtered_growth = growth_array
            
            # Calculate statistics
            mean_growth = filtered_growth.mean()
            median_growth = np.median(filtered_growth)
            std_growth = filtered_growth.std()
            min_growth = filtered_growth.min()
            max_growth = filtered_growth.max()
            
            logger.info(f"Industry {industry} growth statistics after filtering - Mean: {mean_growth:.1%}, Median: {median_growth:.1%}, Std: {std_growth:.1%}")
            logger.info(f"Growth range: {min_growth:.1%} to {max_growth:.1%}")
            
            # Generate forecasted growth rates for future years with a declining pattern
            # Typically, growth rates start higher and decline toward a longer-term average
            forecasted_growth_rates = []
            
            # If we have historical_growth_mean in the benchmarks, use that as a reference
            long_term_growth = None
            if self.industry_model.industry_benchmarks is not None:
                industry_row = self.industry_model.industry_benchmarks[
                    self.industry_model.industry_benchmarks['industry'] == industry
                ]
                if not industry_row.empty:
                    if 'historical_growth_mean_median' in industry_row.columns:
                        long_term_growth = float(industry_row['historical_growth_mean_median'].iloc[0])
                        logger.info(f"Using benchmark historical_growth_mean_median: {long_term_growth:.1%}")
                    elif 'historical_growth_mean_mean' in industry_row.columns:
                        long_term_growth = float(industry_row['historical_growth_mean_mean'].iloc[0])
                        logger.info(f"Using benchmark historical_growth_mean_mean: {long_term_growth:.1%}")
            
            # If no benchmark data, use the calculated mean
            if long_term_growth is None:
                long_term_growth = mean_growth
                logger.info(f"Using calculated mean as long-term growth: {long_term_growth:.1%}")
            
            # Generate forecast with gradual decline toward long-term average
            for i in range(years):
                # Start with slightly above mean and decline toward long-term
                weight = 1.0 - (i / years)
                year_growth = (mean_growth * 1.1) * weight + long_term_growth * (1 - weight)
                forecasted_growth_rates.append(year_growth)
            
            # Log the forecasted rates
            forecast_str = ", ".join([f"{rate:.1%}" for rate in forecasted_growth_rates])
            logger.info(f"Forecasted industry growth rates: {forecast_str}")
            
            # Prepare result
            result = {
                'average_growth_rates': forecasted_growth_rates,
                'company_count': company_count,
                'companies_with_growth': companies_with_growth,
                'historical_mean_growth': mean_growth,
                'historical_median_growth': median_growth,
                'growth_dispersion': std_growth,
                'growth_range': (min_growth, max_growth),
                'data_points': len(filtered_growth)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating industry growth rates: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_fallback_industry_growth(industry)
    
    def _get_fallback_industry_growth(self, industry: str) -> Dict:
        """Get fallback industry growth stats when calculation fails."""
        logger.info(f"Using fallback industry growth for {industry}")
        
        # Default growth rates by industry - these are conservative estimates
        industry_growth_defaults = {
            'Semiconductors': [0.18, 0.16, 0.14, 0.12, 0.11],
            'Electronics': [0.15, 0.14, 0.13, 0.12, 0.10],
            'Banking': [0.08, 0.07, 0.07, 0.06, 0.06],
            'Telecommunications': [0.06, 0.05, 0.05, 0.04, 0.04],
            'Financial Services': [0.09, 0.08, 0.08, 0.07, 0.07],
            'Computer Hardware': [0.12, 0.11, 0.10, 0.09, 0.08],
            'Food & Beverage': [0.07, 0.06, 0.06, 0.05, 0.05],
            'Retail': [0.08, 0.07, 0.07, 0.06, 0.06],
            'Healthcare': [0.10, 0.09, 0.09, 0.08, 0.08],
            'Utilities': [0.04, 0.04, 0.04, 0.03, 0.03],
            'Materials': [0.06, 0.06, 0.05, 0.05, 0.05],
            'Industrial': [0.07, 0.07, 0.06, 0.06, 0.05]
        }
        
        # Find closest industry match or use default
        industry_lower = industry.lower()
        growth_rates = None
        
        for ind, rates in industry_growth_defaults.items():
            if ind.lower() in industry_lower or industry_lower in ind.lower():
                growth_rates = rates
                logger.info(f"Found matching fallback rates for {industry} -> {ind}")
                break
        
        # If no match, use a medium growth default
        if growth_rates is None:
            growth_rates = industry_growth_defaults.get('Industrial', [0.10, 0.09, 0.08, 0.07, 0.06])
            logger.info(f"Using general fallback rates for {industry}")
        
        # Try to get mean growth from benchmarks if available
        avg_growth = None
        if hasattr(self, 'industry_model') and self.industry_model and self.industry_model.industry_benchmarks is not None:
            industry_row = self.industry_model.industry_benchmarks[
                self.industry_model.industry_benchmarks['industry'] == industry
            ]
            if not industry_row.empty:
                if 'historical_growth_mean_median' in industry_row.columns:
                    avg_growth = float(industry_row['historical_growth_mean_median'].iloc[0])
                elif 'historical_growth_mean_mean' in industry_row.columns:
                    avg_growth = float(industry_row['historical_growth_mean_mean'].iloc[0])
        
        # Create fallback stats
        return {
            'average_growth_rates': growth_rates,
            'company_count': 0,  # Indicate this is fallback data
            'companies_with_growth': 0,
            'historical_mean_growth': avg_growth or growth_rates[0],
            'historical_median_growth': avg_growth or growth_rates[0],
            'growth_dispersion': 0.02,  # Nominal value
            'growth_range': (growth_rates[-1] * 0.8, growth_rates[0] * 1.2),
            'is_fallback': True
        }

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
