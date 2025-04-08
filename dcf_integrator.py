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
                
                # Store detected industry
                results['detected_industry'] = industry
                
                # If industry is None and we have a db_provider, try to detect from database
                if industry is None and hasattr(self, 'db_provider'):
                    try:
                        db_industry = self.db_provider.get_industry(ticker)
                        if db_industry:
                            logger.info(f"Found industry '{db_industry}' for {ticker} in database")
                            industry = db_industry
                            results['detected_industry'] = industry
                    except Exception as e:
                        logger.error(f"Error getting industry from database: {e}")
                
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
            
            # Handle the 'default' industry case - try to find a better match
            if industry == 'default' or industry is None:
                logger.info("Default or None industry detected, attempting to find a suitable industry match")
                
                # Try to find an appropriate industry based on the ticker if available
                ticker = getattr(self, 'ticker', None)
                if ticker and isinstance(ticker, str):
                    # Try to get industry from the database first
                    try:
                        db_industry = self.db_provider.get_industry(ticker)
                        if db_industry and db_industry != 'default' and db_industry is not None:
                            industry = db_industry
                            logger.info(f"Using database-provided industry '{industry}' instead of default")
                    except Exception as e:
                        logger.debug(f"Could not get industry from database: {e}")
                    
                    # If still default, try pattern matching for Taiwan stocks
                    if industry == 'default' or industry is None:
                        if '.TW' in ticker:
                            base_number = ticker.split('.')[0]
                            # Match semiconductor companies
                            if base_number in ['2330', '2454', '2379', '2337', '2308', '2303', '2409', '2344', '2351']:
                                industry = 'semiconductors'
                                logger.info(f"Matched ticker {ticker} to industry 'semiconductors'")
                            # Match electronics companies    
                            elif base_number.startswith('23') or base_number in ['2317', '2356']:
                                industry = 'electronics'
                                logger.info(f"Matched ticker {ticker} to industry 'electronics'")
                            # Match computer hardware companies
                            elif base_number in ['2382', '2353', '2357', '2324', '2376']:
                                industry = 'computer_hardware'
                                logger.info(f"Matched ticker {ticker} to industry 'computer_hardware'")
                            # Match telecom companies
                            elif base_number in ['2412', '3045', '4904', '4977']:
                                industry = 'telecommunications'
                                logger.info(f"Matched ticker {ticker} to industry 'telecommunications'")
                            # Match financial service companies
                            elif base_number.startswith('26') or base_number.startswith('27'):
                                industry = 'financial_services' 
                                logger.info(f"Matched ticker {ticker} to industry 'financial_services'")
            
            # Check if industry name needs standardization
            industry_lower = industry.lower() if industry else 'default'
            
            # Convert spaces to underscores for file matching
            industry_file_name = industry_lower.replace(' ', '_')
            
            # Get industry training data that has revenue information
            industry_file = os.path.join(
                self.industry_model.data_dir, 
                f"{industry_file_name}_training.csv"
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
                    best_match = None
                    best_score = 0
                    
                    for file in potential_files:
                        # Remove _training.csv to get the industry name
                        file_industry = file.replace('_training.csv', '')
                        
                        # Score based on string similarity
                        score = 0
                        if industry_lower in file_industry:
                            # Direct substring match gets high score
                            score = 10
                        else:
                            # Calculate word-based similarity 
                            industry_words = set(industry_lower.split('_'))
                            file_words = set(file_industry.split('_'))
                            common_words = industry_words.intersection(file_words)
                            score = len(common_words) * 5  # 5 points per common word
                            
                            # For industries with multiple words, partial matching
                            if len(industry_words) > 1 or len(file_words) > 1:
                                for i_word in industry_words:
                                    for f_word in file_words:
                                        if i_word in f_word or f_word in i_word:
                                            score += 2  # 2 points per partial word match
                            
                        if score > best_score:
                            best_score = score
                            best_match = file
                    
                    # Use the best match if score is high enough
                    if best_score > 0:
                        industry_file = os.path.join(self.industry_model.data_dir, best_match)
                        clean_industry_name = best_match.replace('_training.csv', '').replace('_', ' ')
                        logger.info(f"Found potential match: {best_match} (score: {best_score})")
                        industry = clean_industry_name  # Update the industry name
                
                # If still no match, use industry benchmarks if available
                if not os.path.exists(industry_file):
                    logger.warning(f"Using fallback from industry benchmarks for {industry}")
                    return self._get_fallback_industry_growth(industry)
            
            # Load industry training data
            industry_df = pd.read_csv(industry_file)
            logger.info(f"Loaded {len(industry_df)} records from {industry_file}")
            
            if 'revenue' not in industry_df.columns:
                logger.warning(f"No revenue data found in {industry_file}")
                return self._get_fallback_industry_growth(industry)
            
            # Calculate growth rates for each company
            all_growth_rates = []
            stats = {}
            
            # Group by stock_id and calculate growth rates
            for stock_id, data in industry_df.groupby('stock_id'):
                if len(data) < 2:
                    continue
                    
                # Sort by timestamp or date
                time_field = 'timestamp' if 'timestamp' in data.columns else 'date' 
                if time_field in data.columns:
                    data = data.sort_values(time_field)
                
                # Calculate growth rate
                if 'revenue' in data.columns:
                    try:
                        growth_rates = data['revenue'].pct_change().dropna()
                        
                        if growth_rates.empty:
                            continue
                            
                        # Drop outliers (extreme growth rates)
                        growth_rates = growth_rates[(growth_rates > -0.5) & (growth_rates < 2.0)]
                        
                        if not growth_rates.empty:
                            all_growth_rates.extend(growth_rates.tolist())
                    except Exception as e:
                        logger.debug(f"Error calculating growth rates for {stock_id}: {e}")
                        continue
            
            # Calculate statistics
            if not all_growth_rates:
                logger.warning(f"No valid growth rates calculated for {industry}")
                return self._get_fallback_industry_growth(industry)
                
            # Count unique companies
            unique_companies = industry_df['stock_id'].nunique()
            companies_with_growth = len(all_growth_rates) // 2  # Rough estimate
            
            # Calculate key statistics
            mean_growth = np.mean(all_growth_rates)
            median_growth = np.median(all_growth_rates)
            growth_dispersion = np.std(all_growth_rates)
            growth_range = (min(all_growth_rates), max(all_growth_rates))
            
            # Generate projected future growth rates with decay
            # Start with the mean growth and apply decay
            base_growth = max(0.03, min(0.25, mean_growth))  # Cap between 3% and 25%
            
            # Define industry-specific decay factors
            decay_by_industry = {
                'semiconductors': 0.85,
                'electronics': 0.87,
                'telecommunications': 0.92,
                'financial services': 0.90,
                'computer hardware': 0.88,
                'utilities': 0.95,
                'default': 0.90
            }
            
            # Get decay factor for this industry
            decay_factor = decay_by_industry.get(
                industry.lower(), 
                decay_by_industry['default']
            )
            
            # Generate growth rates
            growth_rates = [base_growth]
            for i in range(1, years):
                # Apply stronger decay in early years
                year_decay = decay_factor ** (1 + 0.1 * min(i, 3))
                next_growth = growth_rates[-1] * year_decay
                # Apply minimum floor
                next_growth = max(0.02, next_growth)
                growth_rates.append(next_growth)
            
            # Format results
            return {
                'average_growth_rates': growth_rates,
                'company_count': unique_companies,
                'companies_with_growth': companies_with_growth,
                'historical_mean_growth': mean_growth,
                'historical_median_growth': median_growth,
                'growth_dispersion': growth_dispersion,
                'growth_range': growth_range,
                'is_fallback': False
            }
            
        except Exception as e:
            logger.error(f"Error calculating industry growth rates: {e}")
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
