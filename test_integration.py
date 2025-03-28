import logging
from ml_predictor import GrowthPredictor
from dcf_model import DCFModel
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ml_dcf_integration(stock_code="2330.TW"):
    """Test the integration between ML predictions and DCF model."""
    logger.info(f"Testing ML + DCF integration for {stock_code}")
    
    # Get ML predictions
    predictor = GrowthPredictor(stock_code)
    predictions = predictor.predict_all_factors(forecast_years=5)
    
    if not predictions:
        logger.error("Failed to get ML predictions")
        return
    
    # Log predictions 
    for factor, values in predictions.items():
        if factor != 'forecast_years':
            logger.info(f"{factor}: {[f'{x:.2%}' for x in values]}")
    
    # Create DCF model with ML predictions
    dcf = DCFModel(
        stock_code=stock_code,
        forecast_years=5,
        perpetual_growth_rate=0.025,
        manual_growth_rates=predictions['growth_rates'],
        manual_capex_factors=predictions['capex_factors'], 
        manual_wc_factors=predictions['wc_factors'],
        manual_depr_factors=predictions['depr_factors'],
        manual_tax_factors=predictions['tax_factors']
    )
    
    # Get FCF projections
    fcf_list = dcf.forecast_fcf_list()
    if not fcf_list:
        logger.error("Failed to forecast FCF")
        return
        
    # Calculate stock price
    price = dcf.calculate_stock_price()
    
    # Run a validation check on the result
    if price is not None:
        # Check if price is realistic
        current_price = dcf.stock.info.get('regularMarketPrice', 0)
        
        if current_price > 0:
            price_ratio = price / current_price
            if price_ratio < 0.5 or price_ratio > 2.0:
                logger.warning(f"Calculated price ({price:.2f}) is significantly different from current price ({current_price:.2f})")
            else:
                logger.info(f"Price seems reasonable: calculated={price:.2f}, current={current_price:.2f}")
        
        logger.info(f"Estimated stock price: {price:.2f}")
    else:
        logger.error("Failed to calculate a valid stock price")
    
    # Create a summary table
    years = range(1, 6)
    data = {
        'Year': [f"Year {y}" for y in years],
        'Growth': [f"{x:.2%}" for x in predictions['growth_rates']],
        'CAPEX Ratio': [f"{x:.2%}" for x in predictions['capex_factors']],
        'Depreciation Ratio': [f"{x:.2%}" for x in predictions['depr_factors']],
        'WC Ratio': [f"{x:.2%}" for x in predictions['wc_factors']],
        'Tax Rate': [f"{x:.2%}" for x in predictions['tax_factors']],
        'FCF': [f"{fcf:,.0f}" for fcf in fcf_list]
    }
    
    df = pd.DataFrame(data)
    logger.info("\nProjection Summary:\n" + df.to_string())
    
    return price

if __name__ == "__main__":
    test_ml_dcf_integration()
