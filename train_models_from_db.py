import os
import logging
import time
from background_data_collector import BackgroundDataCollector
from industry_valuation_model import IndustryValuationModel
from dcf_integrator import IntegratedValuationModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("db_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_industry_models_from_db():
    """Train industry valuation models using data from background collector."""
    
    # Initialize the background data collector
    collector = BackgroundDataCollector(
        db_path="finance_data.db",
        collection_interval=12  # hours
    )
    
    # Check if we have enough data
    stats = collector.get_db_stats()
    if stats.get('stocks_with_complete_data', 0) < 10:
        logger.warning("Not enough data in database. Starting data collection...")
        
        # Start collection in background
        collector.start_scheduler()
        
        # Wait for initial data collection
        logger.info("Waiting for initial data collection (15 minutes)...")
        time.sleep(15 * 60)  # Wait 15 minutes
        
        # Stop the scheduler
        collector.stop_scheduler()
        
        # Check again if we have enough data
        stats = collector.get_db_stats()
        if stats.get('stocks_with_complete_data', 0) < 10:
            logger.error("Still not enough data to train models. Try running the collector longer.")
            return False
    
    # Initialize the industry model with the background collector
    model = IndustryValuationModel(
        data_dir="industry_data_from_db",
        background_collector=collector
    )
    
    # Prepare data and train models
    logger.info("Training industry models from database data...")
    results = model.train_with_db_data(force_retrain=True)
    
    if not results:
        logger.error("Failed to train models")
        return False
    
    # Log training results
    for industry, result in results.items():
        if 'error' in result:
            logger.error(f"Error training model for {industry}: {result['error']}")
        else:
            logger.info(f"Successfully trained model for {industry} with {result.get('samples', 0)} samples")
    
    logger.info("Model training complete")
    return True

def test_valuation_with_trained_models():
    """Test valuations using the models trained from database data."""
    # Initialize collector for reference
    collector = BackgroundDataCollector(db_path="finance_data.db")
    
    # Get status information
    status = collector.get_collection_status()
    
    # Find a few stocks with complete data for testing
    test_candidates = []
    for industry, group in status.groupby('industry'):
        # Get stocks with all data types present
        valid_stocks = group[
            group['fs_last_update'].notna() & 
            group['bs_last_update'].notna() & 
            group['cf_last_update'].notna() & 
            group['price_last_update'].notna()
        ]['stock_id'].tolist()[:2]  # Take up to 2 stocks per industry
        
        for stock in valid_stocks:
            test_candidates.append((stock, industry))
    
    if not test_candidates:
        logger.error("No stocks with complete data found for testing")
        return False
    
    # Initialize the integrated valuation model (which uses industry model)
    valuation_model = IntegratedValuationModel(
        use_ml=True, 
        use_dl=True, 
        use_industry=True
    )
    
    # Run valuations for test candidates
    for stock_id, industry in test_candidates:
        logger.info(f"Running valuation for {stock_id} ({industry})...")
        
        try:
            result = valuation_model.run_valuation(stock_id, industry)
            
            # Print results
            logger.info(f"Valuation results for {stock_id}:")
            for model_name, price in result['models'].items():
                if price is not None:
                    logger.info(f"  {model_name.replace('_', ' ').title()}: {price:.2f}")
            
            # Print industry-adjusted results
            for key in result:
                if key.endswith('_industry_adjusted'):
                    adj = result[key]
                    logger.info(f"  {key.replace('_', ' ').title()}: {adj['adjusted_valuation']:.2f} (adjustment: {adj['total_adjustment']:.2f})")
        
        except Exception as e:
            logger.error(f"Error running valuation for {stock_id}: {e}")
    
    return True

if __name__ == "__main__":
    logger.info("Starting industry model training from database")
    
    success = train_industry_models_from_db()
    
    if success:
        logger.info("Testing valuations with trained models")
        test_valuation_with_trained_models()
    
    logger.info("Process complete")
