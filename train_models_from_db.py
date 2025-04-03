import os
import logging
import time
from background_data_collector import BackgroundDataCollector
from industry_valuation_model import IndustryValuationModel
from dcf_integrator import IntegratedValuationModel
import pandas as pd

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

def check_training_data():
    """Check if training data files exist and are valid."""
    data_dir = "industry_data_from_db"
    if not os.path.exists(data_dir):
        logger.error(f"Directory {data_dir} not found. Please run debug_train_from_db.py first.")
        return []
    
    training_files = [f for f in os.listdir(data_dir) if f.endswith('_training.csv')]
    if not training_files:
        logger.error(f"No training data files found in {data_dir}. Please run debug_train_from_db.py first.")
        return []
        
    valid_industries = []
    for file in training_files:
        industry = file.replace('_training.csv', '').replace('_', ' ')
        try:
            df = pd.read_csv(os.path.join(data_dir, file))
            if len(df) > 0:
                logger.info(f"Found valid training data for {industry} with {len(df)} records")
                valid_industries.append(industry)
            else:
                logger.warning(f"Training file for {industry} exists but contains no records")
        except Exception as e:
            logger.warning(f"Error reading training file {file}: {e}")
    
    return valid_industries

def train_industry_models_from_db():
    """Train industry valuation models using data from background collector."""
    
    # Check for existing training data first
    valid_industries = check_training_data()
    
    if not valid_industries:
        # Initialize the background data collector
        collector = BackgroundDataCollector(
            db_path="finance_data.db",
            collection_interval=1  # Collect every hour instead of 12
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
    else:
        # No need for collector when using existing data files
        collector = None
    
    # Initialize the industry model with the background collector and correct data directory
    model = IndustryValuationModel(
        data_dir="industry_data_from_db",
        background_collector=collector
    )
    
    # Prepare data and train models - target only valid industries if we have them
    logger.info("Training industry models from database data...")
    results = model.train_industry_models(industries=valid_industries, force_retrain=True)
    
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
    # Find trained models
    model_dir = "industry_data_from_db/models"
    if not os.path.exists(model_dir):
        logger.error("No models directory found. Models may not be trained yet.")
        return False
    
    # Get list of available models
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
    if not model_files:
        logger.error("No trained models found. Run training first.")
        return False
        
    # Extract industries from model filenames
    available_industries = [f.replace('_model.keras', '').replace('_', ' ') for f in model_files]
    logger.info(f"Found trained models for {len(available_industries)} industries: {available_industries}")
    
    # Initialize the integrated valuation model (which uses industry model)
    valuation_model = IntegratedValuationModel(
        use_ml=True, 
        use_dl=True, 
        use_industry=True
    )
    
    # Test stocks for each available industry
    # These are example Taiwan stock IDs for different industries
    test_stocks = {
        'semiconductors': ['2330.TW', '2454.TW'],
        'electronics': ['2317.TW', '2354.TW'],
        'computer hardware': ['2353.TW', '2357.TW'],
        'telecommunications': ['3045.TW', '4904.TW'],
        'financial services': ['2812.TW', '2880.TW']
    }
    
    # Run valuations for available industries
    for industry in available_industries:
        lower_industry = industry.lower()
        # Find test stocks for this industry
        stocks = test_stocks.get(lower_industry)
        if not stocks:
            logger.warning(f"No test stocks defined for {industry}")
            continue
            
        # Test with the first available stock
        stock_id = stocks[0]
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
