import os
import logging
from industry_valuation_model import IndustryValuationModel
from industry_data_collector import TaiwanIndustryDataCollector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_industry_models():
    """Generate and train industry models for later use in the app."""
    # Specify key industries to focus on
    priority_industries = [
        "Semiconductors",
        "Electronics Manufacturing",
        "Computer Hardware",
        "Banking",
        "Telecommunications",
        "Financial Services",
        "Food & Beverage",
        "Retail",
        "Healthcare",
        "Utilities"
    ]
    
    # Create data directory if it doesn't exist
    os.makedirs("industry_data", exist_ok=True)
    os.makedirs("industry_data/models", exist_ok=True)
    
    # Initialize collector
    collector = TaiwanIndustryDataCollector(
        lookback_years=5,
        rate_limit_delay=1.5,
        max_retries=3
    )
    
    # Check if we have data already
    industry_data_path = os.path.join("industry_data", "taiwan_industry_financial_data.pkl")
    if not os.path.exists(industry_data_path):
        logger.info("No existing industry data found. Collecting data...")
        
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
    else:
        logger.info("Using existing industry data")
    
    # Initialize model
    logger.info("Initializing industry valuation model...")
    model = IndustryValuationModel(background_collector=collector)
    
    # Train models for all available industries
    logger.info("Training industry models...")
    training_results = model.train_industry_models(force_retrain=False)
    
    # Log results
    logger.info(f"Successfully trained {len(training_results)} industry models")
    for industry, results in training_results.items():
        if 'samples' in results:
            logger.info(f"Industry: {industry}, Samples: {results['samples']}")
    
    logger.info("Pre-training complete. Models are ready for use in the app.")

if __name__ == "__main__":
    generate_industry_models()
