#!/usr/bin/env python3
"""
Verify training data files for model training.
"""

import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_training_data(data_dir="industry_data_from_db"):
    """Verify that training data exists and is properly formatted."""
    # Check if directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Directory {data_dir} not found!")
        return False
    
    # Check for training CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_training.csv')]
    
    if not csv_files:
        logger.error(f"No training CSV files found in {data_dir}")
        return False
    
    logger.info(f"Found {len(csv_files)} training data files")
    
    # Check each file
    valid_files = []
    for csv_file in csv_files:
        industry = csv_file.replace('_training.csv', '').replace('_', ' ')
        file_path = os.path.join(data_dir, csv_file)
        
        try:
            # Read file
            df = pd.read_csv(file_path)
            
            # Check if file has data
            if df.empty:
                logger.warning(f"{csv_file} is empty")
                continue
                
            # Check required columns
            required_columns = ['revenue', 'future_6m_return']
            useful_columns = ['operating_margin', 'net_margin', 'roa', 'roe', 'historical_growth_mean']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"{csv_file} is missing required columns: {missing_columns}")
                continue
            
            present_useful = [col for col in useful_columns if col in df.columns]
            
            # Check model directory exists
            model_dir = os.path.join(data_dir, "models")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                logger.info(f"Created models directory: {model_dir}")
            
            logger.info(f"✓ {csv_file}: {len(df)} records, {len(df.columns)} columns")
            logger.info(f"  Key metrics available: {present_useful}")
            valid_files.append(csv_file)
            
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
    
    if not valid_files:
        logger.error("No valid training files found!")
        return False
        
    logger.info(f"Found {len(valid_files)} valid training files")
    return True

def main():
    """Main function"""
    logger.info("Verifying training data...")
    if verify_training_data():
        logger.info("Training data verified successfully!")
        logger.info("You can now run train_models_from_db.py to train models.")
    else:
        logger.error("Training data verification failed!")
        logger.info("Run debug_train_from_db.py to generate training data first.")

if __name__ == "__main__":
    main()
