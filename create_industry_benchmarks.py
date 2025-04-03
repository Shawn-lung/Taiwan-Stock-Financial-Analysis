#!/usr/bin/env python3
"""
Create industry benchmarks file from trained data.
"""

import os
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_industry_benchmarks(data_dir="industry_data_from_db", output_dir=None):
    """
    Create industry benchmarks file from training datasets.
    """
    if output_dir is None:
        output_dir = data_dir
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Directory {data_dir} not found!")
        return False
    
    # Find training CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_training.csv')]
    
    if not csv_files:
        logger.error(f"No training CSV files found in {data_dir}")
        return False
    
    logger.info(f"Found {len(csv_files)} training data files")
    
    # Process each industry file
    industry_metrics = []
    
    for csv_file in csv_files:
        industry = csv_file.replace('_training.csv', '').replace('_', ' ')
        file_path = os.path.join(data_dir, csv_file)
        
        try:
            # Read file
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"{csv_file} is empty, skipping")
                continue
            
            # Calculate metrics
            metrics = {
                'industry': industry,
                'stock_count': df['stock_id'].nunique() if 'stock_id' in df.columns else 0,
                'record_count': len(df)
            }
            
            # Calculate median values for key metrics
            for metric in ['historical_growth_mean', 'operating_margin', 'net_margin', 
                         'roa', 'roe', 'debt_to_equity']:
                if metric in df.columns:
                    # Calculate median
                    median_val = df[metric].median()
                    if pd.notna(median_val):
                        metrics[f'{metric}_median'] = median_val
                    else:
                        metrics[f'{metric}_median'] = 0.0  # Default value
                    
                    # Calculate mean
                    mean_val = df[metric].mean()
                    if pd.notna(mean_val):
                        metrics[f'{metric}_mean'] = mean_val
                    else:
                        metrics[f'{metric}_mean'] = 0.0  # Default value
            
            industry_metrics.append(metrics)
            logger.info(f"Processed benchmarks for {industry}: {len(df)} records")
            
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
    
    # Create DataFrame from metrics
    if not industry_metrics:
        # Create a default benchmarks DataFrame with minimum required structure
        default_industries = ["Semiconductors", "Electronics", "Computer Hardware", 
                            "Financial Services", "Telecommunications"]
        
        industry_metrics = []
        for ind in default_industries:
            metrics = {
                'industry': ind,
                'stock_count': 5,
                'record_count': 100,
                'historical_growth_mean_median': 0.10,
                'operating_margin_median': 0.15,
                'net_margin_median': 0.08,
                'roa_median': 0.05,
                'roe_median': 0.12,
                'debt_to_equity_median': 0.5
            }
            industry_metrics.append(metrics)
        
        logger.warning("Created default industry benchmarks as no valid data was found")
    
    # Create benchmarks DataFrame
    benchmarks_df = pd.DataFrame(industry_metrics)
    
    # Save to both directories to ensure it's found
    benchmark_file = os.path.join(data_dir, 'industry_benchmarks.csv')
    benchmarks_df.to_csv(benchmark_file, index=False)
    logger.info(f"Saved industry benchmarks to {benchmark_file}")
    
    if output_dir != data_dir:
        benchmark_file2 = os.path.join(output_dir, 'industry_benchmarks.csv')
        benchmarks_df.to_csv(benchmark_file2, index=False)
        logger.info(f"Also saved industry benchmarks to {benchmark_file2}")
    
    # Also save to original industry_data directory
    orig_dir = "industry_data"
    if os.path.exists(orig_dir):
        os.makedirs(orig_dir, exist_ok=True)
        benchmark_file3 = os.path.join(orig_dir, 'industry_benchmarks.csv')
        benchmarks_df.to_csv(benchmark_file3, index=False)
        logger.info(f"Also saved industry benchmarks to {benchmark_file3}")
    
    return True

if __name__ == "__main__":
    logger.info("Creating industry benchmarks...")
    
    # Create benchmarks in both directories
    create_industry_benchmarks("industry_data_from_db", "industry_data")
    
    logger.info("Benchmark creation complete!")
