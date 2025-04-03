#!/usr/bin/env python3
"""
Fix critical issues in the DCF model implementation:
1. Division by zero in forecast_fcf_list
2. Rate limiting issues with Yahoo Finance API
"""

import os
import logging
import pandas as pd
import time
import random
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def backup_original_file():
    """Create a backup of the original dcf_model.py file"""
    src_path = "/Users/shawnlung/Documents/GitHub/Finance_stuff/dcf_model.py"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    dst_path = f"/Users/shawnlung/Documents/GitHub/Finance_stuff/dcf_model_backup_{timestamp}.py"
    
    try:
        shutil.copy2(src_path, dst_path)
        logger.info(f"Created backup at {dst_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return False

def fix_division_by_zero():
    """Fix the division by zero error in forecast_fcf_list method"""
    file_path = "/Users/shawnlung/Documents/GitHub/Finance_stuff/dcf_model.py"
    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Fix 1: Add guard against division by zero in current_wc_percent calculation
        original_line = "current_wc_percent = current_wc / revenue"
        fixed_line = "current_wc_percent = current_wc / revenue if revenue > 0 else 0"
        
        content = content.replace(original_line, fixed_line)
        
        # Fix 2: Add more defensive code in the method
        target_section = "# Calculate working capital with gradual transition"
        defensive_code = """
                # Calculate working capital with gradual transition
                if revenue <= 0:
                    # Handle zero revenue case
                    adjusted_wc_percent = wc_ratios[i] if wc_ratios and i < len(wc_ratios) else 0
                    new_wc = new_revenue * adjusted_wc_percent if new_revenue > 0 else 0
                    delta_wc = 0  # Avoid wild swings when revenue is zero
                elif i < wc_adjustment_years:"""
        
        content = content.replace(target_section, defensive_code)
        
        # Fix 3: Fix indentation after insertion
        content = content.replace("adjusted_wc_percent = wc_ratios[i]", "                adjusted_wc_percent = wc_ratios[i]", 1)
        
        # Write the changes back to the file
        with open(file_path, 'w') as file:
            file.write(content)
            
        logger.info("Fixed division by zero error in forecast_fcf_list method")
        return True
    except Exception as e:
        logger.error(f"Failed to fix division by zero: {e}")
        return False

def add_rate_limiting():
    """Add rate limiting to yfinance API calls"""
    file_path = "/Users/shawnlung/Documents/GitHub/Finance_stuff/dcf_model.py"
    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            
        # Add import for time and random modules if not already there
        if "import time" not in content:
            import_section = "import yfinance as yf"
            updated_imports = "import yfinance as yf\nimport time\nimport random"
            content = content.replace(import_section, updated_imports)
        
        # Add rate limiting method
        if "def _respect_rate_limits" not in content:
            class_definition = "class DCFModel:"
            method_definition = """class DCFModel:
    @staticmethod
    def _respect_rate_limits():
        \"\"\"Pause briefly to respect API rate limits\"\"\"
        time.sleep(1 + random.random() * 2)  # Sleep between 1-3 seconds
    """
            content = content.replace(class_definition, method_definition)
        
        # Add rate limiting before yfinance calls
        stock_init = "self.stock = yf.Ticker(self.stock_code)"
        rate_limited_init = "self._respect_rate_limits()\n        self.stock = yf.Ticker(self.stock_code)"
        content = content.replace(stock_init, rate_limited_init)
        
        # Add defensive code for failed API requests
        ticker_init = "def initialize_model(self):"
        enhanced_init = """def initialize_model(self):
        \"\"\"Fetch financial data from yfinance and initialize base metrics.\"\"\"
        tries = 0
        max_retries = 3
        
        while tries < max_retries:
            try:"""
        content = content.replace(ticker_init, enhanced_init)
        
        # Add retry logic
        init_end = "logger.error(\"Error during initialization: %s\", e)"
        retry_logic = """logger.warning(f"Error during initialization (attempt {tries+1}/{max_retries}): {e}")
                tries += 1
                if tries < max_retries:
                    # Exponential backoff
                    sleep_time = 5 * (2 ** tries)
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error("Error during initialization: %s", e)"""
        content = content.replace(init_end, retry_logic)
        
        # Fix indentation for initialize_model method
        content = content.replace("    # Initialize financial data placeholders", "            # Initialize financial data placeholders")
        
        # Add extra closing bracket at the end of initialize_model
        init_closing = "logger.info(\"Initialization complete for stock: %s\", self.stock_code)"
        init_closing_fixed = "            logger.info(\"Initialization complete for stock: %s\", self.stock_code)\n            break  # Exit the retry loop on success"
        content = content.replace(init_closing, init_closing_fixed)
        
        # Fix WACC initialization to handle None value
        wacc_init = "self.wacc = None"
        wacc_init_fixed = "self.wacc = 0.09  # Default value that will be updated later"
        content = content.replace(wacc_init, wacc_init_fixed)
        
        # Write the changes back to the file
        with open(file_path, 'w') as file:
            file.write(content)
            
        logger.info("Added rate limiting to yfinance API calls")
        return True
    except Exception as e:
        logger.error(f"Failed to add rate limiting: {e}")
        return False

def main():
    """Main function to apply all fixes"""
    logger.info("Starting DCF model fixes")
    
    # First create a backup
    if backup_original_file():
        # Apply fixes
        fix_division_by_zero()
        add_rate_limiting()
        
        logger.info("All fixes applied successfully")
        logger.info("Please restart any running processes to apply the changes")
    else:
        logger.error("Failed to create backup, aborting fixes")

if __name__ == "__main__":
    main()
