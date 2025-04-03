#!/usr/bin/env python3
"""
Patch IndustryValuationModel to use database data instead of yfinance API calls.
"""

import os
import re
import sys
import logging
import shutil

# Add parent directory to path to import util
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_directory_exists():
    """Create util directory if it doesn't exist."""
    util_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'util')
    os.makedirs(util_dir, exist_ok=True)
    
    # Create __init__.py in util directory if it doesn't exist
    init_file = os.path.join(util_dir, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('# Utility modules')
        logger.info(f"Created {init_file}")

def backup_file(filepath):
    """Create a backup of the file."""
    backup_path = f"{filepath}.bak"
    if not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
        logger.info(f"Created backup at {backup_path}")
    return backup_path

def patch_industry_valuation_model():
    """Patch industry_valuation_model.py to use database instead of yfinance."""
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          "industry_valuation_model.py")
    
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return False
    
    # Create backup
    backup_file(filepath)
    
    try:
        with open(filepath, 'r') as file:
            content = file.read()
        
        # Add import for DBFinancialDataProvider
        import_section = "import logging"
        new_import = "import logging\nfrom util.db_data_provider import DBFinancialDataProvider"
        content = content.replace(import_section, new_import)
        
        # Add DB data provider initialization in __init__
        init_method = "def __init__(self, data_dir: str = \"industry_data\", background_collector = None):"
        init_with_db = """def __init__(self, data_dir: str = "industry_data", background_collector = None, db_path: str = "finance_data.db"):
        \"\"\"Initialize the industry valuation model.
        
        Args:
            data_dir: Directory containing industry data
            background_collector: Optional BackgroundDataCollector instance
            db_path: Path to the SQLite database file
        \"\"\"
        self.db_path = db_path
        self.db_provider = DBFinancialDataProvider(db_path)"""
        
        content = content.replace(init_method, init_with_db)
        
        # Write the modified content back
        with open(filepath, 'w') as file:
            file.write(content)
        
        logger.info(f"Successfully patched {filepath} to use database provider")
        return True
        
    except Exception as e:
        logger.error(f"Error patching {filepath}: {e}")
        return False

def patch_dcf_integrator():
    """Patch dcf_integrator.py to use database instead of yfinance."""
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          "dcf_integrator.py")
    
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return False
    
    # Create backup
    backup_file(filepath)
    
    try:
        with open(filepath, 'r') as file:
            content = file.read()
        
        # Add import for DBFinancialDataProvider
        import_section = "import logging"
        new_import = "import logging\nfrom util.db_data_provider import DBFinancialDataProvider"
        content = content.replace(import_section, new_import)
        
        # Add DB data provider initialization in __init__
        init_method = "def __init__(self, use_ml: bool = True, use_dl: bool = True, use_industry: bool = True):"
        init_with_db = """def __init__(self, use_ml: bool = True, use_dl: bool = True, use_industry: bool = True, db_path: str = "finance_data.db"):
        \"\"\"Initialize the integrated valuation model.
        
        Args:
            use_ml: Whether to use ML predictions for growth factors
            use_dl: Whether to use deep learning predictions
            use_industry: Whether to apply industry-specific adjustments
            db_path: Path to the SQLite database file
        \"\"\"
        self.use_ml = use_ml
        self.use_dl = use_dl
        self.use_industry = use_industry
        self.db_path = db_path
        self.db_provider = DBFinancialDataProvider(db_path)"""
        
        content = content.replace(init_method, init_with_db)
        
        # Write the modified content back
        with open(filepath, 'w') as file:
            file.write(content)
        
        logger.info(f"Successfully patched {filepath} to use database provider")
        return True
        
    except Exception as e:
        logger.error(f"Error patching {filepath}: {e}")
        return False

def main():
    """Apply all patches."""
    logger.info("Starting patches to use database instead of yfinance")
    
    # Ensure util directory exists
    ensure_directory_exists()
    
    # Patch industry valuation model
    if patch_industry_valuation_model():
        logger.info("Successfully patched industry valuation model")
    
    # Patch DCF integrator
    if patch_dcf_integrator():
        logger.info("Successfully patched DCF integrator")
    
    logger.info("Completed all patches")

if __name__ == "__main__":
    main()
