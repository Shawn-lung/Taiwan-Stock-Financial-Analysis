import sqlite3
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CollectionMonitor:
    """Monitor the progress of the background data collection."""
    
    def __init__(self, db_path="finance_data.db"):
        """Initialize the collection monitor.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        
        if not os.path.exists(db_path):
            logger.error(f"Database file not found: {db_path}")
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        logger.info(f"Connected to database: {db_path}")
    
    def get_collection_progress(self):
        """Get overall collection progress statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get total number of stocks
            total_stocks = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM stock_info", 
                conn
            ).iloc[0]['count']
            
            # Get stocks with different data types
            collection_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(DISTINCT CASE WHEN data_type = 'financial_statement' AND status = 'success' THEN stock_id END) as fs_count,
                    COUNT(DISTINCT CASE WHEN data_type = 'balance_sheet' AND status = 'success' THEN stock_id END) as bs_count,
                    COUNT(DISTINCT CASE WHEN data_type = 'cash_flow' AND status = 'success' THEN stock_id END) as cf_count,
                    COUNT(DISTINCT CASE WHEN data_type = 'price_data' AND status = 'success' THEN stock_id END) as price_count
                FROM collection_log
            """, conn)
            
            # Get stocks with complete data (all four data types)
            complete_stocks = pd.read_sql_query("""
                SELECT COUNT(*) as count
                FROM (
                    SELECT stock_id
                    FROM collection_log
                    WHERE status = 'success'
                    GROUP BY stock_id
                    HAVING 
                        SUM(CASE WHEN data_type = 'financial_statement' THEN 1 ELSE 0 END) > 0 AND
                        SUM(CASE WHEN data_type = 'balance_sheet' THEN 1 ELSE 0 END) > 0 AND
                        SUM(CASE WHEN data_type = 'cash_flow' THEN 1 ELSE 0 END) > 0 AND
                        SUM(CASE WHEN data_type = 'price_data' THEN 1 ELSE 0 END) > 0
                ) as complete
            """, conn).iloc[0]['count']
            
            # Get collection rate over time
            collection_rate = pd.read_sql_query("""
                SELECT 
                    date(timestamp) as collection_date,
                    COUNT(*) as total_attempts,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_attempts,
                    COUNT(DISTINCT stock_id) as unique_stocks
                FROM collection_log
                GROUP BY date(timestamp)
                ORDER BY collection_date
            """, conn)
            
            # Get errors by type
            error_types = pd.read_sql_query("""
                SELECT 
                    data_type,
                    COUNT(*) as error_count
                FROM collection_log
                WHERE status = 'error'
                GROUP BY data_type
                ORDER BY error_count DESC
            """, conn)
            
            # Get recently collected stocks
            recent_collections = pd.read_sql_query("""
                SELECT 
                    cl.stock_id,
                    si.stock_name,
                    si.industry,
                    MAX(cl.timestamp) as last_collected,
                    COUNT(DISTINCT cl.data_type) as data_types_collected
                FROM collection_log cl
                JOIN stock_info si ON cl.stock_id = si.stock_id
                WHERE cl.status = 'success'
                GROUP BY cl.stock_id
                ORDER BY last_collected DESC
                LIMIT 10
            """, conn)
            
            conn.close()
            
            # Calculate completion percentage
            completion = {
                'financial_statements': round((collection_stats.iloc[0]['fs_count'] / total_stocks) * 100, 1),
                'balance_sheets': round((collection_stats.iloc[0]['bs_count'] / total_stocks) * 100, 1),
                'cash_flows': round((collection_stats.iloc[0]['cf_count'] / total_stocks) * 100, 1), 
                'price_data': round((collection_stats.iloc[0]['price_count'] / total_stocks) * 100, 1),
                'complete_stocks': round((complete_stocks / total_stocks) * 100, 1)
            }
            
            # Prepare result
            result = {
                'total_stocks': total_stocks,
                'complete_stocks': complete_stocks,
                'completion_percentage': completion,
                'collection_rate': collection_rate,
                'error_types': error_types,
                'recent_collections': recent_collections
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting collection progress: {e}")
            return None
    
    def get_industry_coverage(self):
        """Get data collection coverage by industry."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get industry coverage
            industry_coverage = pd.read_sql_query("""
                SELECT 
                    si.industry,
                    COUNT(DISTINCT si.stock_id) as total_stocks,
                    COUNT(DISTINCT CASE WHEN cl.status = 'success' AND cl.data_type = 'financial_statement' THEN cl.stock_id END) as fs_count,
                    COUNT(DISTINCT CASE WHEN cl.status = 'success' AND cl.data_type = 'balance_sheet' THEN cl.stock_id END) as bs_count,
                    COUNT(DISTINCT CASE WHEN cl.status = 'success' AND cl.data_type = 'cash_flow' THEN cl.stock_id END) as cf_count,
                    COUNT(DISTINCT CASE WHEN cl.status = 'success' AND cl.data_type = 'price_data' THEN cl.stock_id END) as price_count
                FROM stock_info si
                LEFT JOIN collection_log cl ON si.stock_id = cl.stock_id
                GROUP BY si.industry
                ORDER BY total_stocks DESC
            """, conn)
            
            conn.close()
            
            # Calculate completion percentages
            for col in ['fs_count', 'bs_count', 'cf_count', 'price_count']:
                industry_coverage[f'{col}_pct'] = (industry_coverage[col] / industry_coverage['total_stocks']) * 100
            
            return industry_coverage
            
        except Exception as e:
            logger.error(f"Error getting industry coverage: {e}")
            return None
    
    def get_top_stocks_data(self, industry=None, top_n=5):
        """Get detailed information about top stocks (by data completeness)."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query
            query = """
                SELECT 
                    si.stock_id,
                    si.stock_name,
                    si.industry,
                    MAX(CASE WHEN cl.data_type = 'financial_statement' AND cl.status = 'success' THEN cl.timestamp ELSE NULL END) as fs_last_update,
                    MAX(CASE WHEN cl.data_type = 'balance_sheet' AND cl.status = 'success' THEN cl.timestamp ELSE NULL END) as bs_last_update,
                    MAX(CASE WHEN cl.data_type = 'cash_flow' AND cl.status = 'success' THEN cl.timestamp ELSE NULL END) as cf_last_update,
                    MAX(CASE WHEN cl.data_type = 'price_data' AND cl.status = 'success' THEN cl.timestamp ELSE NULL END) as price_last_update,
                    (CASE 
                        WHEN MAX(CASE WHEN cl.data_type = 'financial_statement' AND cl.status = 'success' THEN 1 ELSE 0 END) +
                             MAX(CASE WHEN cl.data_type = 'balance_sheet' AND cl.status = 'success' THEN 1 ELSE 0 END) +
                             MAX(CASE WHEN cl.data_type = 'cash_flow' AND cl.status = 'success' THEN 1 ELSE 0 END) +
                             MAX(CASE WHEN cl.data_type = 'price_data' AND cl.status = 'success' THEN 1 ELSE 0 END) = 4 THEN 'Complete'
                        ELSE 'Partial'
                    END) as completeness,
                    COUNT(DISTINCT CASE WHEN cl.status = 'success' THEN cl.data_type END) as data_types_collected
                FROM stock_info si
                LEFT JOIN collection_log cl ON si.stock_id = cl.stock_id
            """
            
            # Add industry filter if specified
            if industry:
                query += f" WHERE si.industry = '{industry}'"
            
            query += """
                GROUP BY si.stock_id
                ORDER BY data_types_collected DESC, fs_last_update DESC
                LIMIT ?
            """
            
            top_stocks = pd.read_sql_query(query, conn, params=(top_n,))
            
            conn.close()
            return top_stocks
            
        except Exception as e:
            logger.error(f"Error getting top stocks data: {e}")
            return None
    
    def sample_financial_data(self, stock_id):
        """Get a sample of financial data for a specific stock."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get financial statements
            financial_statements = pd.read_sql_query(
                "SELECT date, metric_type, value FROM financial_statements WHERE stock_id = ? ORDER BY date DESC LIMIT 20",
                conn,
                params=(stock_id,)
            )
            
            # Get balance sheets
            balance_sheets = pd.read_sql_query(
                "SELECT date, metric_type, value FROM balance_sheets WHERE stock_id = ? ORDER BY date DESC LIMIT 20",
                conn,
                params=(stock_id,)
            )
            
            # Get cash flows
            cash_flows = pd.read_sql_query(
                "SELECT date, metric_type, value FROM cash_flows WHERE stock_id = ? ORDER BY date DESC LIMIT 20",
                conn,
                params=(stock_id,)
            )
            
            # Get price data
            price_data = pd.read_sql_query(
                "SELECT date, open, close, volume FROM stock_prices WHERE stock_id = ? ORDER BY date DESC LIMIT 30",
                conn,
                params=(stock_id,)
            )
            
            conn.close()
            
            return {
                'financial_statements': financial_statements,
                'balance_sheets': balance_sheets,
                'cash_flows': cash_flows,
                'price_data': price_data
            }
            
        except Exception as e:
            logger.error(f"Error getting sample financial data: {e}")
            return None
    
    def plot_collection_progress(self):
        """Plot collection progress over time."""
        try:
            progress = self.get_collection_progress()
            if not progress or 'collection_rate' not in progress:
                logger.error("Failed to get collection rate data for plotting")
                return
            
            collection_rate = progress['collection_rate']
            collection_rate['success_rate'] = (collection_rate['successful_attempts'] / collection_rate['total_attempts']) * 100
            
            plt.figure(figsize=(14, 8))
            
            # Plot 1: Collection attempts by day
            plt.subplot(2, 1, 1)
            sns.barplot(x='collection_date', y='total_attempts', data=collection_rate, color='skyblue', label='Total')
            sns.barplot(x='collection_date', y='successful_attempts', data=collection_rate, color='green', label='Success')
            plt.title('Daily Collection Attempts')
            plt.ylabel('Number of Attempts')
            plt.xlabel('')
            plt.legend()
            plt.xticks(rotation=45)
            
            # Plot 2: Success rate by day
            plt.subplot(2, 1, 2)
            sns.lineplot(x='collection_date', y='success_rate', data=collection_rate, marker='o', color='green')
            plt.title('Daily Success Rate')
            plt.ylabel('Success Rate (%)')
            plt.xlabel('Date')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return plt
            
        except Exception as e:
            logger.error(f"Error plotting collection progress: {e}")
            return None
    
    def plot_industry_coverage(self):
        """Plot data collection coverage by industry."""
        try:
            industry_coverage = self.get_industry_coverage()
            if industry_coverage is None or industry_coverage.empty:
                logger.error("Failed to get industry coverage data for plotting")
                return
            
            # Filter to top 10 industries by stock count
            top_industries = industry_coverage.nlargest(10, 'total_stocks')
            
            # Create plot
            plt.figure(figsize=(14, 8))
            
            # Plot coverage percentages
            ind = np.arange(len(top_industries))
            width = 0.2
            
            plt.bar(ind - width*1.5, top_industries['fs_count_pct'], width, label='Financial Statements')
            plt.bar(ind - width/2, top_industries['bs_count_pct'], width, label='Balance Sheets')
            plt.bar(ind + width/2, top_industries['cf_count_pct'], width, label='Cash Flows')
            plt.bar(ind + width*1.5, top_industries['price_count_pct'], width, label='Price Data')
            
            plt.xlabel('Industry')
            plt.ylabel('Coverage (%)')
            plt.title('Data Collection Coverage by Industry (Top 10)')
            plt.xticks(ind, top_industries['industry'], rotation=45, ha='right')
            plt.legend(loc='best')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            return plt
            
        except Exception as e:
            logger.error(f"Error plotting industry coverage: {e}")
            return None

    def generate_summary_report(self):
        """Generate a comprehensive summary report of data collection."""
        try:
            progress = self.get_collection_progress()
            industry_coverage = self.get_industry_coverage()
            
            if not progress or not industry_coverage is not None:
                logger.error("Failed to get data for summary report")
                return
            
            # Print text summary
            print("\n" + "="*60)
            print(" DATA COLLECTION SUMMARY REPORT ")
            print("="*60)
            
            print(f"\nTotal Stocks in Database: {progress['total_stocks']}")
            print(f"Stocks with Complete Data: {progress['complete_stocks']} ({progress['completion_percentage']['complete_stocks']}%)")
            
            print("\nData Type Completion:")
            print(f"  Financial Statements: {progress['completion_percentage']['financial_statements']}%")
            print(f"  Balance Sheets: {progress['completion_percentage']['balance_sheets']}%")
            print(f"  Cash Flows: {progress['completion_percentage']['cash_flows']}%")
            print(f"  Price Data: {progress['completion_percentage']['price_data']}%")
            
            print("\nRecently Collected Stocks:")
            if 'recent_collections' in progress and not progress['recent_collections'].empty:
                for _, row in progress['recent_collections'].iterrows():
                    print(f"  {row['stock_id']} ({row['stock_name']}): {row['data_types_collected']} data types, last collected {row['last_collected']}")
            
            print("\nTop 5 Industries by Coverage:")
            if not industry_coverage.empty:
                top5 = industry_coverage.nlargest(5, 'total_stocks')
                for _, row in top5.iterrows():
                    avg_coverage = (row['fs_count_pct'] + row['bs_count_pct'] + row['cf_count_pct'] + row['price_count_pct']) / 4
                    print(f"  {row['industry']}: {row['total_stocks']} stocks, {avg_coverage:.1f}% avg coverage")
            
            print("\nData Collection Rate:")
            if 'collection_rate' in progress and not progress['collection_rate'].empty:
                latest = progress['collection_rate'].iloc[-1]
                print(f"  Latest day ({latest['collection_date']}): {latest['total_attempts']} attempts, {latest['successful_attempts']} successful ({(latest['successful_attempts']/latest['total_attempts']*100):.1f}%)")
                
                # Calculate overall totals
                total_attempts = progress['collection_rate']['total_attempts'].sum()
                successful = progress['collection_rate']['successful_attempts'].sum()
                success_rate = (successful / total_attempts) * 100 if total_attempts > 0 else 0
                print(f"  Overall: {total_attempts} attempts, {successful} successful ({success_rate:.1f}%)")
            
            print("\n" + "="*60)
            
            # Generate plots
            collection_plot = self.plot_collection_progress()
            if collection_plot:
                collection_plot.savefig('collection_progress.png')
                print("Collection progress plot saved as 'collection_progress.png'")
            
            coverage_plot = self.plot_industry_coverage()
            if coverage_plot:
                coverage_plot.savefig('industry_coverage.png')
                print("Industry coverage plot saved as 'industry_coverage.png'")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return False

# Example usage
if __name__ == "__main__":
    monitor = CollectionMonitor()
    
    print("\nChecking collection progress...")
    monitor.generate_summary_report()
    
    # Check specific industries
    print("\nChecking semiconductors industry...")
    semiconductor_stocks = monitor.get_top_stocks_data(industry="Semiconductors", top_n=5)
    if semiconductor_stocks is not None and not semiconductor_stocks.empty:
        print(f"\nTop 5 semiconductor stocks with most complete data:")
        print(semiconductor_stocks[['stock_id', 'stock_name', 'completeness', 'data_types_collected']])
        
        # If there are complete stocks, show sample data for the first one
        complete_stocks = semiconductor_stocks[semiconductor_stocks['completeness'] == 'Complete']
        if not complete_stocks.empty:
            sample_stock = complete_stocks.iloc[0]['stock_id']
            print(f"\nSample financial data for {sample_stock}:")
            sample_data = monitor.sample_financial_data(sample_stock)
            if sample_data:
                # Show financial statement sample
                if not sample_data['financial_statements'].empty:
                    print("\nFinancial Statements Sample:")
                    print(sample_data['financial_statements'].head(5))
                
                # Show price data sample
                if not sample_data['price_data'].empty:
                    print("\nPrice Data Sample:")
                    print(sample_data['price_data'].head(5))
