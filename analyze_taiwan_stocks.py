import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaiwanStockAnalyzer:
    """Analyze Taiwan stock data that has been collected in the database."""
    
    def __init__(self, db_path="finance_data.db"):
        """Initialize the Taiwan stock analyzer.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        
        if not os.path.exists(db_path):
            logger.error(f"Database file not found: {db_path}")
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        logger.info(f"Connected to database: {db_path}")
    
    def get_complete_stocks(self, industry=None, min_years=2):
        """Get a list of stocks with complete data (all data types available).
        
        Args:
            industry: Optional industry to filter stocks
            min_years: Minimum years of financial data required
            
        Returns:
            DataFrame with complete stocks information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query
            query = """
                SELECT 
                    si.stock_id,
                    si.stock_name,
                    si.industry,
                    COUNT(DISTINCT fs.date) as fs_years
                FROM stock_info si
                INNER JOIN (
                    -- Ensure the stock has all four data types
                    SELECT stock_id
                    FROM collection_log
                    WHERE status = 'success'
                    GROUP BY stock_id
                    HAVING 
                        SUM(CASE WHEN data_type = 'financial_statement' THEN 1 ELSE 0 END) > 0 AND
                        SUM(CASE WHEN data_type = 'balance_sheet' THEN 1 ELSE 0 END) > 0 AND
                        SUM(CASE WHEN data_type = 'cash_flow' THEN 1 ELSE 0 END) > 0 AND
                        SUM(CASE WHEN data_type = 'price_data' THEN 1 ELSE 0 END) > 0
                ) complete ON si.stock_id = complete.stock_id
                INNER JOIN financial_statements fs ON si.stock_id = fs.stock_id
                WHERE metric_type = 'Revenue' OR metric_type = 'OperatingRevenue'
            """
            
            # Add industry filter if specified
            if industry:
                query += f" AND si.industry = '{industry}'"
            
            query += """
                GROUP BY si.stock_id
                HAVING fs_years >= ?
                ORDER BY si.industry, si.stock_id
            """
            
            complete_stocks = pd.read_sql_query(query, conn, params=(min_years,))
            
            conn.close()
            
            logger.info(f"Found {len(complete_stocks)} stocks with complete data")
            return complete_stocks
            
        except Exception as e:
            logger.error(f"Error getting complete stocks: {e}")
            return pd.DataFrame()
    
    def calculate_basic_metrics(self, stock_id):
        """Calculate basic financial metrics for a stock.
        
        Args:
            stock_id: The stock ID to calculate metrics for
            
        Returns:
            DataFrame with financial metrics by year
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get financial statements
            financials = pd.read_sql_query(
                "SELECT date, metric_type, value FROM financial_statements WHERE stock_id = ? ORDER BY date",
                conn,
                params=(stock_id,)
            )
            
            # Get balance sheets
            balance = pd.read_sql_query(
                "SELECT date, metric_type, value FROM balance_sheets WHERE stock_id = ? ORDER BY date",
                conn,
                params=(stock_id,)
            )
            
            # Get cash flows
            cashflow = pd.read_sql_query(
                "SELECT date, metric_type, value FROM cash_flows WHERE stock_id = ? ORDER BY date",
                conn,
                params=(stock_id,)
            )
            
            conn.close()
            
            # Check if we have enough data
            if financials.empty or balance.empty:
                logger.warning(f"Insufficient data for {stock_id}")
                return pd.DataFrame()
            
            # Group by year (use report date's year)
            financials['year'] = pd.to_datetime(financials['date']).dt.year
            balance['year'] = pd.to_datetime(balance['date']).dt.year
            if not cashflow.empty:
                cashflow['year'] = pd.to_datetime(cashflow['date']).dt.year
            
            # Get years with data
            years = sorted(financials['year'].unique())
            
            # Prepare results
            metrics = []
            
            for year in years:
                year_metrics = {'year': year}
                
                # Get revenue
                year_fin = financials[financials['year'] == year]
                rev_row = year_fin[year_fin['metric_type'].isin(['Revenue', 'OperatingRevenue'])]
                if not rev_row.empty:
                    revenue = rev_row['value'].iloc[0]
                    year_metrics['revenue'] = revenue
                    
                    # Calculate growth if we have prior year
                    if len(metrics) > 0 and 'revenue' in metrics[-1]:
                        prior_rev = metrics[-1]['revenue']
                        if prior_rev > 0:
                            growth = (revenue - prior_rev) / prior_rev
                            year_metrics['revenue_growth'] = growth
                
                # Get operating income
                op_row = year_fin[year_fin['metric_type'].isin(['OperatingIncome', 'OperatingProfit'])]
                if not op_row.empty:
                    op_income = op_row['value'].iloc[0]
                    year_metrics['operating_income'] = op_income
                    
                    # Calculate operating margin
                    if 'revenue' in year_metrics and year_metrics['revenue'] > 0:
                        year_metrics['operating_margin'] = op_income / year_metrics['revenue']
                
                # Get net income
                net_row = year_fin[year_fin['metric_type'].isin(['NetIncome', 'ProfitAfterTax'])]
                if not net_row.empty:
                    net_income = net_row['value'].iloc[0]
                    year_metrics['net_income'] = net_income
                    
                    # Calculate net margin
                    if 'revenue' in year_metrics and year_metrics['revenue'] > 0:
                        year_metrics['net_margin'] = net_income / year_metrics['revenue']
                
                # Get balance sheet metrics
                year_bal = balance[balance['year'] == year]
                
                # Total assets
                assets_row = year_bal[year_bal['metric_type'].isin(['TotalAssets', 'Assets'])]
                if not assets_row.empty:
                    total_assets = assets_row['value'].iloc[0]
                    year_metrics['total_assets'] = total_assets
                    
                    # Calculate ROA if we have net income
                    if 'net_income' in year_metrics and total_assets > 0:
                        year_metrics['roa'] = year_metrics['net_income'] / total_assets
                
                # Equity
                equity_row = year_bal[year_bal['metric_type'].isin(['TotalEquity', 'StockholdersEquity'])]
                if not equity_row.empty:
                    equity = equity_row['value'].iloc[0]
                    year_metrics['equity'] = equity
                    
                    # Calculate ROE if we have net income
                    if 'net_income' in year_metrics and equity > 0:
                        year_metrics['roe'] = year_metrics['net_income'] / equity
                    
                    # Calculate debt-to-equity if we have total assets
                    if 'total_assets' in year_metrics and equity > 0:
                        liabilities = year_metrics['total_assets'] - equity
                        year_metrics['debt_to_equity'] = liabilities / equity
                
                # Get cash flow metrics if available
                if not cashflow.empty:
                    year_cf = cashflow[cashflow['year'] == year]
                    
                    # Operating cash flow
                    ocf_row = year_cf[year_cf['metric_type'].isin(['CashFlowsFromOperatingActivities', 'NetCashProvidedByOperatingActivities'])]
                    if not ocf_row.empty:
                        ocf = ocf_row['value'].iloc[0]
                        year_metrics['operating_cash_flow'] = ocf
                        
                        # Cash flow to revenue
                        if 'revenue' in year_metrics and year_metrics['revenue'] > 0:
                            year_metrics['ocf_to_revenue'] = ocf / year_metrics['revenue']
                    
                    # Capital expenditure
                    capex_row = year_cf[year_cf['metric_type'].isin(['PropertyAndPlantAndEquipment', 'AcquisitionOfPropertyPlantAndEquipment'])]
                    if not capex_row.empty:
                        capex = abs(capex_row['value'].iloc[0])  # Use absolute value
                        year_metrics['capex'] = capex
                        
                        # CAPEX to revenue
                        if 'revenue' in year_metrics and year_metrics['revenue'] > 0:
                            year_metrics['capex_to_revenue'] = capex / year_metrics['revenue']
                        
                        # Free cash flow if we have OCF
                        if 'operating_cash_flow' in year_metrics:
                            year_metrics['free_cash_flow'] = year_metrics['operating_cash_flow'] - capex
                
                metrics.append(year_metrics)
            
            # Convert to DataFrame
            metrics_df = pd.DataFrame(metrics)
            return metrics_df
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {stock_id}: {e}")
            return pd.DataFrame()
    
    def analyze_industry(self, industry, top_n=10):
        """Analyze an industry to find top-performing stocks.
        
        Args:
            industry: Industry to analyze
            top_n: Number of top stocks to return
            
        Returns:
            DataFrame with industry analysis
        """
        try:
            # Get complete stocks in this industry
            stocks = self.get_complete_stocks(industry=industry)
            
            if stocks.empty:
                logger.warning(f"No stocks found for industry: {industry}")
                return pd.DataFrame()
            
            # Calculate metrics for each stock
            industry_metrics = []
            
            for _, row in stocks.iterrows():
                stock_id = row['stock_id']
                stock_name = row['stock_name']
                
                metrics = self.calculate_basic_metrics(stock_id)
                
                if not metrics.empty and len(metrics) >= 2:  # At least 2 years of data
                    # Calculate average metrics
                    avg_metrics = {
                        'stock_id': stock_id,
                        'stock_name': stock_name
                    }
                    
                    # Use last 3 years or all available years if less than 3
                    recent_metrics = metrics.tail(min(3, len(metrics)))
                    
                    # Calculate average metrics from recent years
                    for col in ['revenue_growth', 'operating_margin', 'net_margin', 'roa', 'roe', 'ocf_to_revenue']:
                        if col in recent_metrics.columns:
                            avg_metrics[f'avg_{col}'] = recent_metrics[col].mean()
                    
                    # Also get latest values
                    latest = metrics.iloc[-1]
                    for col in ['revenue', 'operating_income', 'net_income', 'total_assets', 'equity']:
                        if col in latest:
                            avg_metrics[f'latest_{col}'] = latest[col]
                    
                    industry_metrics.append(avg_metrics)
            
            # Convert to DataFrame
            industry_df = pd.DataFrame(industry_metrics)
            
            if industry_df.empty:
                logger.warning(f"No analyzable stocks found in {industry}")
                return pd.DataFrame()
            
            # Rank stocks by various metrics
            if 'avg_revenue_growth' in industry_df.columns:
                industry_df['growth_rank'] = industry_df['avg_revenue_growth'].rank(ascending=False)
            
            if 'avg_operating_margin' in industry_df.columns:
                industry_df['margin_rank'] = industry_df['avg_operating_margin'].rank(ascending=False)
            
            if 'avg_roe' in industry_df.columns:
                industry_df['roe_rank'] = industry_df['avg_roe'].rank(ascending=False)
            
            # Calculate overall score (average of ranks)
            rank_cols = [col for col in industry_df.columns if col.endswith('_rank')]
            if rank_cols:
                industry_df['overall_score'] = industry_df[rank_cols].mean(axis=1)
                industry_df = industry_df.sort_values('overall_score')
            
            # Check if we have any valid metrics
            if industry_df.empty:
                logger.warning(f"No valid metrics found for {industry} industry stocks")
                return pd.DataFrame()
                
            # Make sure we have at least some ranking metrics
            rank_cols = [col for col in industry_df.columns if col.endswith('_rank')]
            if not rank_cols:
                # If no rankings, return raw data
                logger.warning(f"No ranking metrics available for {industry} stocks")
                return industry_df.head(top_n)
                
            # Return top N stocks
            return industry_df.head(top_n)
            
        except Exception as e:
            logger.error(f"Error analyzing industry {industry}: {e}")
            return pd.DataFrame()
    
    def find_value_stocks(self, min_roe=0.1, max_debt_equity=1.0, top_n=20):
        """Find value stocks based on financial criteria.
        
        Args:
            min_roe: Minimum ROE to consider
            max_debt_equity: Maximum debt-to-equity ratio
            top_n: Number of top stocks to return
            
        Returns:
            DataFrame with value stock candidates
        """
        try:
            # Get all stocks with complete data
            all_stocks = self.get_complete_stocks()
            
            if all_stocks.empty:
                logger.warning("No stocks found with complete data")
                return pd.DataFrame()
            
            # Calculate metrics and filter based on criteria
            value_candidates = []
            
            for _, row in all_stocks.iterrows():
                stock_id = row['stock_id']
                stock_name = row['stock_name']
                industry = row['industry']
                
                metrics = self.calculate_basic_metrics(stock_id)
                
                if not metrics.empty and len(metrics) >= 2:  # At least 2 years of data
                    # Get latest metrics
                    latest = metrics.iloc[-1].to_dict()
                    
                    # Calculate 3-year averages
                    recent = metrics.tail(min(3, len(metrics)))
                    avg_metrics = {
                        'avg_revenue_growth': recent['revenue_growth'].mean() if 'revenue_growth' in recent else None,
                        'avg_operating_margin': recent['operating_margin'].mean() if 'operating_margin' in recent else None,
                        'avg_net_margin': recent['net_margin'].mean() if 'net_margin' in recent else None,
                        'avg_roe': recent['roe'].mean() if 'roe' in recent else None,
                        'avg_roa': recent['roa'].mean() if 'roa' in recent else None,
                        'avg_debt_to_equity': recent['debt_to_equity'].mean() if 'debt_to_equity' in recent else None,
                    }
                    
                    # Apply value criteria
                    meets_criteria = (
                        avg_metrics['avg_roe'] is not None and avg_metrics['avg_roe'] >= min_roe and
                        avg_metrics['avg_debt_to_equity'] is not None and avg_metrics['avg_debt_to_equity'] <= max_debt_equity
                    )
                    
                    if meets_criteria:
                        candidate = {
                            'stock_id': stock_id,
                            'stock_name': stock_name,
                            'industry': industry
                        }
                        
                        # Add metrics
                        candidate.update(avg_metrics)
                        
                        # Add latest values 
                        for key in ['revenue', 'net_income', 'equity']:
                            if key in latest:
                                candidate[f'latest_{key}'] = latest[key]
                        
                        value_candidates.append(candidate)
            
            # Convert to DataFrame
            candidates_df = pd.DataFrame(value_candidates)
            
            if candidates_df.empty:
                logger.warning("No stocks meet the value criteria")
                return pd.DataFrame()
            
            # Rank by ROE
            candidates_df = candidates_df.sort_values('avg_roe', ascending=False)
            
            # Return top N candidates
            return candidates_df.head(top_n)
            
        except Exception as e:
            logger.error(f"Error finding value stocks: {e}")
            return pd.DataFrame()
    
    def get_price_performance(self, stock_id, days=365):
        """Get price performance for a stock over a period.
        
        Args:
            stock_id: The stock ID to analyze
            days: Number of days to analyze
            
        Returns:
            DataFrame with price performance metrics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get price data
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            price_data = pd.read_sql_query(
                "SELECT date, open, close, high, low, volume FROM stock_prices WHERE stock_id = ? AND date >= ? ORDER BY date",
                conn,
                params=(stock_id, cutoff_date)
            )
            
            conn.close()
            
            if price_data.empty:
                logger.warning(f"No price data found for {stock_id}")
                return None
            
            # Convert date and sort
            price_data['date'] = pd.to_datetime(price_data['date'])
            price_data = price_data.sort_values('date')
            
            # Calculate returns
            price_data['daily_return'] = price_data['close'].pct_change()
            
            # Calculate cumulative return
            start_price = price_data.iloc[0]['close']
            end_price = price_data.iloc[-1]['close']
            total_return = (end_price / start_price) - 1
            
            # Calculate volatility (standard deviation of returns)
            volatility = price_data['daily_return'].std() * (252 ** 0.5)  # Annualized
            
            # Calculate max drawdown
            price_data['cumulative_return'] = (1 + price_data['daily_return']).cumprod() - 1
            price_data['rolling_max'] = price_data['cumulative_return'].cummax()
            price_data['drawdown'] = price_data['rolling_max'] - price_data['cumulative_return']
            max_drawdown = price_data['drawdown'].max()
            
            # Results
            performance = {
                'start_date': price_data.iloc[0]['date'],
                'end_date': price_data.iloc[-1]['date'],
                'days': (price_data.iloc[-1]['date'] - price_data.iloc[0]['date']).days,
                'start_price': start_price,
                'end_price': end_price,
                'total_return': total_return,
                'annualized_return': (1 + total_return) ** (365 / max(1, (price_data.iloc[-1]['date'] - price_data.iloc[0]['date']).days)) - 1,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': (total_return / max(volatility, 0.0001)) if volatility > 0 else 0,
                'avg_volume': price_data['volume'].mean(),
                'price_data': price_data
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting price performance for {stock_id}: {e}")
            return None
    
    def find_momentum_stocks(self, min_return=0.1, max_stocks=20):
        """Find stocks with strong price momentum.
        
        Args:
            min_return: Minimum return over the period to consider
            max_stocks: Maximum number of stocks to return
            
        Returns:
            DataFrame with momentum stock candidates
        """
        try:
            # Get all stocks with complete data
            all_stocks = self.get_complete_stocks()
            
            if all_stocks.empty:
                logger.warning("No stocks found with complete data")
                return pd.DataFrame()
            
            # Calculate price performance for each stock
            momentum_candidates = []
            
            for _, row in all_stocks.iterrows():
                stock_id = row['stock_id']
                stock_name = row['stock_name']
                industry = row['industry']
                
                # Get 6-month performance
                performance = self.get_price_performance(stock_id, days=180)
                
                if performance and performance['total_return'] >= min_return:
                    candidate = {
                        'stock_id': stock_id,
                        'stock_name': stock_name,
                        'industry': industry,
                        'return_6m': performance['total_return'],
                        'annualized_return': performance['annualized_return'],
                        'volatility': performance['volatility'],
                        'sharpe_ratio': performance['sharpe_ratio'],
                        'max_drawdown': performance['max_drawdown'],
                        'avg_volume': performance['avg_volume']
                    }
                    
                    momentum_candidates.append(candidate)
            
            # Convert to DataFrame
            candidates_df = pd.DataFrame(momentum_candidates)
            
            if candidates_df.empty:
                logger.warning("No stocks meet the momentum criteria")
                return pd.DataFrame()
            
            # Rank by return
            candidates_df = candidates_df.sort_values('return_6m', ascending=False)
            
            # Return top stocks
            return candidates_df.head(max_stocks)
            
        except Exception as e:
            logger.error(f"Error finding momentum stocks: {e}")
            return pd.DataFrame()
    
    def generate_stock_report(self, stock_id):
        """Generate a comprehensive report for a stock.
        
        Args:
            stock_id: The stock ID to analyze
            
        Returns:
            Dictionary with stock report data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get stock info
            stock_info = pd.read_sql_query(
                "SELECT stock_id, stock_name, industry FROM stock_info WHERE stock_id = ?",
                conn,
                params=(stock_id,)
            )
            
            if stock_info.empty:
                logger.warning(f"Stock {stock_id} not found")
                conn.close()
                return None
            
            stock_name = stock_info.iloc[0]['stock_name']
            industry = stock_info.iloc[0]['industry']
            
            # Calculate financial metrics
            metrics = self.calculate_basic_metrics(stock_id)
            
            # Get price performance
            perf_1y = self.get_price_performance(stock_id, days=365)
            perf_6m = self.get_price_performance(stock_id, days=180)
            
            conn.close()
            
            # Create report
            report = {
                'stock_id': stock_id,
                'stock_name': stock_name,
                'industry': industry,
                'financial_metrics': metrics if not metrics.empty else None,
                'price_performance_1y': perf_1y,
                'price_performance_6m': perf_6m
            }
            
            # Calculate growth trends
            if not metrics.empty and len(metrics) >= 3:
                # Revenue growth trend
                if 'revenue' in metrics.columns:
                    revenue_trend = metrics['revenue'].pct_change().dropna()
                    report['revenue_growth_trend'] = revenue_trend.tolist() if not revenue_trend.empty else None
                
                # Profit margin trend
                if 'net_margin' in metrics.columns:
                    margin_trend = metrics['net_margin'].dropna()
                    report['margin_trend'] = margin_trend.tolist() if not margin_trend.empty else None
                
                # ROE trend
                if 'roe' in metrics.columns:
                    roe_trend = metrics['roe'].dropna()
                    report['roe_trend'] = roe_trend.tolist() if not roe_trend.empty else None
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report for {stock_id}: {e}")
            return None
    
    def plot_financial_trends(self, stock_id):
        """Plot key financial trends for a stock.
        
        Args:
            stock_id: The stock ID to analyze
            
        Returns:
            Matplotlib figure object
        """
        try:
            # Get metrics
            metrics = self.calculate_basic_metrics(stock_id)
            
            if metrics.empty:
                logger.warning(f"No financial metrics found for {stock_id}")
                return None
            
            # Get stock name
            conn = sqlite3.connect(self.db_path)
            stock_info = pd.read_sql_query(
                "SELECT stock_name FROM stock_info WHERE stock_id = ?",
                conn,
                params=(stock_id,)
            )
            conn.close()
            
            stock_name = stock_info.iloc[0]['stock_name'] if not stock_info.empty else stock_id
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"Financial Trends: {stock_id} - {stock_name}", fontsize=16)
            
            # Plot revenue and growth
            ax1 = axes[0, 0]
            if 'revenue' in metrics.columns:
                metrics['revenue'].plot(ax=ax1, marker='o', color='blue')
                ax1.set_title('Revenue')
                ax1.set_ylabel('Amount')
                ax1.grid(True, alpha=0.3)
                
                # Add secondary axis for growth
                if 'revenue_growth' in metrics.columns:
                    ax1_2 = ax1.twinx()
                    metrics['revenue_growth'].plot(ax=ax1_2, marker='s', color='red', linestyle='--')
                    ax1_2.set_ylabel('Growth Rate')
                    ax1_2.set_ylim([-0.5, 1.0])  # Reasonable range for growth rates
                    ax1_2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Plot margins
            ax2 = axes[0, 1]
            margin_cols = [col for col in ['operating_margin', 'net_margin'] if col in metrics.columns]
            if margin_cols:
                metrics[margin_cols].plot(ax=ax2, marker='o')
                ax2.set_title('Profitability Margins')
                ax2.set_ylabel('Margin')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([-0.2, 0.5])  # Reasonable range for margins
                ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Plot returns
            ax3 = axes[1, 0]
            return_cols = [col for col in ['roe', 'roa'] if col in metrics.columns]
            if return_cols:
                metrics[return_cols].plot(ax=ax3, marker='o')
                ax3.set_title('Return Metrics')
                ax3.set_ylabel('Return')
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim([-0.2, 0.5])  # Reasonable range for returns
                ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Plot cash flow metrics
            ax4 = axes[1, 1]
            cf_cols = [col for col in ['ocf_to_revenue', 'capex_to_revenue'] if col in metrics.columns]
            if cf_cols:
                metrics[cf_cols].plot(ax=ax4, marker='o')
                ax4.set_title('Cash Flow Metrics')
                ax4.set_ylabel('Ratio to Revenue')
                ax4.grid(True, alpha=0.3)
                ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting trends for {stock_id}: {e}")
            return None
    
    def plot_price_performance(self, stock_id, days=365):
        """Plot price performance for a stock.
        
        Args:
            stock_id: The stock ID to analyze
            days: Number of days to analyze
            
        Returns:
            Matplotlib figure object
        """
        try:
            # Get price performance
            performance = self.get_price_performance(stock_id, days=days)
            
            if not performance or 'price_data' not in performance:
                logger.warning(f"No price data found for {stock_id}")
                return None
            
            # Get stock name
            conn = sqlite3.connect(self.db_path)
            stock_info = pd.read_sql_query(
                "SELECT stock_name FROM stock_info WHERE stock_id = ?",
                conn,
                params=(stock_id,)
            )
            conn.close()
            
            stock_name = stock_info.iloc[0]['stock_name'] if not stock_info.empty else stock_id
            
            # Extract price data
            price_data = performance['price_data']
            
            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot price
            ax1 = axes[0]
            ax1.plot(price_data['date'], price_data['close'], color='blue')
            ax1.set_title(f"Price Performance: {stock_id} - {stock_name} (Return: {performance['total_return']:.1%})")
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            
            # Plot volume
            ax2 = axes[1]
            ax2.bar(price_data['date'], price_data['volume'], color='gray', alpha=0.7)
            ax2.set_ylabel('Volume')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting price performance for {stock_id}: {e}")
            return None

# Example usage
if __name__ == "__main__":
    analyzer = TaiwanStockAnalyzer()
    
    print("\nAnalyzing Taiwan stocks...\n")
    
    # Find complete stocks
    complete_stocks = analyzer.get_complete_stocks(min_years=2)
    print(f"Found {len(complete_stocks)} stocks with complete data")
    
    # Analyze semiconductor industry
    print("\nAnalyzing Semiconductor industry...")
    semicon_analysis = analyzer.analyze_industry("Semiconductors", top_n=5)
    if not semicon_analysis.empty:
        print("\nTop 5 Semiconductor stocks by financial metrics:")
        # Select only columns that exist in the DataFrame
        display_cols = ['stock_id', 'stock_name']
        for col in ['avg_revenue_growth', 'avg_operating_margin', 'avg_roe']:
            if col in semicon_analysis.columns:
                display_cols.append(col)
        print(semicon_analysis[display_cols])
    
    # Find value stocks
    print("\nFinding value stocks...")
    value_stocks = analyzer.find_value_stocks(min_roe=0.10, max_debt_equity=0.8, top_n=5)
    if not value_stocks.empty:
        print("\nTop value stock candidates:")
        # Select only columns that exist in the DataFrame
        display_cols = ['stock_id', 'stock_name', 'industry']
        for col in ['avg_roe', 'avg_debt_to_equity']:
            if col in value_stocks.columns:
                display_cols.append(col)
        print(value_stocks[display_cols])
    
    # Find momentum stocks
    print("\nFinding momentum stocks...")
    momentum_stocks = analyzer.find_momentum_stocks(min_return=0.10, max_stocks=5)
    if not momentum_stocks.empty:
        print("\nTop momentum stock candidates:")
        # Select only columns that exist in the DataFrame
        display_cols = ['stock_id', 'stock_name', 'industry']
        for col in ['return_6m', 'sharpe_ratio']:
            if col in momentum_stocks.columns:
                display_cols.append(col)
        print(momentum_stocks[display_cols])
    
    # Generate report for a sample stock (use a stock from the semiconductor list if available)
    sample_stock = None
    if not semicon_analysis.empty:
        sample_stock = semicon_analysis.iloc[0]['stock_id']
    elif not value_stocks.empty:
        sample_stock = value_stocks.iloc[0]['stock_id']
    
    if sample_stock:
        print(f"\nGenerating report for {sample_stock}...")
        report = analyzer.generate_stock_report(sample_stock)
        if report:
            print(f"\nStock: {report['stock_id']} - {report['stock_name']} ({report['industry']})")
            
            if report['price_performance_1y']:
                print(f"1-Year Return: {report['price_performance_1y']['total_return']:.2%}")
                print(f"Annualized Return: {report['price_performance_1y']['annualized_return']:.2%}")
                print(f"Volatility: {report['price_performance_1y']['volatility']:.2%}")
                print(f"Sharpe Ratio: {report['price_performance_1y']['sharpe_ratio']:.2f}")
            
            # Plot financial trends
            try:
                fig = analyzer.plot_financial_trends(sample_stock)
                if fig:
                    fig.savefig(f"{sample_stock}_financials.png")
                    print(f"Financial trends plot saved as {sample_stock}_financials.png")
                
                # Plot price performance
                fig = analyzer.plot_price_performance(sample_stock)
                if fig:
                    fig.savefig(f"{sample_stock}_price.png")
                    print(f"Price performance plot saved as {sample_stock}_price.png")
            except Exception as e:
                print(f"Error generating plots: {e}")

