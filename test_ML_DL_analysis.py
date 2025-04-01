import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DCFResultAnalyzer:
    """Analyze and visualize results from multi-model DCF valuations."""
    
    def __init__(self, results_file_path: str):
        """Initialize with results file path."""
        self.results_path = results_file_path
        self.results_df = None
        self.load_data()
        
    def load_data(self):
        """Load DCF results data."""
        try:
            if self.results_path.endswith('.csv'):
                self.results_df = pd.read_csv(self.results_path)
            else:
                # For direct dataframe input from clipboard
                self.results_df = pd.read_clipboard()
            
            # Convert price columns to numeric
            numeric_columns = ['Market Price', 'Standard DCF', 'ML-Enhanced DCF', 'ML+DL Ensemble DCF']
            for col in numeric_columns:
                if col in self.results_df.columns:
                    self.results_df[col] = pd.to_numeric(self.results_df[col], errors='coerce')
                    
            # Convert P/V ratio columns, handling the 'x' suffix properly
            ratio_columns = ['Standard P/V', 'ML P/V', 'ML+DL P/V']
            for col in ratio_columns:
                if col in self.results_df.columns:
                    # Make sure we handle both string and numeric formats
                    self.results_df[col] = self.results_df[col].apply(
                        lambda x: float(str(x).replace('x', '')) if pd.notna(x) and 'nan' not in str(x).lower() else np.nan
                    )
                    
            # Parse ML Growth Rates and DL Growth Rates from string to list
            for rates_col in ['ML Growth Rates', 'DL Growth Rates']:
                if rates_col in self.results_df.columns:
                    self.results_df[rates_col] = self.results_df[rates_col].apply(
                        lambda x: eval(x) if isinstance(x, str) and x.strip() and 'nan' not in x.lower() else None
                    )
                    
            logger.info(f"Loaded data with {len(self.results_df)} stocks")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.results_df = pd.DataFrame()
            
    def analyze_model_performance(self):
        """Analyze how different DCF models perform relative to market prices."""
        if self.results_df.empty:
            logger.error("No data available for analysis")
            return
            
        try:
            # Calculate median P/V ratios by industry
            industry_pv = self.results_df.groupby('Industry')[['Standard P/V', 'ML P/V', 'ML+DL P/V']].median()
            
            # Calculate average deviation from market price by model
            self.results_df['Std_Dev'] = (self.results_df['Standard DCF'] - self.results_df['Market Price']).abs() / self.results_df['Market Price']
            self.results_df['ML_Dev'] = (self.results_df['ML-Enhanced DCF'] - self.results_df['Market Price']).abs() / self.results_df['Market Price']
            self.results_df['ML+DL_Dev'] = (self.results_df['ML+DL Ensemble DCF'] - self.results_df['Market Price']).abs() / self.results_df['Market Price']
            
            model_deviation = self.results_df[['Std_Dev', 'ML_Dev', 'ML+DL_Dev']].mean()
            
            # Calculate correlation of growth rates with valuation accuracy
            correlations = {}
            for year in range(5):
                yr_correlations = []
                for stock in self.results_df.itertuples():
                    if hasattr(stock, 'ML_Growth_Rates') and stock.ML_Growth_Rates:
                        growth_rate = stock.ML_Growth_Rates[year]
                        if hasattr(stock, 'ML_Dev') and not pd.isna(stock.ML_Dev):
                            yr_correlations.append((growth_rate, stock.ML_Dev))
                
                if yr_correlations:
                    growth_rates, deviations = zip(*yr_correlations)
                    correlations[f'Year {year+1}'] = np.corrcoef(growth_rates, deviations)[0, 1]
            
            return {
                'industry_pv': industry_pv,
                'model_deviation': model_deviation,
                'growth_correlation': correlations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing model performance: {e}")
            return None
    
    def identify_valuation_anomalies(self):
        """Identify stocks with unusual valuation patterns."""
        if self.results_df.empty:
            logger.error("No data available for analysis")
            return
            
        try:
            # Define anomaly patterns
            anomalies = {
                'highly_undervalued': self.results_df[
                    (self.results_df['Standard P/V'] < 0.5) & 
                    (self.results_df['ML P/V'] < 0.5)
                ],
                'highly_overvalued': self.results_df[
                    (self.results_df['Standard P/V'] > 2.0) & 
                    (self.results_df['ML P/V'] > 2.0)
                ],
                'model_disagreement': self.results_df[
                    ((self.results_df['Standard P/V'] < 0.9) & (self.results_df['ML P/V'] > 1.5)) |
                    ((self.results_df['Standard P/V'] > 1.5) & (self.results_df['ML P/V'] < 0.9))
                ],
                'dl_impact': self.results_df[
                    abs(self.results_df['ML P/V'] - self.results_df['ML+DL P/V']) > 0.5
                ]
            }
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error identifying anomalies: {e}")
            return None
    
    def analyze_growth_predictions(self):
        """Analyze growth rate predictions across models."""
        if self.results_df.empty:
            logger.error("No data available for analysis")
            return
            
        try:
            # Calculate average ML growth rates by industry
            industry_growth = {}
            for industry in self.results_df['Industry'].unique():
                industry_df = self.results_df[self.results_df['Industry'] == industry]
                growth_by_year = [[] for _ in range(5)]
                
                for stock in industry_df.itertuples():
                    if hasattr(stock, 'ML_Growth_Rates') and stock.ML_Growth_Rates:
                        for i, rate in enumerate(stock.ML_Growth_Rates):
                            if i < len(growth_by_year):
                                growth_by_year[i].append(rate)
                
                industry_growth[industry] = [
                    np.mean(year_rates) if year_rates else None
                    for year_rates in growth_by_year
                ]
            
            # Analyze DL model patterns with detailed statistics
            dl_patterns = {
                'constant_values': [],
                'repeated_values': [],
                'stocks_with_constant_dl': [],
                'value_frequency': {},
                'most_common_values': []
            }
            
            all_dl_values = []
            
            for stock in self.results_df.itertuples():
                if hasattr(stock, 'DL_Growth_Rates') and stock.DL_Growth_Rates:
                    # Collect all values for frequency analysis
                    all_dl_values.extend(stock.DL_Growth_Rates)
                    
                    # Check for repeated values within this stock's predictions
                    dl_unique_values = set(stock.DL_Growth_Rates)
                    
                    # Count value occurrences for this stock
                    value_counts = {}
                    for value in stock.DL_Growth_Rates:
                        value_counts[value] = value_counts.get(value, 0) + 1
                        # Track global value frequency
                        dl_patterns['value_frequency'][value] = dl_patterns['value_frequency'].get(value, 0) + 1
                    
                    # Find values that appear multiple times
                    repeated = [(val, count) for val, count in value_counts.items() if count > 1]
                    if repeated:
                        for val, count in repeated:
                            if val not in dl_patterns['repeated_values']:
                                dl_patterns['repeated_values'].append(val)
                    
                    # Check if this stock has mostly constant values (≤2 unique values)
                    if len(dl_unique_values) <= 2:
                        dl_patterns['stocks_with_constant_dl'].append((stock.Ticker, stock.Name, stock.DL_Growth_Rates))
            
            # Find most common values across all stocks
            if all_dl_values:
                # Get frequency of each value
                value_counts = {}
                for value in all_dl_values:
                    value_counts[value] = value_counts.get(value, 0) + 1
                
                # Sort by frequency
                sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
                dl_patterns['most_common_values'] = sorted_values[:5]  # Top 5 most common values
                
                # Store all unique values
                dl_patterns['constant_values'] = list(sorted(set(all_dl_values)))
            
            # Calculate ML vs DL comparison metrics
            ml_dl_comparison = self._compare_ml_dl_growth()
            
            return {
                'industry_growth': industry_growth,
                'dl_patterns': dl_patterns,
                'ml_dl_comparison': ml_dl_comparison
            }
            
        except Exception as e:
            logger.error(f"Error analyzing growth predictions: {e}")
            return None

    def _compare_ml_dl_growth(self):
        """Compare ML and DL growth predictions."""
        comparison = {
            'avg_difference': [],
            'correlation': [],
            'stocks_with_large_diff': []
        }
        
        for stock in self.results_df.itertuples():
            if (hasattr(stock, 'ML_Growth_Rates') and stock.ML_Growth_Rates and 
                hasattr(stock, 'DL_Growth_Rates') and stock.DL_Growth_Rates):
                
                # Calculate average difference between models
                diffs = [abs(ml - dl) for ml, dl in zip(stock.ML_Growth_Rates, stock.DL_Growth_Rates)]
                avg_diff = np.mean(diffs)
                comparison['avg_difference'].append(avg_diff)
                
                # Check for large differences
                if avg_diff > 0.15:  # If average difference > 15%
                    comparison['stocks_with_large_diff'].append((stock.Ticker, avg_diff))
        
        # Calculate overall average difference
        if comparison['avg_difference']:
            comparison['overall_avg_diff'] = np.mean(comparison['avg_difference'])
        
        return comparison
    
    def visualize_industry_valuations(self):
        """Create visualization of valuations by industry."""
        if self.results_df.empty:
            logger.error("No data available for visualization")
            return
            
        try:
            # Group by industry and calculate median P/V
            industry_pv = self.results_df.groupby('Industry')[['Standard P/V', 'ML P/V', 'ML+DL P/V']].median().reset_index()
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Set up grouped bar chart
            industries = industry_pv['Industry']
            x = np.arange(len(industries))
            width = 0.25
            
            # Create bars for each model
            plt.bar(x - width, industry_pv['Standard P/V'], width, label='Standard DCF')
            plt.bar(x, industry_pv['ML P/V'], width, label='ML-Enhanced DCF')
            plt.bar(x + width, industry_pv['ML+DL P/V'], width, label='ML+DL Ensemble DCF')
            
            # Add fair value line
            plt.axhline(y=1.0, color='red', linestyle='--', label='Fair Value (P/V = 1.0)')
            
            # Customize chart
            plt.xlabel('Industry')
            plt.ylabel('Price-to-Value Ratio (P/V)')
            plt.title('Median P/V Ratio by Industry and Model')
            plt.xticks(x, industries, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            # Save the grouped bar chart showing P/V ratios by industry and model
            plt.savefig('industry_valuations.png')
            plt.close()
            
            # Create growth prediction chart
            self.visualize_growth_predictions()
            
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing industry valuations: {e}")
            return False
            
    def visualize_growth_predictions(self):
        """Visualize growth predictions by industry."""
        try:
            # Get industry growth analysis
            growth_analysis = self.analyze_growth_predictions()
            if not growth_analysis or not growth_analysis['industry_growth']:
                return False
            
            industry_growth = growth_analysis['industry_growth']
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            years = range(1, 6)
            has_data = False
            for industry, growth_rates in industry_growth.items():
                if growth_rates and all(rate is not None for rate in growth_rates):
                    plt.plot(years, [rate * 100 for rate in growth_rates], marker='o', label=industry)
                    has_data = True
            
            # Only add legend if we have plot data
            if has_data:
                plt.legend()
                
            plt.xlabel('Forecast Year')
            plt.ylabel('Growth Rate (%)')
            plt.title('ML-Predicted Growth Rates by Industry')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the line chart showing growth rates by industry
            plt.savefig('industry_growth_predictions.png')
            plt.close()
            
            # Create Price/Value heatmap
            self.create_pv_heatmap()
            
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing growth predictions: {e}")
            return False
    
    def create_pv_heatmap(self):
        """Create a heatmap showing P/V ratios by stock and model."""
        try:
            # Prepare data for heatmap
            heatmap_data = self.results_df[['Ticker', 'Industry', 'Standard P/V', 'ML P/V', 'ML+DL P/V']].copy()
            heatmap_data = heatmap_data.dropna(subset=['Standard P/V', 'ML P/V', 'ML+DL P/V'])
            
            # Sort by industry
            heatmap_data = heatmap_data.sort_values('Industry')
            
            # Pivot for heatmap format
            pivot_data = heatmap_data.set_index(['Ticker', 'Industry'])
            
            # Create figure
            plt.figure(figsize=(10, len(heatmap_data) * 0.5 + 2))
            
            # Create heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                      center=1.0, vmin=0.5, vmax=2.0, cbar_kws={'label': 'P/V Ratio'})
            
            plt.title('Price-to-Value Ratios by Model (Green: Undervalued, Red: Overvalued)')
            plt.tight_layout()
            
            # Save the heatmap visualization
            plt.savefig('pv_ratio_heatmap.png')
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating P/V heatmap: {e}")
            return False
    
    def run_comprehensive_analysis(self):
        """Run all analyses and generate a summary report."""
        try:
            model_performance = self.analyze_model_performance()
            anomalies = self.identify_valuation_anomalies()
            growth_analysis = self.analyze_growth_predictions()
            
            # Create visualizations
            self.visualize_industry_valuations()
            
            # Generate summary report
            report = []
            report.append("# DCF Model Analysis Summary Report")
            report.append("\n## 1. Model Performance Analysis")
            
            if model_performance:
                report.append("\n### Average Model Deviation from Market Price:")
                for model, dev in model_performance['model_deviation'].items():
                    model_name = {
                        'Std_Dev': 'Standard DCF', 
                        'ML_Dev': 'ML-Enhanced DCF', 
                        'ML+DL_Dev': 'ML+DL Ensemble DCF'
                    }.get(model, model)
                    report.append(f"- {model_name}: {dev:.2f} ({dev*100:.1f}%)")
                
                report.append("\n### Industry Median P/V Ratios:")
                for industry, row in model_performance['industry_pv'].iterrows():
                    report.append(f"- {industry}: Standard = {row['Standard P/V']:.2f}x, "
                                f"ML = {row['ML P/V']:.2f}x, ML+DL = {row['ML+DL P/V']:.2f}x")
            
            report.append("\n## 2. Valuation Anomalies")
            
            if anomalies:
                report.append("\n### Highly Undervalued Stocks (P/V < 0.5):")
                for stock in anomalies['highly_undervalued'].itertuples():
                    # Fix column name access and formatting issues
                    std_pv = getattr(stock, 'Standard P/V', np.nan)
                    ml_pv = getattr(stock, 'ML P/V', np.nan)
                    
                    # Format properly, handling NaN values
                    std_pv_str = f"{std_pv:.2f}x" if pd.notna(std_pv) else "N/A"
                    ml_pv_str = f"{ml_pv:.2f}x" if pd.notna(ml_pv) else "N/A"
                    
                    report.append(f"- {stock.Name} ({stock.Ticker}): Standard P/V = {std_pv_str}, ML P/V = {ml_pv_str}")
                
                report.append("\n### Highly Overvalued Stocks (P/V > 2.0):")
                for stock in anomalies['highly_overvalued'].itertuples():
                    std_pv = getattr(stock, 'Standard P/V', np.nan)
                    ml_pv = getattr(stock, 'ML P/V', np.nan)
                    
                    # Fix the formatting issue by checking if value is numeric
                    std_pv_str = f"{std_pv:.2f}x" if pd.notna(std_pv) else "N/A"
                    ml_pv_str = f"{ml_pv:.2f}x" if pd.notna(ml_pv) else "N/A"
                    
                    report.append(f"- {stock.Name} ({stock.Ticker}): Standard P/V = {std_pv_str}, ML P/V = {ml_pv_str}")
                
                report.append("\n### Model Disagreement Cases:")
                for stock in anomalies['model_disagreement'].itertuples():
                    std_pv = getattr(stock, 'Standard P/V', np.nan)
                    ml_pv = getattr(stock, 'ML P/V', np.nan)
                    
                    # Fix the formatting issue by checking if value is numeric
                    std_pv_str = f"{std_pv:.2f}x" if pd.notna(std_pv) else "N/A"
                    ml_pv_str = f"{ml_pv:.2f}x" if pd.notna(ml_pv) else "N/A"
                    
                    report.append(f"- {stock.Name} ({stock.Ticker}): Standard P/V = {std_pv_str}, ML P/V = {ml_pv_str}")
            
            report.append("\n## 3. Growth Prediction Analysis")
            
            if growth_analysis:
                report.append("\n### Industry Growth Projections (ML Model):")
                industry_growth = growth_analysis['industry_growth']
                
                # Sort industries by typical growth rate (using first year as reference)
                sorted_industries = sorted(
                    [(ind, rates) for ind, rates in industry_growth.items() if rates and rates[0] is not None],
                    key=lambda x: x[1][0],
                    reverse=True
                )
                
                for industry, growth_rates in sorted_industries:
                    if all(rate is not None for rate in growth_rates):
                        growth_str = ", ".join([f"Year {i+1}: {rate:.1%}" for i, rate in enumerate(growth_rates)])
                        report.append(f"- {industry}: {growth_str}")
                
                report.append("\n### Deep Learning Model Assessment:")
                
                if 'dl_patterns' in growth_analysis:
                    dl_patterns = growth_analysis['dl_patterns']
                    
                    # Report on most common values
                    if 'most_common_values' in dl_patterns and dl_patterns['most_common_values']:
                        report.append("- **Common Values Analysis**: The deep learning model most frequently outputs these values:")
                        for value, count in dl_patterns['most_common_values']:
                            report.append(f"  - {value:.1%}: appears {count} times")
                    
                    # Report on repeated values within predictions
                    if 'repeated_values' in dl_patterns and dl_patterns['repeated_values']:
                        repeated_values_str = ", ".join([f"{v:.1%}" for v in sorted(dl_patterns['repeated_values'])])
                        report.append(f"- **PATTERN DETECTED**: The deep learning model frequently repeats specific values within predictions: {repeated_values_str}")
                    
                    # Report on stocks with constant DL predictions
                    if 'stocks_with_constant_dl' in dl_patterns and dl_patterns['stocks_with_constant_dl']:
                        report.append(f"- **CONSTANT PREDICTIONS**: {len(dl_patterns['stocks_with_constant_dl'])} stocks have nearly constant DL predictions:")
                        for ticker, name, values in dl_patterns['stocks_with_constant_dl'][:5]:  # Show up to 5 examples
                            formatted_values = [f"{v:.1%}" for v in values]
                            report.append(f"  - {name} ({ticker}): {formatted_values}")
                        if len(dl_patterns['stocks_with_constant_dl']) > 5:
                            report.append(f"  - ...and {len(dl_patterns['stocks_with_constant_dl']) - 5} more")
                
                # ML vs DL comparison
                if 'ml_dl_comparison' in growth_analysis and growth_analysis['ml_dl_comparison'].get('overall_avg_diff'):
                    ml_dl = growth_analysis['ml_dl_comparison']
                    report.append(f"\n### ML vs DL Comparison:")
                    report.append(f"- Average difference between ML and DL predictions: {ml_dl['overall_avg_diff']:.1%}")
                    
                    if 'stocks_with_large_diff' in ml_dl and ml_dl['stocks_with_large_diff']:
                        report.append(f"- Stocks with large differences between ML and DL predictions:")
                        for ticker, diff in ml_dl['stocks_with_large_diff'][:5]:  # Show up to 5 examples
                            report.append(f"  - {ticker}: {diff:.1%} average difference")
            
            report.append("\n## 4. Recommendations")
            report.append("\n1. **Deep Learning Model Issues**: The deep learning model shows problematic patterns including "
                        "repeated values (particularly 0.01 and 0.3) across different stocks. Consider improving the model "
                        "training process to capture more stock-specific patterns.")
            
            report.append("\n2. **Industry-Specific Patterns**: The valuation disparities between industries suggest a need "
                        "for industry-specific calibration, especially for banking, semiconductor, and hardware sectors.")
            
            report.append("\n3. **Hardware and Electronic Stocks**: Companies in these sectors show particularly large "
                        "discrepancies between standard DCF and ML-enhanced valuations, suggesting growth assumptions may "
                        "need industry-specific adjustments.")
            
            report.append("\n4. **Ensemble Model Effectiveness**: The ML+DL ensemble model is not consistently improving "
                        "upon the ML-only model, indicating the DL component needs further refinement before it adds value.")
                    
            # Save report
            with open('dcf_analysis_report.md', 'w') as f:
                f.write('\n'.join(report))
            
            return '\n'.join(report)
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return f"Error generating report: {str(e)}"

    def format_pv_ratios(self, df):
        """Format P/V ratios properly for display."""
        formatted_df = df.copy()
        
        # Format P/V ratio columns
        ratio_columns = ['Standard P/V', 'ML P/V', 'ML+DL P/V']
        for col in ratio_columns:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.2f}x" if pd.notna(x) else "N/A"
                )
        
        return formatted_df


if __name__ == "__main__":
    analyzer = DCFResultAnalyzer("ml_dl_dcf_analysis_results.csv")
    report = analyzer.run_comprehensive_analysis()
    print("\nAnalysis complete! Report saved to dcf_analysis_report.md")
    print("\nKey findings:")
    
    # Print key sections from the report
    if "Deep Learning Model Assessment" in report:
        dl_section = report.split("Deep Learning Model Assessment:")[1].split("\n## 4")[0]
        print(f"\n- {dl_section.strip()}")
    
    # Print first recommendation
    if "Recommendations" in report:
        rec_section = report.split("Recommendations")[1].split("\n2.")[0]
        print(f"\n- {rec_section.strip()}")
