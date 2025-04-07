import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from dcf_integrator import IntegratedValuationModel
from dcf_model import DCFModel

# Configure page
st.set_page_config(
    page_title="Taiwan Stock Financial Analysis System",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if they don't exist yet
if 'valuation_results' not in st.session_state:
    st.session_state.valuation_results = None
if 'sensitivity_results' not in st.session_state:
    st.session_state.sensitivity_results = None
if 'ticker' not in st.session_state:
    st.session_state.ticker = None
if 'financial_data' not in st.session_state:
    st.session_state.financial_data = None
if 'available_industries' not in st.session_state:
    # Get available industry models from the industry_data_from_db folder
    industry_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "industry_data_from_db")
    available_industries = []
    
    if os.path.exists(industry_folder):
        for filename in os.listdir(industry_folder):
            if filename.endswith('_training.csv'):
                # Extract industry name from filename (remove _training.csv)
                industry_name = filename.replace('_training.csv', '')
                # Convert underscores to spaces and capitalize words
                industry_name = ' '.join(word.capitalize() for word in industry_name.split('_'))
                available_industries.append(industry_name)
    
    # Add standard industries if no models found or as fallback
    default_industries = [
        "Semiconductors", "Electronics", "Banking", "Telecommunications", 
        "Financial Services", "Computer Hardware", "Food & Beverage", "Retail", 
        "Healthcare", "Utilities", "Materials", "Electronics Manufacturing"
    ]
    
    # Create a combined and sorted list with Auto-detect at the top
    all_industries = sorted(list(set(available_industries + default_industries)))
    all_industries.insert(0, "Auto-detect")  # Put Auto-detect at the beginning
    
    st.session_state.available_industries = all_industries

# Function to perform valuation
def run_valuation(ticker, industry, forecast_years, perpetual_growth, use_ml, use_dl, use_industry):
    try:
        with st.spinner('Running valuation models...'):
            model = IntegratedValuationModel(
                use_ml=use_ml,
                use_dl=use_dl,
                use_industry=use_industry
            )
            
            result = model.run_valuation(
                ticker=ticker,
                industry=industry if industry != "Auto-detect" else None
            )
            
            st.session_state.valuation_results = result
            st.session_state.ticker = ticker
            
            # Get financial data
            dcf = DCFModel(stock_code=ticker)
            st.session_state.financial_data = dcf.get_financial_data()
            
            # Get current market data (price)
            try:
                market_data = dcf.get_market_data()
                if market_data:
                    result['current_price'] = market_data.get('price')
                    result['market_cap'] = market_data.get('market_cap')
            except Exception as e:
                st.warning(f"Could not retrieve current market price: {str(e)}")
            
            # Get industry average growth if available
            if industry and industry != "Auto-detect":
                try:
                    # Get industry benchmarks from industry_valuation_model
                    if model.industry_model and model.industry_model.industry_benchmarks is not None:
                        industry_benchmarks = model.industry_model.industry_benchmarks
                        industry_row = industry_benchmarks[industry_benchmarks['industry'] == industry]
                        
                        if not industry_row.empty and 'historical_growth_mean_median' in industry_row.columns:
                            result['industry_avg_growth'] = float(industry_row['historical_growth_mean_median'].iloc[0])
                        elif not industry_row.empty and 'historical_growth_mean_mean' in industry_row.columns:
                            result['industry_avg_growth'] = float(industry_row['historical_growth_mean_mean'].iloc[0])
                except Exception as e:
                    st.warning(f"Could not retrieve industry average growth: {str(e)}")
            
            return result
    except Exception as e:
        st.error(f"Error in valuation: {str(e)}")
        return None

# Function to perform sensitivity analysis
def run_sensitivity_analysis(ticker, base_price):
    try:
        with st.spinner('Running sensitivity analysis...'):
            # Create base DCF model
            dcf = DCFModel(stock_code=ticker)
            
            # Define factors to analyze
            factors = ['growth_rates', 'wacc', 'perpetual_growth_rate']
            results = {}
            
            for factor in factors:
                # Store original value
                original_value = getattr(dcf, factor) if hasattr(dcf, factor) else None
                
                if factor == 'growth_rates':
                    # For growth rates, adjust the first year's growth rate
                    if dcf.manual_growth_rates:
                        original = dcf.manual_growth_rates[0]
                        # High case: +20%
                        dcf.manual_growth_rates[0] = original * 1.2
                        high_val = dcf.calculate_stock_price()
                        
                        # Low case: -20%
                        dcf.manual_growth_rates[0] = original * 0.8
                        low_val = dcf.calculate_stock_price()
                        
                        # Restore original
                        dcf.manual_growth_rates[0] = original
                    else:
                        high_val = low_val = None
                elif factor == 'wacc':
                    # High case: +20%
                    dcf.wacc = dcf.wacc * 1.2
                    high_val = dcf.calculate_stock_price()
                    
                    # Low case: -20%
                    dcf.wacc = dcf.wacc * 0.8 / 1.2  # Adjusting from the high case
                    low_val = dcf.calculate_stock_price()
                    
                    # Restore original
                    dcf.wacc = original_value
                elif factor == 'perpetual_growth_rate':
                    # High case: +20%
                    dcf.perpetual_growth_rate = min(dcf.wacc - 0.01, dcf.perpetual_growth_rate * 1.2)
                    high_val = dcf.calculate_stock_price()
                    
                    # Low case: -20%
                    dcf.perpetual_growth_rate = max(0.01, dcf.perpetual_growth_rate * 0.8 / 1.2)
                    low_val = dcf.calculate_stock_price()
                    
                    # Restore original
                    dcf.perpetual_growth_rate = original_value
                
                # Calculate impact
                if high_val and low_val and base_price:
                    high_impact = (high_val - base_price) / base_price
                    low_impact = (low_val - base_price) / base_price
                    results[factor] = {'high': high_impact, 'low': low_impact, 
                                      'high_val': high_val, 'low_val': low_val}
            
            st.session_state.sensitivity_results = results
            return results
    except Exception as e:
        st.error(f"Error in sensitivity analysis: {str(e)}")
        return None

# Function to format financial metrics for display
def format_financial_metrics(df):
    if df is None or not isinstance(df, pd.DataFrame):
        return None
        
    metrics_to_show = [
        'Total Revenue', 'Operating Income', 'Net Income', 
        'Operating Margin', 'Net Margin', 'ROE',
        'Total Assets', 'Total Debt', 'Total Equity'
    ]
    
    available_metrics = [m for m in metrics_to_show if m in df.index]
    if not available_metrics:
        return None
    
    subset = df.loc[available_metrics]
    
    # Format the values
    formatted = subset.copy()
    for idx in formatted.index:
        if 'Margin' in idx or 'ROE' in idx:
            formatted.loc[idx] = formatted.loc[idx].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else x)
        else:
            formatted.loc[idx] = formatted.loc[idx].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
    
    return formatted

# Main app UI
def main():
    st.title("Taiwan Stock Financial Analysis System")
    
    st.sidebar.image("./img/stock.png", width=100)
    
    st.sidebar.header("Valuation Parameters")
    
    # Input parameters
    ticker = st.sidebar.text_input("Stock Code:", "2330.TW")
    
    industry = st.sidebar.selectbox(
        "Industry:",
        st.session_state.available_industries
    )
    
    forecast_years = st.sidebar.slider("Forecast Years:", 1, 15, 5)
    perpetual_growth = st.sidebar.slider("Perpetual Growth Rate:", 0.5, 5.0, 2.5) / 100  # Convert to decimal
    
    # Model options
    st.sidebar.subheader("Model Options")
    col1, col2 = st.sidebar.columns(2)
    use_ml = col1.checkbox("Use ML Model", True)
    use_dl = col2.checkbox("Use Deep Learning", True)
    use_industry = st.sidebar.checkbox("Apply Industry Adjustments", True)
    
    # Run valuation button
    if st.sidebar.button("Calculate Intrinsic Value", type="primary", use_container_width=True):
        result = run_valuation(ticker, industry, forecast_years, perpetual_growth, use_ml, use_dl, use_industry)
        if result:
            st.success(f"Valuation completed for {result['ticker']}")
    
    # Display results if available
    if st.session_state.valuation_results:
        result = st.session_state.valuation_results
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Summary", "Financial Data", "Growth Predictions", "Sensitivity Analysis", "Detailed Report"
        ])
        
        # Tab 1: Summary
        with tab1:
            st.header("Valuation Summary")
            
            # Display ticker and industry info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ticker", result['ticker'])
            with col2:
                if 'detected_industry' in result:
                    st.metric("Industry", result['detected_industry'])
            
            # Display valuations in columns
            if 'models' in result:
                st.subheader("Valuation Results")
                
                cols = st.columns(len(result['models']))
                
                for i, (model_name, price) in enumerate(result['models'].items()):
                    display_name = model_name.replace('_', ' ').title()
                    
                    # Find industry adjusted value if available
                    adjusted_value = None
                    adjustment_key = f"{model_name}_industry_adjusted"
                    if adjustment_key in result:
                        adjusted_value = result[adjustment_key]['adjusted_valuation']
                        
                    # Display the price with delta if adjusted value exists
                    with cols[i]:
                        if adjusted_value:
                            st.metric(
                                display_name, 
                                f"{price:,.2f}", 
                                f"{adjusted_value - price:+,.2f} (Adj)",
                                delta_color="normal"
                            )
                        else:
                            st.metric(display_name, f"{price:,.2f}")
            
            # Display model comparison chart
            if 'models' in result:
                st.subheader("Model Comparison")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                models = result['models']
                
                # Prepare data for plotting
                model_names = [name.replace('_', ' ').title() for name in models.keys()]
                prices = [price if price is not None else 0 for price in models.values()]
                
                # Create the bar chart
                bars = ax.bar(model_names, prices, color=['blue', 'green', 'purple'])
                
                # Add value labels on the bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{height:,.2f}',
                        ha='center', va='bottom'
                    )
                
                ax.set_title('Valuation by Model')
                ax.set_ylabel('Intrinsic Value')
                plt.xticks(rotation=15)
                plt.tight_layout()
                
                st.pyplot(fig)
            
            # Display industry adjustments if available
            adjustment_keys = [k for k in result if k.endswith('_industry_adjusted')]
            if adjustment_keys:
                st.subheader("Industry Adjustments")
                
                # Create a DataFrame for the adjustments
                adj_data = []
                for key in adjustment_keys:
                    adj = result[key]
                    model_name = key.replace('_industry_adjusted', '').replace('_', ' ').title()
                    
                    # Add to data
                    adj_data.append({
                        "Model": model_name,
                        "Base Value": f"{adj['base_valuation']:,.2f}",
                        "Adjusted Value": f"{adj['adjusted_valuation']:,.2f}",
                        "Adjustment Factor": f"{adj['total_adjustment']:.2f}x",
                        "Expected Return": f"{adj.get('expected_return', 0):.1%}" if adj.get('expected_return') else "N/A"
                    })
                    
                # Convert to DataFrame and display
                adj_df = pd.DataFrame(adj_data)
                st.dataframe(adj_df, hide_index=True, use_container_width=True)
        
        # Tab 2: Financial Data
        with tab2:
            st.header("Financial Data")
            
            if st.session_state.financial_data is not None:
                # Format and display financial metrics
                formatted_metrics = format_financial_metrics(st.session_state.financial_data)
                if formatted_metrics is not None:
                    st.dataframe(formatted_metrics, use_container_width=True)
                    
                    # Financial trends chart
                    st.subheader("Financial Trends")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    financial_data = st.session_state.financial_data
                    
                    # Plot revenue trend
                    if 'Total Revenue' in financial_data.index:
                        revenue = financial_data.loc['Total Revenue']
                        ax.plot(
                            revenue.index, revenue.values, 'b-o', 
                            label='Revenue'
                        )
                        
                    # Plot net income trend on the same chart
                    if 'Net Income' in financial_data.index:
                        net_income = financial_data.loc['Net Income']
                        ax.plot(
                            net_income.index, net_income.values, 'g-s', 
                            label='Net Income'
                        )
                    
                    ax.set_title('Revenue and Net Income Trends')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                else:
                    st.info("No financial metrics available for display.")
            else:
                st.info("Run a valuation first to see financial data.")
        
        # Tab 3: Growth Predictions
        with tab3:
            st.header("Growth Predictions")
            
            ml_predictions = result.get('ml_predictions', {})
            dl_predictions = result.get('dl_predictions', [])
            
            if ml_predictions:
                # Create a DataFrame for the predictions
                predictions_data = []
                
                # Year headers
                headers = ["Factor"] + [f"Year {i+1}" for i in range(min(5, len(ml_predictions.get('growth_rates', []))))]
                
                # Growth rates row
                growth_row = ["Growth Rates"]
                for rate in ml_predictions.get('growth_rates', [])[:5]:
                    growth_row.append(f"{rate:.1%}")
                predictions_data.append(growth_row)
                
                # Other factors
                factor_mapping = {
                    'capex_factors': 'CAPEX Factors',
                    'wc_factors': 'Working Capital',
                    'depr_factors': 'Depreciation',
                    'tax_factors': 'Tax Factors'
                }
                
                for key, display_name in factor_mapping.items():
                    if key in ml_predictions:
                        row = [display_name]
                        for val in ml_predictions[key][:5]:
                            row.append(f"{val:.2f}")
                        predictions_data.append(row)
                
                # DL growth rates if available
                if dl_predictions:
                    dl_row = ["DL Growth Rates"]
                    for rate in dl_predictions[:5]:
                        dl_row.append(f"{rate:.1%}")
                    predictions_data.append(dl_row)
                
                # Industry average growth from new calculated industry growth stats
                if 'industry_growth_stats' in result:
                    industry_growth_rates = result['industry_growth_stats']['average_growth_rates']
                    company_count = result['industry_growth_stats']['company_count']
                    
                    # Add a row with all growth rates for each year
                    industry_avg_row = ["Industry Avg Growth"]
                    for rate in industry_growth_rates[:min(5, len(industry_growth_rates))]:
                        industry_avg_row.append(f"{rate:.1%}")
                    predictions_data.append(industry_avg_row)
                    
                    # Add historical stats with more details
                    hist_mean = result['industry_growth_stats']['historical_mean_growth']
                    hist_median = result['industry_growth_stats']['historical_median_growth']
                    hist_stats_row = ["Industry Historical"]
                    hist_stats_row.append(f"Mean: {hist_mean:.1%}, Median: {hist_median:.1%}, Companies: {company_count}")
                    predictions_data.append(hist_stats_row)
                # Fallback to simple industry average growth benchmark
                elif 'industry_avg_growth' in result:
                    industry_row = ["Industry Avg Growth"]
                    for i in range(min(5, len(ml_predictions.get('growth_rates', [])))):
                        if i == 0:  # Only add the value to the first cell
                            industry_row.append(f"{result['industry_avg_growth']:.1%}")
                        else:
                            industry_row.append("")  # Empty cells for other years
                    predictions_data.append(industry_row)
                
                # Current stock price if available
                if 'current_price' in result and result['current_price']:
                    price_row = ["Current Stock Price"]
                    price_row.append(f"{result['current_price']:,.2f}")
                    # Add empty cells for other years to align table
                    for _ in range(min(5, len(ml_predictions.get('growth_rates', []))) - 1):
                        price_row.append("")
                    predictions_data.append(price_row)
                
                # Display the predictions table
                st.dataframe(pd.DataFrame(predictions_data, columns=headers), hide_index=True, use_container_width=True)
                
                # Growth rate chart
                st.subheader("Growth Rate Predictions")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                years = list(range(1, len(ml_predictions.get('growth_rates', []))+1))
                
                # Plot ML growth rates
                ax.plot(
                    years, 
                    [g*100 for g in ml_predictions.get('growth_rates', [])], 
                    'b-o', 
                    label='ML Growth'
                )
                
                # Add DL growth rates if available
                if dl_predictions:
                    ax.plot(
                        years[:len(dl_predictions)], 
                        [g*100 for g in dl_predictions], 
                        'r-s', 
                        label='DL Growth'
                    )
                
                # Add industry average growth line from new calculation if available
                if 'industry_growth_stats' in result:
                    industry_growth_rates = result['industry_growth_stats']['average_growth_rates']
                    ax.plot(
                        years[:len(industry_growth_rates)], 
                        [g*100 for g in industry_growth_rates], 
                        color='purple',
                        marker='o', 
                        linestyle='--',
                        label=f'Industry Avg (n={result["industry_growth_stats"]["company_count"]})'
                    )
                # Fallback to simple industry average if industry_growth_stats not available
                elif 'industry_avg_growth' in result:
                    industry_avg = result['industry_avg_growth'] * 100
                    ax.axhline(
                        y=industry_avg,
                        color='purple',
                        linestyle=':',
                        label=f'Industry Avg ({industry_avg:.1f}%)'
                    )
                
                # Add terminal growth line
                perpetual = perpetual_growth * 100
                ax.axhline(
                    y=perpetual, 
                    color='g', 
                    linestyle='--', 
                    label=f'Terminal ({perpetual:.1f}%)'
                )
                
                ax.set_title('Predicted Growth Rates')
                ax.set_xlabel('Year')
                ax.set_ylabel('Growth Rate (%)')
                ax.set_xticks(years)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # ML vs DL comparison if both available
                if dl_predictions and 'growth_rates' in ml_predictions:
                    st.subheader("ML vs Deep Learning Comparison")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create width for bars
                    bar_width = 0.35
                    index = np.arange(len(years[:len(dl_predictions)]))
                    
                    # Plot ML and DL side by side
                    ax.bar(
                        index - bar_width/2, 
                        [g*100 for g in ml_predictions['growth_rates'][:len(dl_predictions)]], 
                        bar_width, 
                        label='ML Model'
                    )
                    
                    ax.bar(
                        index + bar_width/2, 
                        [g*100 for g in dl_predictions], 
                        bar_width, 
                        label='Deep Learning'
                    )
                    
                    # Add industry average growth from new calculation if available
                    if 'industry_growth_stats' in result:
                        industry_growth_rates = result['industry_growth_stats']['average_growth_rates']
                        
                        # Only plot for the overlapping years
                        max_years = min(len(dl_predictions), len(industry_growth_rates))
                        ax.plot(
                            index[:max_years], 
                            [g*100 for g in industry_growth_rates[:max_years]], 
                            color='purple',
                            marker='o',
                            linestyle='--',
                            label=f'Industry Avg (n={result["industry_growth_stats"]["company_count"]})'
                        )
                    # Fallback to simple industry average
                    elif 'industry_avg_growth' in result:
                        industry_avg = result['industry_avg_growth'] * 100
                        ax.axhline(
                            y=industry_avg,
                            color='purple',
                            linestyle=':',
                            label=f'Industry Avg ({industry_avg:.1f}%)'
                        )
                    
                    ax.set_title('ML vs Deep Learning Growth Predictions')
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Growth Rate (%)')
                    ax.set_xticks(index)
                    ax.set_xticklabels([f'Year {y}' for y in range(1, len(dl_predictions)+1)])
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                # Add industry growth statistics details if available
                if 'industry_growth_stats' in result:
                    st.subheader("Industry Growth Statistics")
                    
                    stats = result['industry_growth_stats']
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Companies Analyzed", stats['company_count'])
                        st.metric("Historical Mean Growth", f"{stats['historical_mean_growth']:.1%}")
                    
                    with col2:
                        st.metric("Historical Median Growth", f"{stats['historical_median_growth']:.1%}")
                        st.metric("Growth Dispersion", f"{stats['growth_dispersion']:.1%}")
                    
                    # Add range information
                    min_growth, max_growth = stats['growth_range']
                    st.text(f"Growth Range: {min_growth:.1%} to {max_growth:.1%}")
            else:
                st.info("No growth predictions available. Enable ML model to see predictions.")
        
        # Tab 4: Sensitivity Analysis
        with tab4:
            st.header("Sensitivity Analysis")
            
            # Button to run sensitivity analysis
            if 'models' in result and 'standard_dcf' in result['models']:
                base_price = result['models']['standard_dcf']
                
                if st.button("Run Sensitivity Analysis", type="primary"):
                    sensitivity_results = run_sensitivity_analysis(st.session_state.ticker, base_price)
                
                # Display sensitivity results if available
                if st.session_state.sensitivity_results:
                    sensitivity = st.session_state.sensitivity_results
                    
                    # Create a DataFrame for the results
                    sens_data = []
                    
                    for factor, impact in sensitivity.items():
                        # Format the factor name for display
                        if factor == 'growth_rates':
                            display_name = 'Growth Rate'
                        elif factor == 'wacc':
                            display_name = 'WACC'
                        elif factor == 'perpetual_growth_rate':
                            display_name = 'Terminal Growth'
                        else:
                            display_name = factor.replace('_', ' ').title()
                            
                        # Format the impact values
                        high_impact = impact.get('high', 0) * 100
                        low_impact = impact.get('low', 0) * 100
                        
                        sens_data.append({
                            "Factor": display_name,
                            "+20% Impact": f"{high_impact:+.1f}%",
                            "-20% Impact": f"{low_impact:+.1f}%",
                            "+20% Price": f"{impact.get('high_val', 0):,.2f}",
                            "-20% Price": f"{impact.get('low_val', 0):,.2f}"
                        })
                    
                    # Display the table
                    sens_df = pd.DataFrame(sens_data)
                    st.dataframe(sens_df, hide_index=True, use_container_width=True)
                    
                    # Create sensitivity chart
                    st.subheader("Impact on Stock Price")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Extract factors and impact values
                    factors = [row["Factor"] for row in sens_data]
                    high_impacts = [impact.get('high', 0) * 100 for impact in sensitivity.values()]
                    low_impacts = [impact.get('low', 0) * 100 for impact in sensitivity.values()]
                    
                    # Create grouped bar chart
                    x = np.arange(len(factors))
                    width = 0.35
                    
                    ax.bar(x - width/2, high_impacts, width, label='+20%', color='green', alpha=0.6)
                    ax.bar(x + width/2, low_impacts, width, label='-20%', color='red', alpha=0.6)
                    
                    ax.set_ylabel('Impact on Stock Price (%)')
                    ax.set_title('Sensitivity Analysis')
                    ax.set_xticks(x)
                    ax.set_xticklabels(factors, rotation=45)
                    ax.legend()
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Tornado chart
                    st.subheader("Tornado Chart")
                    
                    # Sort factors by impact
                    total_impacts = [abs(h) + abs(l) for h, l in zip(high_impacts, low_impacts)]
                    sorted_indices = np.argsort(total_impacts)[::-1]  # Descending order
                    
                    sorted_factors = [factors[i] for i in sorted_indices]
                    sorted_high = [high_impacts[i] for i in sorted_indices]
                    sorted_low = [low_impacts[i] for i in sorted_indices]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    y_pos = np.arange(len(sorted_factors))
                    
                    # Create horizontal bars
                    ax.barh(y_pos, sorted_high, 0.4, label='+20%', color='green', alpha=0.6)
                    ax.barh(y_pos, sorted_low, 0.4, label='-20%', color='red', alpha=0.6)
                    
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(sorted_factors)
                    ax.set_xlabel('Impact on Stock Price (%)')
                    ax.set_title('Tornado Chart - Sensitivity Analysis')
                    ax.legend()
                    plt.tight_layout()
                    
                    st.pyplot(fig)
            else:
                st.info("Standard DCF valuation is required for sensitivity analysis.")
        
        # Tab 5: Detailed Report
        with tab5:
            st.header("Detailed Valuation Report")
            
            # Create a detailed report
            report = []
            report.append(f"## Detailed Valuation Results for {result['ticker']}")
            
            # Basic information
            if 'detected_industry' in result:
                report.append(f"**Detected Industry:** {result['detected_industry']}")
            report.append("")
            
            # Base valuations
            report.append("### Base Valuations")
            if 'models' in result:
                for model, price in result['models'].items():
                    display_name = model.replace('_', ' ').title()
                    report.append(f"**{display_name}:** {price:,.2f}")
            report.append("")
            
            # ML predictions if available
            if 'ml_predictions' in result:
                ml_pred = result['ml_predictions']
                report.append("### ML Growth Predictions")
                
                for factor, values in ml_pred.items():
                    display_name = factor.replace('_', ' ').title()
                    formatted_values = []
                    
                    # Check if values is an iterable (like a list) or a single value
                    if isinstance(values, (list, tuple, np.ndarray)):
                        for val in values:
                            if 'growth' in factor or 'rates' in factor:
                                formatted_values.append(f"{val:.1%}")
                            else:
                                formatted_values.append(f"{val:.2f}")
                    else:
                        # Handle single value case
                        val = values
                        if 'growth' in factor or 'rates' in factor:
                            formatted_values.append(f"{val:.1%}")
                        else:
                            formatted_values.append(f"{val:.2f}")
                    
                    report.append(f"**{display_name}:** {', '.join(formatted_values)}")
                report.append("")
            
            # DL predictions if available
            if 'dl_predictions' in result:
                dl_pred = result['dl_predictions']
                report.append("### Deep Learning Growth Predictions")
                
                formatted_values = [f"{val:.1%}" for val in dl_pred]
                report.append(f"**Growth Rates:** {', '.join(formatted_values)}")
                report.append("")
            
            # Industry adjustments if available
            industry_adj_keys = [k for k in result if k.endswith('_industry_adjusted')]
            if industry_adj_keys:
                report.append("### Industry Adjustments")
                
                for key in industry_adj_keys:
                    adj = result[key]
                    model_name = key.replace('_industry_adjusted', '').replace('_', ' ').title()
                    report.append(f"#### {model_name}")
                    report.append(f"* Base Valuation: {adj['base_valuation']:,.2f}")
                    report.append(f"* Adjusted Valuation: {adj['adjusted_valuation']:,.2f}")
                    report.append(f"* Adjustment Factor: {adj['total_adjustment']:.2f}x")
                    
                    if 'industry_factor' in adj:
                        report.append(f"* Industry Factor: {adj['industry_factor']:.2f}x")
                        
                    if 'return_factor' in adj:
                        report.append(f"* Return Factor: {adj['return_factor']:.2f}x")
                        
                    if 'expected_return' in adj and adj['expected_return'] is not None:
                        report.append(f"* Expected 6-Month Return: {adj['expected_return']:.1%}")
                        
                    report.append("")
            
            # Display the report
            st.markdown("\n".join(report))
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                if st.download_button(
                    "Download Report as Markdown", 
                    "\n".join(report),
                    file_name=f"{result['ticker']}_valuation_report.md",
                    mime="text/markdown"
                ):
                    st.success("Report downloaded!")
                    
            with col2:
                # Create CSV data
                data = {
                    'Ticker': result['ticker'],
                    'Industry': result.get('detected_industry', 'Unknown')
                }
                
                # Add model valuations
                if 'models' in result:
                    for model, price in result['models'].items():
                        display_name = model.replace('_', ' ').title()
                        data[display_name] = price
                
                # Add adjusted valuations
                for key in industry_adj_keys:
                    adj = result[key]
                    model_name = key.replace('_industry_adjusted', '').replace('_', ' ').title()
                    data[f"{model_name} Adjusted"] = adj['adjusted_valuation']
                    data[f"{model_name} Adjustment Factor"] = adj['total_adjustment']
                
                # Add ML predictions
                if 'ml_predictions' in result:
                    ml_pred = result['ml_predictions']
                    for factor, values in ml_pred.items():
                        display_name = factor.replace('_', ' ').title()
                        # Check if values is iterable
                        if isinstance(values, (list, tuple, np.ndarray)):
                            for i, val in enumerate(values):
                                data[f"{display_name} Year {i+1}"] = val
                        else:
                            # Handle single value case
                            data[f"{display_name}"] = values
                
                # Add DL predictions
                if 'dl_predictions' in result:
                    dl_pred = result['dl_predictions']
                    for i, val in enumerate(dl_pred):
                        data[f"DL Growth Year {i+1}"] = val
                
                # Create DataFrame for export
                df_export = pd.DataFrame([data])
                
                if st.download_button(
                    "Download Data as CSV",
                    df_export.to_csv(index=False),
                    file_name=f"{result['ticker']}_valuation_data.csv",
                    mime="text/csv"
                ):
                    st.success("Data downloaded!")
    else:
        st.info("Enter a stock code and click 'Calculate Intrinsic Value' to begin valuation.")
        
        # Display welcome information
        st.markdown("""
        ## Welcome to the Taiwan Stock Financial Analysis System
        
        This application provides comprehensive stock valuation using multiple models:
        
        - **Standard DCF Model**: Basic discounted cash flow valuation
        - **ML-Enhanced DCF**: Uses machine learning to predict growth factors
        - **ML+DL Ensemble**: Combines machine learning with deep learning for more robust predictions
        
        The system also provides:
        - Industry-specific valuation adjustments
        - Detailed financial data analysis
        - Growth prediction visualization
        - Sensitivity analysis
        
        ### Getting Started
        
        1. Enter a Taiwan stock code (e.g., 2330.TW) in the sidebar
        2. Select industry or use auto-detection
        3. Adjust forecast parameters if needed
        4. Click "Calculate Intrinsic Value" to run the valuation
        
        For best results, use Taiwan stock codes with the .TW or .TWO suffix.
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Taiwan Stock Financial Analysis System © 2025")

if __name__ == "__main__":
    main()