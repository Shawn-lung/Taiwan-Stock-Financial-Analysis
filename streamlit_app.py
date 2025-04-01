# streamlit_app.py
import streamlit as st
from dcf_model import DCFModel
from ml_predictor import GrowthPredictor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize session state defaults for manual inputs if not already set
if 'manual_growth' not in st.session_state:
    st.session_state.manual_growth = "0.2,0.1,0.1,0.05,0.05"
if 'manual_capex' not in st.session_state:
    st.session_state.manual_capex = "0.2,0.1,0, -0.05, -0.05"
if 'manual_wc' not in st.session_state:
    st.session_state.manual_wc = "0.1,0.05,0,0,0"
if 'manual_depr' not in st.session_state:
    st.session_state.manual_depr = "0,0.05,0.05,0,-0.1"
if 'manual_tax' not in st.session_state:
    st.session_state.manual_tax = "0,0,0,0.05,-0.1"

st.title("Discounted Cash Flow (DCF) Valuation Tool")
st.write("Input your parameters to calculate the intrinsic stock price.")

# Add refresh button
force_refresh = st.checkbox("Force refresh data (ignore cache)")

# Stock and Forecast Inputs
stock_code = st.text_input("Stock Code (e.g., 2330.TW)", "2330.TW")
forecast_years = st.number_input("Forecast Years", min_value=1, max_value=15, value=5)
perpetual_growth_rate = st.number_input("Perpetual Growth Rate", value=0.025, format="%.3f")

st.subheader("Manual Factor Inputs (Comma-Separated Values)")
# Display text inputs with current session state values
manual_growth = st.text_input("Revenue Growth Rates (e.g., 0.2,0.1,0.1,0.05,0.05)", st.session_state.manual_growth)
manual_capex = st.text_input("CAPEX Factors (e.g., 0.2,0.1,0,-0.05,-0.05)", st.session_state.manual_capex)
manual_wc = st.text_input("Working Capital Factors (e.g., 0.1,0.05,0,0,0)", st.session_state.manual_wc)
manual_depr = st.text_input("Depreciation Factors (e.g., 0,0.05,0.05,0,-0.1)", st.session_state.manual_depr)
manual_tax = st.text_input("Tax Rate Factors (e.g., 0,0,0,0.05,-0.1)", st.session_state.manual_tax)

# Add a button to clear manual inputs
if st.button("Clear Manual Inputs"):
    st.session_state.manual_growth = ""
    st.session_state.manual_capex = ""
    st.session_state.manual_wc = ""
    st.session_state.manual_depr = ""
    st.session_state.manual_tax = ""
    st.success("Manual inputs have been cleared. Auto-estimation will be used if fields are left blank.")
    st.rerun()  # Changed from st.experimental_rerun()

def parse_factors(text, expected_length):
    try:
        # If text is empty, return a list of zeros
        if not text.strip():
            return [0.0] * expected_length
        factors = [float(x.strip()) for x in text.split(",")]
        if len(factors) != expected_length:
            st.warning(f"Expected {expected_length} factors but received {len(factors)}. Using default zeros.")
            return [0.0] * expected_length
        return factors
    except Exception as e:
        st.warning("Error parsing factors. Using default zeros.")
        return [0.0] * expected_length

st.subheader("Machine Learning Predictions")
use_ml = st.checkbox("Use ML model for predictions", value=True)
use_dl = st.checkbox("Use Deep Learning for enhanced forecasts", value=True)

st.subheader("Prediction Method")
prediction_method = st.radio(
    "Choose prediction method:",
    ["ML/DL Predictions", "Manual Inputs"]
)

if st.button("Calculate Intrinsic Price"):
    try:
        if prediction_method == "ML/DL Predictions":
            predictor = GrowthPredictor(stock_code)
            st.info("Training ML and Deep Learning models for factor predictions...")
            
            with st.spinner('Processing...'):
                predictions = predictor.predict_all_factors(
                    forecast_years=forecast_years,
                    terminal_growth=perpetual_growth_rate,
                    use_deep_learning=use_dl
                )
                
                if predictions:
                    st.write(f"ML/DL Predictions for {forecast_years} years:")
                    # Display predictions with line chart
                    for factor, values in predictions.items():
                        if factor != 'forecast_years':
                            st.write(f"{factor}: {[f'{x:.1%}' for x in values]}")
                            
                            # Create line chart for growth rates
                            if factor == 'growth_rates':
                                df = pd.DataFrame({
                                    'Year': range(1, len(values) + 1),
                                    'Growth Rate': [x * 100 for x in values]
                                })
                                fig = plt.figure(figsize=(10, 4))
                                plt.plot(df['Year'], df['Growth Rate'], marker='o')
                                plt.axhline(y=perpetual_growth_rate * 100, color='r', linestyle='--', 
                                          label=f'Terminal Growth ({perpetual_growth_rate:.1%})')
                                plt.xlabel('Forecast Year')
                                plt.ylabel('Growth Rate (%)')
                                plt.title('Predicted Growth Rates')
                                plt.legend()
                                st.pyplot(fig)

                    # Create DCF model with predicted factors
                    dcf = DCFModel(
                        stock_code=stock_code,
                        forecast_years=forecast_years,
                        perpetual_growth_rate=perpetual_growth_rate,
                        manual_growth_rates=predictions['growth_rates'],
                        manual_capex_factors=predictions['capex_factors'],
                        manual_wc_factors=predictions['wc_factors'],
                        manual_depr_factors=predictions['depr_factors'],
                        manual_tax_factors=predictions['tax_factors']
                    )
                    # Calculate and display results
                    price = dcf.calculate_stock_price()
                    
                    if price is not None:
                        st.success(f"Estimated Stock Price: {price:.2f} per share")
                        
                        st.subheader("Comprehensive Sensitivity Analysis")
                        sensitivity_results = dcf.perform_comprehensive_sensitivity()
                        if sensitivity_results:
                            st.write("Impact on Stock Price (% change):")
                            
                            # Create sensitivity chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            factors = list(sensitivity_results.keys())
                            high_impacts = [results['high'] * 100 for results in sensitivity_results.values()]
                            low_impacts = [results['low'] * 100 for results in sensitivity_results.values()]
                            
                            x = range(len(factors))
                            width = 0.35
                            
                            ax.bar([i - width/2 for i in x], high_impacts, width, label='+20%', color='green', alpha=0.6)
                            ax.bar([i + width/2 for i in x], low_impacts, width, label='-20%', color='red', alpha=0.6)
                            
                            ax.set_ylabel('Impact on Stock Price (%)')
                            ax.set_title('Sensitivity Analysis')
                            ax.set_xticks(x)
                            ax.set_xticklabels(factors, rotation=45)
                            ax.legend()
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Show detailed numbers in a table
                            df_sensitivity = pd.DataFrame(sensitivity_results).T
                            df_sensitivity.columns = ['+20%', '-20%']
                            df_sensitivity = df_sensitivity.applymap(lambda x: f"{x*100:.1f}%")
                            st.dataframe(df_sensitivity)
                    else:
                        st.error("Failed to calculate the stock price. Please check the input parameters.")
                
        else:
            # For manual inputs, use fixed 5-year forecast
            n = 5
            # Parse manual inputs
            growth_factors = parse_factors(manual_growth, n)
            capex_factors = parse_factors(manual_capex, n)
            wc_factors = parse_factors(manual_wc, n)
            depr_factors = parse_factors(manual_depr, n)
            tax_factors = parse_factors(manual_tax, n)

            # Validate factors
            all_factors_valid = all(
                isinstance(factors, list) and 
                len(factors) == n and 
                all(isinstance(x, float) for x in factors)
                for factors in [growth_factors, capex_factors, wc_factors, depr_factors, tax_factors]
            )

            if not all_factors_valid:
                st.error("Invalid factors generated. Please check inputs.")
            else:
                # Create DCF model with validated factors
                dcf = DCFModel(
                    stock_code=stock_code,
                    forecast_years=n,
                    perpetual_growth_rate=perpetual_growth_rate,
                    manual_growth_rates=growth_factors,
                    manual_capex_factors=capex_factors,
                    manual_wc_factors=wc_factors,
                    manual_depr_factors=depr_factors,
                    manual_tax_factors=tax_factors
                )
                
                # Calculate and display results
                price = dcf.calculate_stock_price()
                
                if price is not None:
                    st.success(f"Estimated Stock Price: {price:.2f} per share")
                    
                    st.subheader("Comprehensive Sensitivity Analysis")
                    sensitivity_results = dcf.perform_comprehensive_sensitivity()
                    if sensitivity_results:
                        st.write("Impact on Stock Price (% change):")
                        
                        # Create sensitivity chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        factors = list(sensitivity_results.keys())
                        high_impacts = [results['high'] * 100 for results in sensitivity_results.values()]
                        low_impacts = [results['low'] * 100 for results in sensitivity_results.values()]
                        
                        x = range(len(factors))
                        width = 0.35
                        
                        ax.bar([i - width/2 for i in x], high_impacts, width, label='+20%', color='green', alpha=0.6)
                        ax.bar([i + width/2 for i in x], low_impacts, width, label='-20%', color='red', alpha=0.6)
                        
                        ax.set_ylabel('Impact on Stock Price (%)')
                        ax.set_title('Sensitivity Analysis')
                        ax.set_xticks(x)
                        ax.set_xticklabels(factors, rotation=45)
                        ax.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show detailed numbers in a table
                        df_sensitivity = pd.DataFrame(sensitivity_results).T
                        df_sensitivity.columns = ['+20%', '-20%']
                        df_sensitivity = df_sensitivity.applymap(lambda x: f"{x*100:.1f}%")
                        st.dataframe(df_sensitivity)
                else:
                    st.error("Failed to calculate the stock price. Please check the input parameters.")
                
    except Exception as e:
        st.error(f"Error in calculation: {str(e)}")
        logger.error(f"Calculation error: {str(e)}", exc_info=True)
