# streamlit_app.py
import streamlit as st
from dcf_model import DCFModel
from ml_predictor import GrowthPredictor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
forecast_years = st.number_input("Forecast Years", min_value=1, max_value=10, value=5)
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
use_ml = st.checkbox("Use ML model for predictions")

if st.button("Calculate Intrinsic Price"):
    n = forecast_years
    
    if use_ml:
        try:
            predictor = GrowthPredictor(stock_code)
            st.info("Training ML model for growth prediction...")
            
            with st.spinner('Loading historical data...'):
                # Add force_refresh parameter
                financial_data = predictor.data_fetcher.get_financial_data(stock_code, force_refresh=force_refresh)
                if predictor.train():
                    growth_prediction = predictor.predict_growth()
                    confidence = predictor.get_prediction_confidence()
                    
                    if growth_prediction is not None:
                        growth_factors = [float(growth_prediction)] * n
                        
                        # Display ML results
                        st.info("ML Model Prediction:")
                        st.write(f"Predicted Growth Rate: {growth_prediction:.2%}")
                        st.write(f"Prediction Confidence: {confidence:.1%}")
                        
                        # Show feature importance
                        if st.checkbox("Show Feature Importance"):
                            st.write("\nFeature Importance for Growth Prediction:")
                            st.dataframe(predictor.feature_importance)
                            
                        if predictor.cv_scores is not None:
                            st.write("Cross-validation scores:", 
                                    [f"{score:.2f}" for score in predictor.cv_scores])
                    else:
                        st.warning("Growth prediction failed, using manual inputs.")
                        growth_factors = parse_factors(manual_growth, n)
                else:
                    st.error("ML model training failed. Using manual inputs.")
                    growth_factors = parse_factors(manual_growth, n)
                    
            # Use manual inputs for other factors
            capex_factors = parse_factors(manual_capex, n)
            wc_factors = parse_factors(manual_wc, n)
            depr_factors = parse_factors(manual_depr, n)
            tax_factors = parse_factors(manual_tax, n)
            
        except Exception as e:
            st.error(f"Error in ML prediction: {str(e)}")
            growth_factors = parse_factors(manual_growth, n)
            capex_factors = parse_factors(manual_capex, n)
            wc_factors = parse_factors(manual_wc, n)
            depr_factors = parse_factors(manual_depr, n)
            tax_factors = parse_factors(manual_tax, n)
    else:
        growth_factors = parse_factors(manual_growth, n)
        capex_factors = parse_factors(manual_capex, n)
        wc_factors = parse_factors(manual_wc, n)
        depr_factors = parse_factors(manual_depr, n)
        tax_factors = parse_factors(manual_tax, n)

    # Ensure all factors are float lists of correct length
    all_factors_valid = all(
        isinstance(factors, list) and 
        len(factors) == n and 
        all(isinstance(x, float) for x in factors)
        for factors in [growth_factors, capex_factors, wc_factors, depr_factors, tax_factors]
    )

    if not all_factors_valid:
        st.error("Invalid factors generated. Please check inputs.")
        pass

    try:
        dcf = DCFModel(
            stock_code=stock_code,
            forecast_years=forecast_years,
            perpetual_growth_rate=perpetual_growth_rate,
            manual_growth_rates=growth_factors,
            manual_capex_factors=capex_factors,
            manual_wc_factors=wc_factors,
            manual_depr_factors=depr_factors,
            manual_tax_factors=tax_factors
        )
        
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
        st.error(f"Error in DCF calculation: {str(e)}")
