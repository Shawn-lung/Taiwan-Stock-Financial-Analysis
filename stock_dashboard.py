import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import logging
from analyze_taiwan_stocks import TaiwanStockAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Taiwan Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize connection to database
@st.cache_resource
def get_connection():
    return sqlite3.connect("finance_data.db", check_same_thread=False)

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return TaiwanStockAnalyzer()

def load_stock_metrics(analyzer, stock_id):
    """Load financial metrics for a specific stock."""
    try:
        return analyzer.calculate_basic_metrics(stock_id)
    except Exception as e:
        st.error(f"Error loading metrics for {stock_id}: {e}")
        return pd.DataFrame()

def load_price_performance(analyzer, stock_id, days=365):
    """Load price performance data for a specific stock."""
    try:
        return analyzer.get_price_performance(stock_id, days=days)
    except Exception as e:
        st.error(f"Error loading price performance for {stock_id}: {e}")
        return None

def display_metric_card(label, value, delta=None, help_text=None):
    """Display a metric with consistent formatting."""
    if isinstance(value, float):
        if abs(value) < 0.01:
            formatted_value = f"{value:.4f}"
        elif abs(value) < 1:
            formatted_value = f"{value:.2f}"
        elif abs(value) > 1000000000:
            formatted_value = f"${value/1000000000:.2f}B"
        elif abs(value) > 1000000:
            formatted_value = f"${value/1000000:.2f}M"
        elif abs(value) > 1000:
            formatted_value = f"${value/1000:.2f}K"
        else:
            formatted_value = f"{value:.2f}"
    else:
        formatted_value = str(value)
        
    if delta is not None:
        if isinstance(delta, float):
            if abs(delta) < 0.01:
                delta = f"{delta:.4f}"
            else:
                delta = f"{delta:.2%}"
        st.metric(label, formatted_value, delta, help=help_text)
    else:
        st.metric(label, formatted_value, help=help_text)

def main():
    st.title("📈 Taiwan Stock Analysis Dashboard")
    
    # Check if database exists
    if not os.path.exists("finance_data.db"):
        st.error("Database file not found. Please make sure your data collection process has started.")
        return
    
    # Initialize analyzer
    analyzer = get_analyzer()
    
    # Dashboard layout with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Stock Explorer", 
        "Industry Analysis", 
        "Value Stocks", 
        "Momentum Stocks"
    ])
    
    # Tab 1: Stock Explorer
    with tab1:
        st.header("Stock Explorer")
        
        # Get all stocks
        conn = get_connection()
        stocks = pd.read_sql_query(
            """
            SELECT si.stock_id, si.stock_name, si.industry
            FROM stock_info si
            INNER JOIN (
                SELECT DISTINCT stock_id
                FROM collection_log
                WHERE status = 'success'
                GROUP BY stock_id
                HAVING 
                    SUM(CASE WHEN data_type = 'financial_statement' THEN 1 ELSE 0 END) > 0 AND
                    SUM(CASE WHEN data_type = 'balance_sheet' THEN 1 ELSE 0 END) > 0
            ) cl ON si.stock_id = cl.stock_id
            ORDER BY si.industry, si.stock_id
            """, 
            conn
        )
        
        # Select industry and stock
        col1, col2 = st.columns(2)
        
        with col1:
            industries = ['All Industries'] + sorted(stocks['industry'].unique().tolist())
            selected_industry = st.selectbox("Filter by Industry", industries)
        
        # Filter stocks by industry if needed
        if selected_industry != 'All Industries':
            filtered_stocks = stocks[stocks['industry'] == selected_industry]
        else:
            filtered_stocks = stocks
        
        with col2:
            stock_options = [f"{row['stock_id']} - {row['stock_name']}" for _, row in filtered_stocks.iterrows()]
            selected_stock = st.selectbox("Select Stock", stock_options)
        
        if selected_stock:
            stock_id = selected_stock.split(" - ")[0]
            stock_name = selected_stock.split(" - ")[1]
            
            st.subheader(f"{stock_id} - {stock_name}")
            
            # Get financial metrics
            metrics = load_stock_metrics(analyzer, stock_id)
            
            # Get price performance
            perf_1y = load_price_performance(analyzer, stock_id, days=365)
            perf_6m = load_price_performance(analyzer, stock_id, days=180)
            
            if not metrics.empty:
                # Show overview metrics
                st.write("### Financial Overview")
                
                # Create metric rows
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Latest revenue
                    if 'revenue' in metrics.columns and not metrics['revenue'].empty:
                        latest_revenue = metrics.iloc[-1]['revenue']
                        prev_revenue = metrics.iloc[-2]['revenue'] if len(metrics) > 1 else None
                        revenue_delta = ((latest_revenue / prev_revenue) - 1) if prev_revenue else None
                        display_metric_card("Revenue", latest_revenue, revenue_delta)
                    
                    # Latest ROE
                    if 'roe' in metrics.columns and not metrics['roe'].empty:
                        latest_roe = metrics.iloc[-1]['roe']
                        prev_roe = metrics.iloc[-2]['roe'] if len(metrics) > 1 else None
                        roe_delta = latest_roe - prev_roe if prev_roe else None
                        display_metric_card("Return on Equity", latest_roe, roe_delta)
                
                with col2:
                    # Net income
                    if 'net_income' in metrics.columns and not metrics['net_income'].empty:
                        latest_income = metrics.iloc[-1]['net_income']
                        prev_income = metrics.iloc[-2]['net_income'] if len(metrics) > 1 else None
                        income_delta = ((latest_income / prev_income) - 1) if prev_income and prev_income != 0 else None
                        display_metric_card("Net Income", latest_income, income_delta)
                    
                    # Net margin
                    if 'net_margin' in metrics.columns and not metrics['net_margin'].empty:
                        latest_margin = metrics.iloc[-1]['net_margin']
                        prev_margin = metrics.iloc[-2]['net_margin'] if len(metrics) > 1 else None
                        margin_delta = latest_margin - prev_margin if prev_margin else None
                        display_metric_card("Net Margin", latest_margin, margin_delta)
                
                with col3:
                    # Operating income
                    if 'operating_income' in metrics.columns and not metrics['operating_income'].empty:
                        latest_op = metrics.iloc[-1]['operating_income']
                        prev_op = metrics.iloc[-2]['operating_income'] if len(metrics) > 1 else None
                        op_delta = ((latest_op / prev_op) - 1) if prev_op and prev_op != 0 else None
                        display_metric_card("Operating Income", latest_op, op_delta)
                    
                    # Debt to equity
                    if 'debt_to_equity' in metrics.columns and not metrics['debt_to_equity'].empty:
                        latest_de = metrics.iloc[-1]['debt_to_equity']
                        display_metric_card("Debt to Equity", latest_de)
                
                with col4:
                    # Revenue growth
                    if 'revenue_growth' in metrics.columns and not metrics['revenue_growth'].empty:
                        latest_growth = metrics.iloc[-1]['revenue_growth']
                        avg_growth = metrics['revenue_growth'].mean()
                        growth_delta = latest_growth - avg_growth
                        display_metric_card("Revenue Growth", latest_growth, growth_delta, 
                                          "Latest year growth vs. average growth")
                    
                    # Operating margin
                    if 'operating_margin' in metrics.columns and not metrics['operating_margin'].empty:
                        latest_op_margin = metrics.iloc[-1]['operating_margin']
                        prev_op_margin = metrics.iloc[-2]['operating_margin'] if len(metrics) > 1 else None
                        op_margin_delta = latest_op_margin - prev_op_margin if prev_op_margin else None
                        display_metric_card("Operating Margin", latest_op_margin, op_margin_delta)
                
                # Show financial trends
                st.write("### Financial Trends")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Revenue trend
                    if 'revenue' in metrics.columns:
                        fig = px.line(metrics, x='year', y='revenue', title="Annual Revenue")
                        fig.update_layout(xaxis_title="Year", yaxis_title="Revenue")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Net margin trend
                    if 'net_margin' in metrics.columns:
                        fig = px.line(metrics, x='year', y='net_margin', title="Net Margin")
                        fig.update_layout(xaxis_title="Year", yaxis_title="Net Margin")
                        fig.update_yaxes(tickformat=".1%")
                        st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # ROE trend
                    if 'roe' in metrics.columns:
                        fig = px.line(metrics, x='year', y='roe', title="Return on Equity")
                        fig.update_layout(xaxis_title="Year", yaxis_title="ROE")
                        fig.update_yaxes(tickformat=".1%")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Operating margin trend
                    if 'operating_margin' in metrics.columns:
                        fig = px.line(metrics, x='year', y='operating_margin', title="Operating Margin")
                        fig.update_layout(xaxis_title="Year", yaxis_title="Operating Margin")
                        fig.update_yaxes(tickformat=".1%")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Show financial data table
                with st.expander("View Full Financial Data"):
                    st.dataframe(metrics)
            
            else:
                st.warning("No financial metrics available for this stock.")
            
            # Display price performance if available
            if perf_1y and 'price_data' in perf_1y:
                st.write("### Price Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # 1Y return
                    display_metric_card("1Y Return", perf_1y['total_return'], 
                                      help="Total return over the past year")
                
                with col2:
                    # 6M return
                    if perf_6m:
                        display_metric_card("6M Return", perf_6m['total_return'],
                                          help="Total return over the past 6 months")
                
                with col3:
                    # Volatility
                    display_metric_card("Volatility (Ann.)", perf_1y['volatility'],
                                      help="Annualized volatility of daily returns")
                
                with col4:
                    # Sharpe ratio
                    display_metric_card("Sharpe Ratio", perf_1y['sharpe_ratio'],
                                      help="Return per unit of risk (higher is better)")
                
                # Extract price data
                price_data = perf_1y['price_data']
                price_data['date'] = pd.to_datetime(price_data['date'])
                price_data = price_data.sort_values('date')
                
                # Create price chart with volume subplot
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.03, row_heights=[0.7, 0.3])
                
                # Add price line
                fig.add_trace(
                    go.Scatter(
                        x=price_data['date'],
                        y=price_data['close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='royalblue', width=1)
                    ),
                    row=1, col=1
                )
                
                # Add volume bars
                fig.add_trace(
                    go.Bar(
                        x=price_data['date'],
                        y=price_data['volume'],
                        name='Volume',
                        marker=dict(color='darkgray')
                    ),
                    row=2, col=1
                )
                
                # Update layout
                fig.update_layout(
                    title=f"1-Year Price Chart: {stock_id} - {stock_name}",
                    height=600,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                
                # Set y-axes titles
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            
    # Tab 2: Industry Analysis
    with tab2:
        st.header("Industry Analysis")
        
        # Get list of industries
        conn = get_connection()
        industries = pd.read_sql_query(
            """
            SELECT DISTINCT industry FROM stock_info
            WHERE industry != 'Other'
            ORDER BY industry
            """,
            conn
        )
        
        # Select industry to analyze
        selected_industry = st.selectbox("Select Industry", industries['industry'].tolist())
        
        if selected_industry:
            try:
                # Show loading spinner while analyzing
                with st.spinner(f"Analyzing {selected_industry} industry..."):
                    industry_analysis = analyzer.analyze_industry(selected_industry, top_n=10)
                
                if not industry_analysis.empty:
                    st.write(f"### Top Stocks in {selected_industry}")
                    
                    # Format the table for display
                    display_df = industry_analysis.copy()
                    
                    # Format percentage columns
                    percentage_cols = []
                    for col in display_df.columns:
                        if col.startswith('avg_') and not col.endswith('_rank'):
                            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
                            percentage_cols.append(col)
                    
                    # Extract columns for display
                    display_cols = ['stock_id', 'stock_name']
                    
                    # Add percentage columns that exist
                    metric_cols = []
                    for col in ['avg_revenue_growth', 'avg_operating_margin', 'avg_net_margin', 'avg_roe']:
                        if col in display_df.columns:
                            display_cols.append(col)
                            metric_cols.append(col)
                    
                    # Add rank columns
                    rank_cols = []
                    for col in display_df.columns:
                        if col.endswith('_rank'):
                            display_cols.append(col)
                            rank_cols.append(col)
                    
                    # Display as a dataframe
                    st.dataframe(display_df[display_cols])
                    
                    # Create visualizations
                    if metric_cols and len(industry_analysis) > 1:
                        st.write("### Industry Metrics Comparison")
                        
                        # Create a metrics comparison chart
                        comp_data = industry_analysis[['stock_name'] + metric_cols].copy()
                        
                        # Create a bar chart for each metric
                        for metric in metric_cols:
                            nice_name = metric.replace('avg_', '').replace('_', ' ').title()
                            
                            fig = px.bar(
                                comp_data.sort_values(metric, ascending=False),
                                x='stock_name',
                                y=metric,
                                title=f"{nice_name} by Stock",
                                labels={'stock_name': 'Stock', metric: nice_name}
                            )
                            
                            # Format Y-axis as percentage
                            fig.update_layout(yaxis_tickformat='.1%')
                            
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No analysis data available for {selected_industry} industry.")
            except Exception as e:
                st.error(f"Error analyzing {selected_industry} industry: {e}")
    
    # Tab 3: Value Stocks
    with tab3:
        st.header("Value Stock Screener")
        
        # Parameters for value screening
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_roe = st.slider("Minimum ROE", 0.0, 0.5, 0.12, 0.01, format="%.2f")
        
        with col2:
            max_debt_equity = st.slider("Maximum Debt-to-Equity", 0.0, 3.0, 1.0, 0.1)
        
        with col3:
            top_n = st.slider("Number of Stocks", 5, 50, 20)
        
        # Run value stock screening
        if st.button("Find Value Stocks"):
            with st.spinner("Screening for value stocks..."):
                value_stocks = analyzer.find_value_stocks(
                    min_roe=min_roe,
                    max_debt_equity=max_debt_equity,
                    top_n=top_n
                )
                
                if not value_stocks.empty:
                    st.write(f"### Found {len(value_stocks)} Value Stock Candidates")
                    
                    # Format the dataframe
                    display_df = value_stocks.copy()
                    
                    # Format percentage columns
                    for col in display_df.columns:
                        if col.startswith('avg_'):
                            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
                    
                    # Extract columns for display
                    display_cols = ['stock_id', 'stock_name', 'industry']
                    
                    # Add percentage columns that exist
                    for col in ['avg_roe', 'avg_debt_to_equity', 'avg_revenue_growth', 'avg_operating_margin']:
                        if col in display_df.columns:
                            display_cols.append(col)
                    
                    # Display as a dataframe
                    st.dataframe(display_df[display_cols])
                    
                    # Create industry breakdown
                    if len(value_stocks) > 1:
                        industry_counts = value_stocks['industry'].value_counts()
                        
                        # Create pie chart
                        fig = px.pie(
                            names=industry_counts.index,
                            values=industry_counts.values,
                            title="Value Stocks by Industry"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No value stocks found matching the criteria.")
    
    # Tab 4: Momentum Stocks
    with tab4:
        st.header("Momentum Stock Screener")
        
        # Parameters for momentum screening
        col1, col2 = st.columns(2)
        
        with col1:
            min_return = st.slider("Minimum 6-month Return", 0.0, 1.0, 0.1, 0.01, format="%.2f")
        
        with col2:
            max_momentum = st.slider("Maximum Stocks", 5, 50, 20)
        
        # Run momentum stock screening
        if st.button("Find Momentum Stocks"):
            with st.spinner("Screening for momentum stocks..."):
                momentum_stocks = analyzer.find_momentum_stocks(
                    min_return=min_return,
                    max_stocks=max_momentum
                )
                
                if not momentum_stocks.empty:
                    st.write(f"### Found {len(momentum_stocks)} Momentum Stock Candidates")
                    
                    # Format the dataframe
                    display_df = momentum_stocks.copy()
                    
                    # Format percentage and ratio columns
                    for col in ['return_6m', 'annualized_return']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
                    
                    # Extract columns for display
                    display_cols = ['stock_id', 'stock_name', 'industry']
                    
                    # Add metrics columns that exist
                    for col in ['return_6m', 'annualized_return', 'sharpe_ratio', 'max_drawdown']:
                        if col in display_df.columns:
                            display_cols.append(col)
                    
                    # Display as a dataframe
                    st.dataframe(display_df[display_cols])
                    
                    # Create return comparison chart
                    if len(momentum_stocks) > 1 and 'return_6m' in momentum_stocks.columns:
                        fig = px.bar(
                            momentum_stocks.head(10),
                            x='stock_name',
                            y='return_6m',
                            title="Top 10 Stocks by 6-Month Return",
                            labels={'stock_name': 'Stock', 'return_6m': '6-Month Return'},
                            text_auto='.1%'
                        )
                        fig.update_layout(yaxis_tickformat='.1%')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create industry breakdown
                        industry_counts = momentum_stocks['industry'].value_counts()
                        
                        # Create pie chart
                        fig = px.pie(
                            names=industry_counts.index,
                            values=industry_counts.values,
                            title="Momentum Stocks by Industry"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No momentum stocks found matching the criteria.")

# Add necessary imports
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Run the app
if __name__ == "__main__":
    main()
