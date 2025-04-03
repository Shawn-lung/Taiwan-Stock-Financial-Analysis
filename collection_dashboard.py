import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from check_collection_progress import CollectionMonitor

# Set page configuration
st.set_page_config(
    page_title="Taiwan Stock Data Collection Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize connection to database
@st.cache_resource
def get_connection():
    return sqlite3.connect("finance_data.db", check_same_thread=False)

def main():
    st.title("📊 Taiwan Stock Data Collection Monitor")
    
    # Check if database exists
    if not os.path.exists("finance_data.db"):
        st.error("Database file not found. Please make sure your data collection process has started.")
        return
    
    # Initialize collection monitor
    monitor = CollectionMonitor()
    
    # Get basic stats
    progress = monitor.get_collection_progress()
    industry_coverage = monitor.get_industry_coverage()
    
    if not progress:
        st.error("Could not retrieve collection progress. Check your database connection.")
        return
        
    # Dashboard layout with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", 
        "Industry Coverage", 
        "Collection Progress", 
        "Data Explorer"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Collection Overview")
        
        # Create metrics at the top
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stocks", f"{progress['total_stocks']:,}")
        
        with col2:
            st.metric("Stocks with Complete Data", 
                      f"{progress['complete_stocks']:,}",
                      f"{progress['completion_percentage']['complete_stocks']}%")
        
        with col3:
            # Calculate overall success rate
            if 'collection_rate' in progress and not progress['collection_rate'].empty:
                total_attempts = progress['collection_rate']['total_attempts'].sum()
                successful = progress['collection_rate']['successful_attempts'].sum()
                success_rate = (successful / total_attempts) * 100 if total_attempts > 0 else 0
                st.metric("Collection Success Rate", f"{success_rate:.1f}%")
            else:
                st.metric("Collection Success Rate", "N/A")
        
        with col4:
            # Last collection date
            if 'collection_rate' in progress and not progress['collection_rate'].empty:
                last_date = progress['collection_rate']['collection_date'].max()
                st.metric("Last Collection Date", last_date)
            else:
                st.metric("Last Collection Date", "N/A")
        
        # Progress bars for each data type
        st.subheader("Data Type Completion")
        col1, col2 = st.columns(2)
        
        with col1:
            fs_pct = progress['completion_percentage']['financial_statements']
            bs_pct = progress['completion_percentage']['balance_sheets']
            st.progress(fs_pct/100, text=f"Financial Statements: {fs_pct}%")
            st.progress(bs_pct/100, text=f"Balance Sheets: {bs_pct}%")
        
        with col2:
            cf_pct = progress['completion_percentage']['cash_flows']
            price_pct = progress['completion_percentage']['price_data']
            st.progress(cf_pct/100, text=f"Cash Flows: {cf_pct}%")
            st.progress(price_pct/100, text=f"Price Data: {price_pct}%")
        
        # Recently collected stocks
        st.subheader("Recently Collected Stocks")
        if 'recent_collections' in progress and not progress['recent_collections'].empty:
            st.dataframe(
                progress['recent_collections'],
                hide_index=True,
                column_config={
                    "stock_id": "Stock ID",
                    "stock_name": "Company Name",
                    "industry": "Industry",
                    "last_collected": st.column_config.DatetimeColumn(
                        "Last Collection Time",
                        format="YYYY-MM-DD HH:mm:ss"
                    ),
                    "data_types_collected": st.column_config.NumberColumn(
                        "Data Types Collected",
                        help="Number of data types successfully collected (max 4)"
                    )
                },
                use_container_width=True
            )
        else:
            st.info("No recently collected stocks found.")
    
    # Tab 2: Industry Coverage
    with tab2:
        st.header("Industry Coverage Analysis")
        
        if industry_coverage is not None and not industry_coverage.empty:
            # Filter options
            st.subheader("Industry Data Coverage")
            min_stocks = int(st.slider("Minimum stocks per industry", 1, 100, 5))
            filtered_industries = industry_coverage[industry_coverage['total_stocks'] >= min_stocks]
            
            # Prepare data for bar chart
            industries = filtered_industries['industry'].tolist()
            total_stocks = filtered_industries['total_stocks'].tolist()
            fs_pct = filtered_industries['fs_count_pct'].tolist()
            bs_pct = filtered_industries['bs_count_pct'].tolist()
            cf_pct = filtered_industries['cf_count_pct'].tolist()
            price_pct = filtered_industries['price_count_pct'].tolist()
            
            # Use plotly for interactive charts
            fig = go.Figure()
            fig.add_trace(go.Bar(x=industries, y=fs_pct, name="Financial Statements", marker_color='#1f77b4'))
            fig.add_trace(go.Bar(x=industries, y=bs_pct, name="Balance Sheets", marker_color='#ff7f0e'))
            fig.add_trace(go.Bar(x=industries, y=cf_pct, name="Cash Flows", marker_color='#2ca02c'))
            fig.add_trace(go.Bar(x=industries, y=price_pct, name="Price Data", marker_color='#d62728'))
            
            fig.update_layout(
                title="Data Coverage by Industry",
                xaxis_title="Industry",
                yaxis_title="Coverage (%)",
                barmode='group',
                height=600,
                xaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Industry stock breakdown table
            st.subheader("Industry Stock Breakdown")
            st.dataframe(
                filtered_industries.sort_values('total_stocks', ascending=False),
                hide_index=True,
                column_config={
                    "industry": "Industry",
                    "total_stocks": st.column_config.NumberColumn("Total Stocks"),
                    "fs_count": st.column_config.NumberColumn("Fin. Statements"),
                    "bs_count": st.column_config.NumberColumn("Balance Sheets"),
                    "cf_count": st.column_config.NumberColumn("Cash Flows"),
                    "price_count": st.column_config.NumberColumn("Price Data"),
                    "fs_count_pct": st.column_config.ProgressColumn("FS %", format="%.1f%%", min_value=0, max_value=100),
                    "bs_count_pct": st.column_config.ProgressColumn("BS %", format="%.1f%%", min_value=0, max_value=100),
                    "cf_count_pct": st.column_config.ProgressColumn("CF %", format="%.1f%%", min_value=0, max_value=100),
                    "price_count_pct": st.column_config.ProgressColumn("Price %", format="%.1f%%", min_value=0, max_value=100)
                },
                use_container_width=True
            )
        else:
            st.info("Industry coverage data not available.")
    
    # Tab 3: Collection Progress
    with tab3:
        st.header("Collection Progress Over Time")
        
        if 'collection_rate' in progress and not progress['collection_rate'].empty:
            collection_rate = progress['collection_rate'].copy()
            collection_rate['success_rate'] = (collection_rate['successful_attempts'] / collection_rate['total_attempts']) * 100
            
            # Prepare date column
            collection_rate['collection_date'] = pd.to_datetime(collection_rate['collection_date'])
            
            # Collection attempts line chart
            fig1 = px.line(
                collection_rate, 
                x="collection_date", 
                y=["total_attempts", "successful_attempts"],
                title="Daily Collection Attempts",
                labels={"value": "Number of Attempts", "collection_date": "Date"},
                color_discrete_sequence=["#1f77b4", "#2ca02c"]
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Success rate line chart
            fig2 = px.line(
                collection_rate, 
                x="collection_date", 
                y="success_rate",
                title="Daily Success Rate",
                labels={"success_rate": "Success Rate (%)", "collection_date": "Date"},
                color_discrete_sequence=["#2ca02c"]
            )
            fig2.update_traces(mode="lines+markers")
            fig2.update_layout(yaxis_range=[0, 105])
            st.plotly_chart(fig2, use_container_width=True)
            
            # Data collection table
            st.subheader("Daily Collection Statistics")
            st.dataframe(
                collection_rate.sort_values('collection_date', ascending=False),
                hide_index=True,
                column_config={
                    "collection_date": st.column_config.DateColumn("Date"),
                    "total_attempts": st.column_config.NumberColumn("Total Attempts"),
                    "successful_attempts": st.column_config.NumberColumn("Successful"),
                    "unique_stocks": st.column_config.NumberColumn("Unique Stocks"),
                    "success_rate": st.column_config.ProgressColumn("Success Rate", format="%.1f%%", min_value=0, max_value=100)
                },
                use_container_width=True
            )
        else:
            st.info("No collection progress data available.")
    
    # Tab 4: Data Explorer
    with tab4:
        st.header("Data Explorer")
        
        # Select industry
        conn = get_connection()
        industries = pd.read_sql_query("SELECT DISTINCT industry FROM stock_info ORDER BY industry", conn)
        selected_industry = st.selectbox("Select Industry", industries['industry'].tolist())
        
        # Get stocks in selected industry
        if selected_industry:
            stocks = pd.read_sql_query(
                "SELECT stock_id, stock_name FROM stock_info WHERE industry = ? ORDER BY stock_id",
                conn,
                params=(selected_industry,)
            )
            
            if stocks.empty:
                st.info(f"No stocks found in {selected_industry} industry.")
            else:
                # Display stock selection
                stock_options = [f"{row['stock_id']} - {row['stock_name']}" for _, row in stocks.iterrows()]
                selected_stock = st.selectbox("Select Stock", stock_options)
                
                if selected_stock:
                    stock_id = selected_stock.split(" - ")[0]
                    
                    # Get financial data samples
                    sample_data = monitor.sample_financial_data(stock_id)
                    
                    if sample_data:
                        st.subheader(f"Financial Data for {selected_stock}")
                        
                        # Create price chart if data available
                        if not sample_data['price_data'].empty:
                            price_data = sample_data['price_data'].copy()
                            price_data['date'] = pd.to_datetime(price_data['date'])
                            price_data = price_data.sort_values('date')
                            
                            fig = px.line(
                                price_data, 
                                x="date", 
                                y="close",
                                title=f"Stock Price - {selected_stock}",
                                labels={"close": "Close Price", "date": "Date"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Create tabs for different data types
                        data_tab1, data_tab2, data_tab3, data_tab4 = st.tabs([
                            "Financial Statements", 
                            "Balance Sheets", 
                            "Cash Flows", 
                            "Price Data"
                        ])
                        
                        with data_tab1:
                            if not sample_data['financial_statements'].empty:
                                st.dataframe(sample_data['financial_statements'], use_container_width=True)
                            else:
                                st.info("No financial statement data available.")
                        
                        with data_tab2:
                            if not sample_data['balance_sheets'].empty:
                                st.dataframe(sample_data['balance_sheets'], use_container_width=True)
                            else:
                                st.info("No balance sheet data available.")
                        
                        with data_tab3:
                            if not sample_data['cash_flows'].empty:
                                st.dataframe(sample_data['cash_flows'], use_container_width=True)
                            else:
                                st.info("No cash flow data available.")
                        
                        with data_tab4:
                            if not sample_data['price_data'].empty:
                                st.dataframe(sample_data['price_data'], use_container_width=True)
                            else:
                                st.info("No price data available.")
                    else:
                        st.info(f"No data available for {selected_stock}.")

if __name__ == "__main__":
    main()
