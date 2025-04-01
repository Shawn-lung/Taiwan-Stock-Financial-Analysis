import time
from background_data_collector import BackgroundDataCollector

# Create collector with 1-hour interval
collector = BackgroundDataCollector(
    db_path="finance_data.db",
    collection_interval=1  # hours
)

# Start collection
collector.start_scheduler()

try:
    print("Background collector started. Will run for 6 days...")
    # Run for 6 days (6 * 24 * 60 * 60 seconds)
    time.sleep(6 * 24 * 60 * 60)
finally:
    # This block will execute even if the computer restarts
    # Make sure to export data at the end
    collector.export_to_csv("exported_data_6days")
    
    # Show statistics
    stats = collector.get_db_stats()
    print("\nDatabase Statistics after 6 days:")
    print(f"Total stocks: {stats.get('stock_info_count', 0)}")
    print(f"Stocks with complete data: {stats.get('stocks_with_complete_data', 0)}")
