import time
from background_data_collector import BackgroundDataCollector
import datetime

# Calculate the start time (1 hour from now)
start_time = datetime.datetime.now() + datetime.timedelta(hours=1)
print(f"Data collection will start at {start_time.strftime('%Y-%m-%d %H:%M:%S')} (in 1 hour)")
print("You can safely leave now. The script will wait and then start collection automatically.")

# Wait for 1 hour before starting the collector
time.sleep(3600)  # 1 hour delay

# Create collector with 1-hour interval
collector = BackgroundDataCollector(
    db_path="finance_data.db",
    collection_interval=1  # hours
)

# Start collection
print(f"Starting data collection now: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
