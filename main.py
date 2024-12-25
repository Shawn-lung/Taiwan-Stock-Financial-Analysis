from DiscountedCashFlow import DCFModel
import yfinance as yf

if __name__ == "__main__":
    manual_rates = [0.2, 0.15, 0.1, 0.05, 0.05]  
    dcf = DCFModel(
        "8069.TWO",
        forecast_years=5,
        perpetual_growth_rate=0.025,
        manual_growth_rates=manual_rates
    )
    estimated_price = dcf.calculate_stock_price()
    print(f"Estimated Price (Manual) = {estimated_price}")
