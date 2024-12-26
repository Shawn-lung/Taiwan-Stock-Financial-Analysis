from DiscountedCashFlow import DCFModel
import yfinance as yf

if __name__ == "__main__":
    dcf = DCFModel(
        "3481.TW",
        forecast_years=5,
        perpetual_growth_rate=0.022,
        #manual_growth_rates=[0.4, 0.2, 0.15, 0.15, 0.15],
        #manual_capex_factors=[0.5, 0.2, 0.15, 0.15, 0.15],
        #manual_wc_factors=[0.1, 0.05, 0, 0, 0],
        #manual_depr_factors=[0.2, 0.1, 0, 0, 0],
        #manual_opincome_factors=[0.25, 0.15, 0.1, 0.08, 0.05],
        #manual_tax_factors=[0, 0, 0, 0.05, -0.1]
    )

    price = dcf.calculate_stock_price()
    print(f"Estimated Price= {price}")