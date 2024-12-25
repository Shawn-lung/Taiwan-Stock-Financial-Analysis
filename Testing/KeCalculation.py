import yfinance as yf
import pandas as pd


class CostOfEquityModel:
    def __init__(self, stock_code, max_growth_rate=0.025):  
        self.stock = yf.Ticker(stock_code)
        self.stock_code = stock_code
        self.stock_dividends = self.stock.dividends
        self.max_growth_rate = max_growth_rate  
    def get_latest_stock_price(self):
        try:
            historical_data = self.stock.history(period="1d")
            latest_price = historical_data['Close'].iloc[-1]
            print(f"Latest stock price for {self.stock_code}: {latest_price:.2f}")
            return latest_price
        except Exception as e:
            print(f"Error fetching stock price for {self.stock_code}: {e}")
            return None

    def get_dividends_data(self):
        if self.stock_dividends.empty:
            print(f"No dividend data available for {self.stock_code}.")
            return None
        return self.stock_dividends

    def calculate_cagr(self):
        dividends = self.get_dividends_data()
        if dividends is None or len(dividends) < 2:
            print(f"Insufficient dividend data for {self.stock_code}. Cannot calculate CAGR.")
            return None

        start_dividend = dividends.iloc[0]
        end_dividend = dividends.iloc[-1]
        years = len(dividends) - 1

        if start_dividend <= 0 or end_dividend <= 0:
            print("Invalid dividend values for CAGR calculation.")
            return None

        cagr = (end_dividend / start_dividend) ** (1 / years) - 1

        adjusted_cagr = min(cagr, self.max_growth_rate)
        print(f"CAGR (Compound Annual Growth Rate): {cagr:.2%}")
        print(f"Adjusted Growth Rate (capped at {self.max_growth_rate:.2%}): {adjusted_cagr:.2%}")
        return adjusted_cagr

    def calculate_cost_of_equity_with_dividends(self):
        latest_price = self.get_latest_stock_price()
        growth_rate = self.calculate_cagr()

        if latest_price is None or growth_rate is None:
            return None

        dividends = self.get_dividends_data()
        if dividends is None:
            return None

        next_dividend = dividends.values[-1] * (1 + growth_rate)
        cost_of_equity = (next_dividend / latest_price) + growth_rate
        print(f"Calculated Cost of Equity (Ke) using dividends: {cost_of_equity:.2%}")
        return cost_of_equity
    def get_beta(self):
        try:
            beta = self.stock.info.get("beta", None)  
            if beta is not None:
                print(f"Beta for {self.stock_code}: {beta:.2f}")
                return beta
            else:
                print(f"Beta information not available for {self.stock_code}.")
                return 1
        except Exception as e:
            print(f"Error fetching Beta for {self.stock_code}: {e}")
            return None
    def calculate_cost_of_equity_with_capm(self, risk_free_rate=0.015):
        beta = self.get_beta()
        market_return = 0.075

        if beta is None or market_return is None:
            return None

        market_risk_premium = market_return - risk_free_rate
        cost_of_equity = risk_free_rate + beta * market_risk_premium
        print(f"Calculated Cost of Equity (Ke) using CAPM: {cost_of_equity:.2%}")
        return cost_of_equity


if __name__ == "__main__":
    stock_code = input('Enter stock code: ') + '.TW'  
    coe_model = CostOfEquityModel(stock_code)
    coe_model.calculate_cost_of_equity_with_dividends()
    coe_model.calculate_cost_of_equity_with_capm()
