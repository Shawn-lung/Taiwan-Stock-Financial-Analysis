import logging
import pandas as pd
import yfinance as yf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WACCModel:
    """
    A model to calculate the Weighted Average Cost of Capital (WACC) for a stock.

    Attributes:
        stock_code (str): The ticker symbol of the stock.
        max_growth_rate (float): Maximum growth rate used in calculations.
        risk_free_rate (float): The risk-free rate for CAPM.
        market_return (float): Expected market return.
        stock_dividends (pd.Series): Historical dividend data.
        annual_financials (DataFrame): Annual financial statement data.
        share_outstanding (int): Number of outstanding shares.
    """

    def __init__(self, stock_code, max_growth_rate=0.025, risk_free_rate=0.0161, market_return=0.075):
        self.stock_code = stock_code
        self.max_growth_rate = max_growth_rate  
        self.risk_free_rate = risk_free_rate  
        self.market_return = market_return  

        self.stock = yf.Ticker(stock_code)
        self.stock_dividends = self.stock.dividends
        self.annual_financials = self.stock.financials
        self.share_outstanding = self.stock.info.get("sharesOutstanding", None)

    def get_latest_stock_price(self):
        """
        Retrieve the latest stock price using yfinance history.
        
        Returns:
            float or None: Latest closing price or None on error.
        """
        try:
            historical_data = self.stock.history(period="1d")
            latest_price = historical_data['Close'].iloc[-1]
            logger.info(f"Latest stock price for {self.stock_code}: {latest_price:.2f}")
            return latest_price
        except Exception as e:
            logger.error(f"Error fetching stock price for {self.stock_code}: {e}")
            return None

    def get_market_cap(self):
        """
        Calculate the market capitalization (Market Value of Equity).
        
        Returns:
            float or None: Market cap or None if data is missing.
        """
        try:
            latest_price = self.get_latest_stock_price()
            shares_outstanding = self.share_outstanding

            if latest_price is None or shares_outstanding is None:
                logger.error("Missing data for market capitalization calculation.")
                return None

            market_cap = latest_price * shares_outstanding
            logger.info(f"Market Cap for {self.stock_code}: {market_cap:.2f}")
            return market_cap
        except Exception as e:
            logger.error(f"Error calculating market cap: {e}")
            return None

    def get_beta(self):
        """
        Retrieve the stock's beta.
        
        Returns:
            float: Beta value (default is 1 if not available).
        """
        try:
            beta = self.stock.info.get("beta", None)
            if beta is not None:
                logger.info(f"Beta for {self.stock_code}: {beta:.2f}")
                return beta
            else:
                logger.info(f"Beta not available for {self.stock_code}. Using default beta = 1.")
                return 1
        except Exception as e:
            logger.error(f"Error fetching beta for {self.stock_code}: {e}")
            return 1

    def calculate_cost_of_equity_with_capm(self):
        """
        Calculate the cost of equity using CAPM.
        
        Returns:
            float: Cost of equity.
        """
        beta = self.get_beta()
        market_risk_premium = self.market_return - self.risk_free_rate
        cost_of_equity = self.risk_free_rate + beta * market_risk_premium
        logger.info(f"Cost of Equity (CAPM) for {self.stock_code}: {cost_of_equity:.2%}")
        return cost_of_equity

    def get_valid_annual_data(self, balance_sheet):
        """
        Retrieve valid annual data for Total Debt and Interest Expense.
        
        Args:
            balance_sheet (DataFrame): The balance sheet data.
        
        Returns:
            tuple: (total_debt, interest_expense) if available, otherwise (None, None).
        """
        try:
            if "Total Debt" not in balance_sheet.index or "Interest Expense" not in self.annual_financials.index:
                logger.error("Missing 'Total Debt' or 'Interest Expense' data.")
                return None, None

            total_debt_series = balance_sheet.loc["Total Debt"]
            interest_expense_series = self.annual_financials.loc["Interest Expense"]

            for year in sorted(total_debt_series.index, reverse=True):
                total_debt = total_debt_series[year]
                interest_expense = interest_expense_series.get(year, None)

                if pd.notna(total_debt) and pd.notna(interest_expense) and total_debt != 0 and interest_expense != 0:
                    interest_expense = abs(interest_expense)
                    logger.info(f"Using data from {year}: Total Debt = {total_debt}, Interest Expense = {interest_expense}")
                    return total_debt, interest_expense

            logger.error("No valid data found for Total Debt and Interest Expense.")
            return None, None

        except Exception as e:
            logger.error(f"Error processing annual data: {e}")
            return None, None

    def get_tax_data(self):
        """
        Calculate the effective tax rate.
        
        Returns:
            float or None: Tax rate or None if not available.
        """
        try:
            tax_provision_series = self.annual_financials.loc["Tax Provision"] if "Tax Provision" in self.annual_financials.index else None
            pretax_income_series = self.annual_financials.loc["Pretax Income"] if "Pretax Income" in self.annual_financials.index else None

            if tax_provision_series is None or pretax_income_series is None:
                logger.error("Missing Tax Provision or Pretax Income data.")
                return None

            for year in tax_provision_series.index:
                tax_provision = tax_provision_series[year]
                pretax_income = pretax_income_series[year]

                if pd.notna(tax_provision) and pd.notna(pretax_income) and pretax_income != 0:
                    tax_rate = tax_provision / pretax_income
                    logger.info(f"Using data from {year}: Tax Rate = {tax_rate:.2%}")
                    return tax_rate

            logger.error("No valid tax data found.")
            return None

        except Exception as e:
            logger.error(f"Error calculating tax data: {e}")
            return None

    def get_cost_of_debt(self):
        """
        Calculate the cost of debt and return it along with total debt.
        
        Returns:
            tuple: (after_tax_cost_of_debt, total_debt) or (None, None) if data is missing.
        """
        try:
            annual_balance_sheet = self.stock.balance_sheet
            total_debt, interest_expense = self.get_valid_annual_data(annual_balance_sheet)
            tax_rate = self.get_tax_data()

            if total_debt is not None and interest_expense is not None:
                cost_of_debt = abs(interest_expense) / total_debt
                logger.info(f"Cost of Debt (K_D): {cost_of_debt:.2%}")

                if tax_rate is not None:
                    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
                    logger.info(f"After-Tax Cost of Debt (K_D): {after_tax_cost_of_debt:.2%}")
                    return after_tax_cost_of_debt, total_debt
                else:
                    logger.info("Missing tax rate. Returning pre-tax cost of debt.")
                    return cost_of_debt, total_debt
            else:
                logger.error("Missing data to calculate cost of debt.")
                return None, None
        except Exception as e:
            logger.error(f"Error fetching cost of debt for {self.stock_code}: {e}")
            return None, None

    def calculate_weights(self):
        """
        Calculate the weights for debt and equity based on market cap and total debt.
        
        Returns:
            tuple: (debt_weight, equity_weight) or (None, None) if data is missing.
        """
        try:
            market_cap = self.get_market_cap()
            _, total_debt = self.get_cost_of_debt()

            if market_cap is None or total_debt is None or total_debt == 0:
                logger.error("Insufficient data to calculate weights.")
                return None, None

            total_value = total_debt + market_cap
            debt_weight = total_debt / total_value
            equity_weight = market_cap / total_value

            logger.info(f"Debt Weight: {debt_weight:.2%}, Equity Weight: {equity_weight:.2%}")
            return debt_weight, equity_weight
        except Exception as e:
            logger.error(f"Error calculating weights: {e}")
            return None, None

    def calculate_wacc(self):
        """
        Calculate the Weighted Average Cost of Capital (WACC).

        Returns:
            float or None: The WACC value or None if calculation fails.
        """
        cost_of_equity = self.calculate_cost_of_equity_with_capm()
        cost_of_debt, _ = self.get_cost_of_debt()
        debt_weight, equity_weight = self.calculate_weights()

        if cost_of_equity is None or cost_of_debt is None or debt_weight is None or equity_weight is None:
            logger.error("Cannot calculate WACC due to missing inputs.")
            return None

        wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt
        logger.info(f"Calculated WACC: {wacc:.2%}")
        return wacc


if __name__ == "__main__":
    wacc_model = WACCModel("8044.TWO")
    wacc_model.calculate_wacc()
