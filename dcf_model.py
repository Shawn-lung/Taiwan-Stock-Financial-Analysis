# dcf_model.py
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import Wacc  # Make sure you have a Wacc module with WACCModel implemented

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DCFModel:
    """
    A Discounted Cash Flow (DCF) Model for intrinsic stock valuation.

    Attributes:
        stock_code (str): Ticker symbol (e.g., "2330.TW").
        forecast_years (int): Number of years to forecast.
        perpetual_growth_rate (float): Growth rate used for terminal value calculation.
        manual_growth_rates (list): Annual revenue growth rates.
        manual_capex_factors (list): Annual CAPEX adjustment factors.
        manual_wc_factors (list): Annual working capital adjustment factors.
        manual_depr_factors (list): Annual depreciation adjustment factors.
        manual_opincome_factors (list): Annual operating income adjustment factors.
        manual_tax_factors (list): Annual tax rate adjustment factors.
    """

    def __init__(
        self,
        stock_code,
        forecast_years=5,
        perpetual_growth_rate=0.025,
        manual_growth_rates=None,
        manual_capex_factors=None,
        manual_wc_factors=None,
        manual_depr_factors=None,
        manual_opincome_factors=None,
        manual_tax_factors=None
    ):
        self.stock_code = stock_code
        self.forecast_years = forecast_years
        self.perpetual_growth_rate = perpetual_growth_rate
        self.manual_growth_rates = manual_growth_rates
        self.manual_capex_factors = manual_capex_factors
        self.manual_wc_factors = manual_wc_factors
        self.manual_depr_factors = manual_depr_factors
        self.manual_opincome_factors = manual_opincome_factors
        self.manual_tax_factors = manual_tax_factors

        # Initialize financial data placeholders
        self.stock = None
        self.income_stmt = None
        self.cash_flow = None
        self.balance_sheet = None
        self.latest_year_income = None
        self.latest_year_cf = None
        self.latest_year_bs = None

        # Base metrics
        self.current_revenue = 0.0
        self.operating_income = 0.0
        self.depreciation = 0.0
        self.capex = 0.0
        self.tax_rate = 0.20
        self.net_debt = 0.0
        self.current_working_capital = 0.0
        self.shares_outstanding = 1

        self.wacc = None
        self.base_year_metrics = {}
        self.hist_metrics = {}

        # Initialize financial data and compute base values
        self.initialize_model()

    def initialize_model(self):
        """Fetch financial data from yfinance and initialize base metrics."""
        try:
            self.stock = yf.Ticker(self.stock_code)
            self.income_stmt = self.stock.financials
            self.cash_flow = self.stock.cashflow
            self.balance_sheet = self.stock.balance_sheet

            # Sort columns by date and select the latest data
            if not self.income_stmt.empty:
                self.income_stmt = self.income_stmt.sort_index(axis=1, ascending=True)
                self.latest_year_income = self.income_stmt.columns[-1]
            if not self.cash_flow.empty:
                self.cash_flow = self.cash_flow.sort_index(axis=1, ascending=True)
                self.latest_year_cf = self.cash_flow.columns[-1]
            if not self.balance_sheet.empty:
                self.balance_sheet = self.balance_sheet.sort_index(axis=1, ascending=True)
                self.latest_year_bs = self.balance_sheet.columns[-1]

            # Get base values from the latest financial statements
            self.current_revenue = self.get_latest_revenue()
            self.operating_income = self.get_latest_operating_income()
            self.depreciation = self.get_latest_depreciation()
            self.capex = self.get_latest_capex()
            self.tax_rate = self.get_tax_rate_estimate()
            self.net_debt = self.get_net_debt()
            self.current_working_capital = self.get_latest_working_capital()

            # Get shares outstanding
            so = self.stock.info.get("sharesOutstanding", 1)
            self.shares_outstanding = so if so and so > 0 else 1

            # Calculate WACC using the external Wacc module
            self.wacc = Wacc.WACCModel(self.stock_code).calculate_wacc()

            # Store base year metrics for potential anomaly detection
            self.base_year_metrics = {
                "Revenue": self.current_revenue,
                "OperatingIncome": self.operating_income,
                "Depreciation": self.depreciation,
                "CAPEX": self.capex,
                "WorkingCapital": self.current_working_capital
            }
            self.hist_metrics = self.prepare_historical_metrics()

            logger.info("Initialization complete for stock: %s", self.stock_code)
        except Exception as e:
            logger.error("Error during initialization: %s", e)

    def get_latest_revenue(self):
        """Extract the latest revenue value from the income statement."""
        if self.latest_year_income is None:
            return 0.0
        try:
            for key in ["Total Revenue", "Revenue"]:
                if key in self.income_stmt.index:
                    val = self.income_stmt.loc[key, self.latest_year_income]
                    if pd.notna(val):
                        return float(val)
            return 0.0
        except Exception as e:
            logger.error("Error fetching latest revenue: %s", e)
            return 0.0

    def get_latest_operating_income(self):
        """Extract the latest operating income from the income statement."""
        if self.latest_year_income is None:
            return 0.0
        try:
            if "Operating Income" in self.income_stmt.index:
                val = self.income_stmt.loc["Operating Income", self.latest_year_income]
                return float(val) if pd.notna(val) else 0.0
            return 0.0
        except Exception as e:
            logger.error("Error fetching operating income: %s", e)
            return 0.0

    def get_latest_depreciation(self):
        """Extract the latest depreciation value from the cash flow statement."""
        if self.latest_year_cf is None:
            return 0.0
        try:
            for key in ["Depreciation", "Depreciation & Amortization"]:
                if key in self.cash_flow.index:
                    val = self.cash_flow.loc[key, self.latest_year_cf]
                    if pd.notna(val):
                        return abs(float(val))
            return 0.0
        except Exception as e:
            logger.error("Error fetching depreciation: %s", e)
            return 0.0

    def get_latest_capex(self):
        """Extract the latest CAPEX from the cash flow statement."""
        if self.latest_year_cf is None:
            return 0.0
        try:
            if "Capital Expenditure" in self.cash_flow.index:
                val = self.cash_flow.loc["Capital Expenditure", self.latest_year_cf]
                if pd.notna(val):
                    return abs(float(val))
            return 0.0
        except Exception as e:
            logger.error("Error fetching CAPEX: %s", e)
            return 0.0

    def get_tax_rate_estimate(self):
        """Estimate the tax rate based on income statement data."""
        if self.latest_year_income is None:
            return 0.20
        try:
            tax_keys = ["Income Tax Expense", "Tax Provision"]
            pretax_keys = ["Income Before Tax", "Pretax Income"]
            tax_val, pretax_val = None, None
            for tk in tax_keys:
                if tk in self.income_stmt.index:
                    val = self.income_stmt.loc[tk, self.latest_year_income]
                    if pd.notna(val):
                        tax_val = float(val)
                        break
            for pk in pretax_keys:
                if pk in self.income_stmt.index:
                    val = self.income_stmt.loc[pk, self.latest_year_income]
                    if pd.notna(val):
                        pretax_val = float(val)
                        break
            if tax_val is not None and pretax_val and pretax_val != 0:
                rate = abs(tax_val / pretax_val)
                return rate if 0 < rate < 0.5 else 0.20
            return 0.20
        except Exception as e:
            logger.error("Error estimating tax rate: %s", e)
            return 0.20

    def get_net_debt(self):
        """Calculate net debt from the balance sheet."""
        if self.latest_year_bs is None:
            return 0.0
        try:
            total_debt = 0.0
            cash_val = 0.0
            if "Total Debt" in self.balance_sheet.index:
                td = self.balance_sheet.loc["Total Debt", self.latest_year_bs]
                if pd.notna(td):
                    total_debt = float(td)
            for key in ["Cash And Cash Equivalents", "Cash", "Cash Cash Equivalents And Short Term Investments"]:
                if key in self.balance_sheet.index:
                    c_val = self.balance_sheet.loc[key, self.latest_year_bs]
                    if pd.notna(c_val):
                        cash_val += float(c_val)
                        break
            return total_debt - cash_val
        except Exception as e:
            logger.error("Error calculating net debt: %s", e)
            return 0.0

    def get_latest_working_capital(self):
        """Calculate the latest working capital from the balance sheet."""
        if self.latest_year_bs is None:
            return 0.0
        try:
            current_assets = 0.0
            current_liabilities = 0.0
            if "Current Assets" in self.balance_sheet.index:
                ca = self.balance_sheet.loc["Current Assets", self.latest_year_bs]
                if pd.notna(ca):
                    current_assets = float(ca)
            if "Current Liabilities" in self.balance_sheet.index:
                cl = self.balance_sheet.loc["Current Liabilities", self.latest_year_bs]
                if pd.notna(cl):
                    current_liabilities = float(cl)
            return current_assets - current_liabilities
        except Exception as e:
            logger.error("Error calculating working capital: %s", e)
            return 0.0

    def prepare_historical_metrics(self):
        """Prepare historical financial metrics for analysis."""
        result = {
            "Revenue": [],
            "OperatingIncome": [],
            "Depreciation": [],
            "CAPEX": [],
            "WorkingCapital": []
        }
        if self.income_stmt.empty:
            return result
        cols = sorted(self.income_stmt.columns)
        past_cols = cols[:-1] if len(cols) > 1 else []
        for col in past_cols:
            rev = 0.0
            for key in ["Total Revenue", "Revenue"]:
                if key in self.income_stmt.index:
                    tmp = self.income_stmt.loc[key, col]
                    if pd.notna(tmp):
                        rev = float(tmp)
                        break
            op = 0.0
            if "Operating Income" in self.income_stmt.index:
                tmpop = self.income_stmt.loc["Operating Income", col]
                if pd.notna(tmpop):
                    op = float(tmpop)
            dep = 0.0
            if not self.cash_flow.empty and col in self.cash_flow.columns:
                for key in ["Depreciation", "Depreciation & Amortization"]:
                    if key in self.cash_flow.index:
                        d_val = self.cash_flow.loc[key, col]
                        if pd.notna(d_val):
                            dep = abs(float(d_val))
                            break
            cap = 0.0
            if not self.cash_flow.empty and col in self.cash_flow.columns:
                if "Capital Expenditure" in self.cash_flow.index:
                    c_val = self.cash_flow.loc["Capital Expenditure", col]
                    if pd.notna(c_val):
                        cap = abs(float(c_val))
            wc = 0.0
            if not self.balance_sheet.empty and col in self.balance_sheet.columns:
                ca = 0.0
                cl = 0.0
                if "Current Assets" in self.balance_sheet.index:
                    ca_val = self.balance_sheet.loc["Current Assets", col]
                    if pd.notna(ca_val):
                        ca = float(ca_val)
                if "Current Liabilities" in self.balance_sheet.index:
                    cl_val = self.balance_sheet.loc["Current Liabilities", col]
                    if pd.notna(cl_val):
                        cl = float(cl_val)
                wc = ca - cl
            result["Revenue"].append(rev)
            result["OperatingIncome"].append(op)
            result["Depreciation"].append(dep)
            result["CAPEX"].append(cap)
            result["WorkingCapital"].append(wc)
        return result

    def estimate_historical_revenue_growth(self, years=3):
        """Estimate the Compound Annual Growth Rate (CAGR) for revenue over a specified period."""
        if self.income_stmt.empty:
            return None
        cols_sorted = sorted(self.income_stmt.columns)
        if len(cols_sorted) < years + 1:
            return None
        rev_key = None
        for key in ["Total Revenue", "Revenue"]:
            if key in self.income_stmt.index:
                rev_key = key
                break
        if not rev_key:
            return None
        older_col = cols_sorted[-(years + 1)]
        newer_col = cols_sorted[-1]
        older_rev = self.income_stmt.loc[rev_key, older_col]
        newer_rev = self.income_stmt.loc[rev_key, newer_col]
        if pd.isna(older_rev) or pd.isna(newer_rev) or older_rev <= 0:
            return None
        cagr = (newer_rev / older_rev) ** (1 / years) - 1
        return cagr if cagr >= -1 else None

    def forecast_fcf_list(self):
        """Forecast Free Cash Flow with improved calculations."""
        try:
            fcf_list = []
            revenue = float(self.current_revenue)
            operating_margin = self.operating_income / revenue
            
            # Normalize base metrics
            base_capex_to_revenue = self.capex / revenue
            base_depr_to_revenue = self.depreciation / revenue
            base_wc_to_revenue = self.current_working_capital / revenue
            
            for i in range(self.forecast_years):
                # Apply growth rate
                growth = self.manual_growth_rates[i] if self.manual_growth_rates else 0.05
                new_revenue = revenue * (1 + growth)
                
                # Operating income with stable margin
                op_income = new_revenue * operating_margin
                
                # CAPEX and depreciation based on revenue
                capex = new_revenue * base_capex_to_revenue * (1 + (self.manual_capex_factors[i] if self.manual_capex_factors else 0))
                depr = new_revenue * base_depr_to_revenue * (1 + (self.manual_depr_factors[i] if self.manual_depr_factors else 0))
                
                # Working capital changes
                new_wc = new_revenue * base_wc_to_revenue
                delta_wc = new_wc - (revenue * base_wc_to_revenue)
                
                # Calculate FCF
                tax_rate = min(max(self.tax_rate * (1 + (self.manual_tax_factors[i] if self.manual_tax_factors else 0)), 0.15), 0.35)
                nopat = op_income * (1 - tax_rate)
                fcf = nopat + depr - capex - delta_wc
                
                # Ensure FCF is reasonable
                max_fcf = new_revenue * 0.3  # Cap FCF at 30% of revenue
                min_fcf = new_revenue * 0.05  # Minimum FCF at 5% of revenue
                fcf = min(max(fcf, min_fcf), max_fcf)
                
                fcf_list.append(float(fcf))
                revenue = new_revenue  # Update for next iteration
                
            logger.info("FCF Projections:")
            for i, fcf in enumerate(fcf_list, 1):
                logger.info(f"Year {i}: {fcf:,.2f}")
                
            return fcf_list
            
        except Exception as e:
            logger.error(f"Error in FCF forecast: {e}")
            return [self.operating_income * 0.1] * self.forecast_years

    def calculate_intrinsic_value(self):
        """Calculate intrinsic value with improved error handling."""
        try:
            if not self.wacc or self.wacc <= self.perpetual_growth_rate:
                logger.error(f"Invalid WACC ({self.wacc}) or growth rate ({self.perpetual_growth_rate})")
                return None

            fcf_list = self.forecast_fcf_list()
            if not fcf_list:
                return None

            # Calculate present value of FCFs
            npv_stage_1 = sum(
                fcf / (1 + self.wacc) ** (i+1)
                for i, fcf in enumerate(fcf_list)
            )

            # Calculate terminal value using normalized FCF
            normalized_fcf = sum(fcf_list[-3:]) / 3  # Average of last 3 years
            terminal_growth = min(self.perpetual_growth_rate, self.wacc - 0.02)
            terminal_value = normalized_fcf * (1 + terminal_growth) / (self.wacc - terminal_growth)
            
            # Discount terminal value
            discounted_tv = terminal_value / (1 + self.wacc) ** len(fcf_list)

            enterprise_value = npv_stage_1 + discounted_tv
            equity_value = enterprise_value + (self.stock.info.get('cash', 0) - self.net_debt)

            # Log calculations
            logger.info("----- DCF Calculation Details -----")
            logger.info(f"Base Revenue: {self.current_revenue:,.2f}")
            logger.info(f"Operating Income: {self.operating_income:,.2f}")
            logger.info(f"WACC: {self.wacc:.2%}")
            logger.info(f"Terminal Growth: {terminal_growth:.2%}")
            logger.info(f"NPV of FCF: {npv_stage_1:,.2f}")
            logger.info(f"Terminal Value: {terminal_value:,.2f}")
            logger.info(f"Discounted TV: {discounted_tv:,.2f}")
            logger.info(f"Net Debt: {self.net_debt:,.2f}")
            logger.info(f"Enterprise Value: {enterprise_value:,.2f}")
            logger.info(f"Equity Value: {equity_value:,.2f}")

            return equity_value

        except Exception as e:
            logger.error(f"Error calculating intrinsic value: {e}")
            return None

    def calculate_stock_price(self):
        """
        Calculate the fair stock price by dividing the intrinsic equity value by the number of shares outstanding.
        """
        eq_value = self.calculate_intrinsic_value()
        if eq_value is None:
            return None
        if self.shares_outstanding <= 0:
            logger.error("Invalid shares outstanding.")
            return None
        fair_price = eq_value / self.shares_outstanding
        logger.info("Calculated Fair Price: %.2f per share", fair_price)
        return fair_price

    def perform_sensitivity_analysis(self, wacc_range=0.02, growth_range=0.01):
        """
        Perform sensitivity analysis on WACC and perpetual growth rate.
        
        Args:
            wacc_range (float): Range to vary WACC (+/-)
            growth_range (float): Range to vary perpetual growth rate (+/-)
        
        Returns:
            dict: Matrix of stock prices under different scenarios
        """
        base_wacc = self.wacc
        base_growth = self.perpetual_growth_rate
        
        wacc_values = np.linspace(base_wacc - wacc_range, base_wacc + wacc_range, 5)
        growth_values = np.linspace(base_growth - growth_range, base_growth + growth_range, 5)
        
        sensitivity_matrix = {}
        
        for w in wacc_values:
            row = {}
            self.wacc = w
            for g in growth_values:
                self.perpetual_growth_rate = g
                price = self.calculate_stock_price()
                row[f"{g:.1%}"] = price if price is not None else 0
            sensitivity_matrix[f"{w:.1%}"] = row
        
        self.wacc = base_wacc
        self.perpetual_growth_rate = base_growth
        
        return sensitivity_matrix

    def perform_comprehensive_sensitivity(self, variation=0.2):
        """
        Perform comprehensive sensitivity analysis on all major inputs.
        
        Args:
            variation (float): Percentage variation to test (+/-)
        
        Returns:
            dict: Impact of each factor on stock price
        """
        base_price = self.calculate_stock_price()
        if base_price is None:
            return None
            
        results = {}
        
        # Test revenue growth sensitivity
        original_growth = self.manual_growth_rates[:]
        high_growth = [g * (1 + variation) for g in original_growth]
        low_growth = [g * (1 - variation) for g in original_growth]
        
        self.manual_growth_rates = high_growth
        high_price = self.calculate_stock_price()
        self.manual_growth_rates = low_growth
        low_price = self.calculate_stock_price()
        self.manual_growth_rates = original_growth
        
        results['Revenue Growth'] = {
            'high': (high_price - base_price) / base_price,
            'low': (low_price - base_price) / base_price
        }
        
        # Test operating margins through operating income factors
        if self.operating_income != 0:
            base_op = self.operating_income
            self.operating_income = base_op * (1 + variation)
            high_price = self.calculate_stock_price()
            self.operating_income = base_op * (1 - variation)
            low_price = self.calculate_stock_price()
            self.operating_income = base_op
            
            results['Operating Margin'] = {
                'high': (high_price - base_price) / base_price,
                'low': (low_price - base_price) / base_price
            }
        
        # Test CAPEX sensitivity
        original_capex = self.manual_capex_factors[:]
        high_capex = [c * (1 + variation) for c in original_capex]
        low_capex = [c * (1 - variation) for c in original_capex]
        
        self.manual_capex_factors = high_capex
        high_price = self.calculate_stock_price()
        self.manual_capex_factors = low_capex
        low_price = self.calculate_stock_price()
        self.manual_capex_factors = original_capex
        
        results['CAPEX'] = {
            'high': (high_price - base_price) / base_price,
            'low': (low_price - base_price) / base_price
        }
        
        # Test Working Capital sensitivity
        original_wc = self.manual_wc_factors[:]
        high_wc = [w * (1 + variation) for w in original_wc]
        low_wc = [w * (1 - variation) for w in original_wc]
        
        self.manual_wc_factors = high_wc
        high_price = self.calculate_stock_price()
        self.manual_wc_factors = low_wc
        low_price = self.calculate_stock_price()
        self.manual_wc_factors = original_wc
        
        results['Working Capital'] = {
            'high': (high_price - base_price) / base_price,
            'low': (low_price - base_price) / base_price
        }

        return results

if __name__ == "__main__":
    dcf = DCFModel(
        "2330.TW",
        forecast_years=5,
        perpetual_growth_rate=0.025,
        manual_growth_rates=[0.2, 0.1, 0.1, 0.05, 0.05],
        manual_capex_factors=[0.2, 0.1, 0, -0.05, -0.05],
        manual_wc_factors=[0.1, 0.05, 0, 0, 0],
        manual_depr_factors=[0, 0.05, 0.05, 0, -0.1],
        manual_tax_factors=[0, 0, 0, 0.05, -0.1]
    )
    price = dcf.calculate_stock_price()
    if price is not None:
        print(f"Estimated Stock Price: {price:.2f}")
    else:
        print("Could not calculate stock price.")
